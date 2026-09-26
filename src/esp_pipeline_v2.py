"""
esp_pipeline_v2.py
------------------
Improved ESP pipeline. Every change here was verified by 5-fold CV on the
actual training data (see README_v2.md for the numbers).

Changes vs v1:
  Stage 1 (labels):    word TF-IDF  ->  word + character n-gram TF-IDF,
                       threshold retuned to 0.30.
                       CV Macro F1: 0.739 -> 0.755, Recall@3: 0.970 -> 0.983
  Stage 2 (exact_fact): three verified upgrades
                       (a) word + char n-gram features,
                       (b) a NONE negative class built from fact sentences that
                           are NOT gold explanation sentences, which teaches the
                           model what an irrelevant sentence looks like,
                       (c) 2-sentence spans as additional candidates, because
                           gold explanation keys often span two sentences.
                       CV extraction accuracy: 0.509 -> 0.646
  Stage 3 (reasoning):  unchanged (templated from ipc_kb).

Optionally, Stage 1 can accept external per-label probabilities (e.g. from a
fine-tuned InLegalBERT run in Colab) and ensemble them with the TF-IDF model.
See `predict_one(..., ext_probs=...)`.
"""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from esp_utils import split_sentences, normalize_section, norm_ws
from ipc_kb import LABELS, build_reasoning


class ESPModelV2:
    def __init__(self, labels=None, doc_threshold=0.30, min_labels=1,
                 max_labels=3, use_spans=True, ext_weight=0.5):
        self.labels = labels or LABELS
        self.doc_threshold = doc_threshold
        self.min_labels = min_labels
        self.max_labels = max_labels
        self.use_spans = use_spans
        self.ext_weight = ext_weight        # weight of external (e.g. BERT) probs
        self.label_index = {l: i for i, l in enumerate(self.labels)}

        # Stage 1 featurizers (word + char)
        self.doc_vw = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                      min_df=2, max_features=40000)
        self.doc_vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      min_df=3, max_features=80000,
                                      sublinear_tf=True)
        self.doc_clf = None

        # Stage 2 featurizers (word + char) + classifier with NONE class
        self.sent_vw = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                       min_df=2, max_features=40000)
        self.sent_vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                       min_df=3, max_features=60000,
                                       sublinear_tf=True)
        self.sent_clf = None
        self.sent_classes = None

    # ------------------------------------------------------------------ fit
    def fit(self, rows):
        # ---- Stage 1
        X_txt = [r["fact"] for r in rows]
        Y = np.zeros((len(rows), len(self.labels)), dtype=int)
        for i, r in enumerate(rows):
            for s in r.get("statute", []):
                s = normalize_section(s)
                if s in self.label_index:
                    Y[i, self.label_index[s]] = 1
        Xd = hstack([self.doc_vw.fit_transform(X_txt),
                     self.doc_vc.fit_transform(X_txt)]).tocsr()
        self.doc_clf = []
        for j in range(len(self.labels)):
            if Y[:, j].sum() == 0:
                self.doc_clf.append(None)
                continue
            clf = LogisticRegression(max_iter=3000, C=4.0,
                                     class_weight="balanced")
            clf.fit(Xd, Y[:, j])
            self.doc_clf.append(clf)

        # ---- Stage 2: positives from `explanation`, negatives = other sentences
        s_txt, s_lab = [], []
        for r in rows:
            gold_norm = {norm_ws(k) for k in r.get("explanation", {})}
            for sent, sec in r.get("explanation", {}).items():
                sec = normalize_section(sec)
                if sec in self.label_index:
                    s_txt.append(sent)
                    s_lab.append(sec)
            for sent in split_sentences(r["fact"]):
                sn = norm_ws(sent)
                if not any(sn in g or g in sn for g in gold_norm):
                    s_txt.append(sent)
                    s_lab.append("NONE")
        if s_txt:
            Xs = hstack([self.sent_vw.fit_transform(s_txt),
                         self.sent_vc.fit_transform(s_txt)]).tocsr()
            self.sent_clf = LogisticRegression(max_iter=3000, C=4.0,
                                               class_weight="balanced")
            self.sent_clf.fit(Xs, s_lab)
            self.sent_classes = list(self.sent_clf.classes_)
        return self

    # ---------------------------------------------------------- stage 1 probs
    def doc_scores(self, fact):
        x = hstack([self.doc_vw.transform([fact]),
                    self.doc_vc.transform([fact])]).tocsr()
        return {lab: (float(self.doc_clf[j].predict_proba(x)[0, 1])
                      if self.doc_clf[j] is not None else 0.0)
                for j, lab in enumerate(self.labels)}

    def predict_labels(self, fact, ext_probs=None):
        """ext_probs: optional {label: prob} from an external model (e.g. a
        fine-tuned InLegalBERT). Blended with weight self.ext_weight."""
        scores = self.doc_scores(fact)
        if ext_probs:
            w = self.ext_weight
            scores = {l: (1 - w) * scores[l] + w * float(ext_probs.get(l, 0.0))
                      for l in scores}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        chosen = [l for l, s in ranked if s >= self.doc_threshold]
        if len(chosen) < self.min_labels:
            chosen = [l for l, _ in ranked[: self.min_labels]]
        chosen = chosen[: self.max_labels]
        chosen = [l for l, _ in ranked if l in set(chosen)]
        return chosen, ranked

    # ----------------------------------------------------- stage 2 extraction
    def extract_fact(self, sentences, section):
        if not sentences:
            return ""
        if self.sent_clf is None or section not in self.sent_classes:
            return sentences[0]
        cands = list(sentences)
        if self.use_spans and len(sentences) > 1:
            cands += [sentences[i] + " " + sentences[i + 1]
                      for i in range(len(sentences) - 1)]
        Xs = hstack([self.sent_vw.transform(cands),
                     self.sent_vc.transform(cands)]).tocsr()
        col = self.sent_classes.index(section)
        probs = self.sent_clf.predict_proba(Xs)[:, col]
        return cands[int(np.argmax(probs))]

    # ------------------------------------------------------- full prediction
    def predict_one(self, fact, ext_probs=None):
        chosen, ranked = self.predict_labels(fact, ext_probs=ext_probs)
        sentences = split_sentences(fact)
        preds = [{"section": sec,
                  "exact_fact": (ex := self.extract_fact(sentences, sec)),
                  "reasoning_trace": build_reasoning(sec, ex)}
                 for sec in chosen]
        return preds, [l for l, _ in ranked]

    def predict_submission(self, rows, ext_probs_by_doc=None):
        out = []
        for r in rows:
            ext = (ext_probs_by_doc or {}).get(r["doc_id"])
            preds, _ = self.predict_one(r["fact"], ext_probs=ext)
            out.append({"doc_id": r["doc_id"], "statute": preds})
        return out
