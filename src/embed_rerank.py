"""
embed_rerank.py  (RUN IN COLAB — open-source, local, NO external API)
---------------------------------------------------------------------
Semantic reranker for the exact_fact extraction stage, using a sentence-
embedding model (sentence-transformers) that downloads once and runs locally.
This addresses the measured bottleneck: the TF-IDF extractor plateaus at
~0.65 accuracy because it only matches words, while ~21 points of headroom
(oracle ceiling 0.863) require recognizing paraphrases.

How it works:
  1. Embed all gold explanation sentences from TRAINING data, grouped by section.
  2. For a test fact, embed each candidate (sentences + 2-sentence spans).
  3. Score each candidate = max cosine similarity to that section's gold pool.
  4. Blend with the v2 classifier probability: score = (1-w)*clf + w*embed.

Includes a --cv mode that measures extraction accuracy with the SAME protocol
used throughout (5-fold, seed 13), so the adopt/reject decision is objective:
  ADOPT only if CV extraction accuracy > 0.655 (the verified v2 number).

NOT RUN/VERIFIED in the sandbox (no downloads there). Syntax-checked only.

Colab usage:
    !pip install -q sentence-transformers
    # honest check first:
    !python embed_rerank.py --train task1.jsonl --cv
    # if it wins, produce a refined submission:
    !python embed_rerank.py --train task1.jsonl --test test.jsonl \
        --submission submission_tier1.jsonl --out submission_tier1_reranked.jsonl
"""
import argparse
import json
import numpy as np
from collections import defaultdict

from esp_utils import read_jsonl, write_jsonl, normalize_section, norm_ws, split_sentences
from esp_pipeline_v2 import ESPModelV2

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # strong general model
# Alternative worth trying if this underperforms: "law-ai/InLegalBERT" via
# mean pooling, or "sentence-transformers/all-MiniLM-L6-v2" (faster).
BLEND_W = 0.5   # weight of embedding similarity vs classifier probability


def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def build_gold_pools(train_rows, encoder):
    pools = defaultdict(list)
    for r in train_rows:
        for sent, sec in r.get("explanation", {}).items():
            pools[normalize_section(sec)].append(sent)
    emb = {}
    for sec, sents in pools.items():
        E = encoder.encode(sents, normalize_embeddings=True,
                           show_progress_bar=False)
        emb[sec] = np.asarray(E)
    return emb


def rerank_pick(model, encoder, pools, fact, section, blend_w=BLEND_W):
    """Return the best exact_fact for `section` using clf prob + embed sim."""
    sents = split_sentences(fact)
    if not sents:
        return ""
    cands = list(sents)
    if len(sents) > 1:
        cands += [sents[i] + " " + sents[i + 1] for i in range(len(sents) - 1)]

    # classifier probabilities (reuse v2's featurizers/classifier)
    from scipy.sparse import hstack
    Xs = hstack([model.sent_vw.transform(cands),
                 model.sent_vc.transform(cands)]).tocsr()
    if model.sent_clf is not None and section in model.sent_classes:
        col = model.sent_classes.index(section)
        clf_p = model.sent_clf.predict_proba(Xs)[:, col]
    else:
        clf_p = np.zeros(len(cands))

    # embedding similarity to the section's gold pool
    if section in pools:
        C = encoder.encode(cands, normalize_embeddings=True,
                           show_progress_bar=False)
        sim = (np.asarray(C) @ pools[section].T).max(axis=1)
        sim = (sim + 1) / 2  # map cosine [-1,1] -> [0,1] to match prob scale
    else:
        sim = np.zeros(len(cands))

    score = (1 - blend_w) * clf_p + blend_w * sim
    return cands[int(np.argmax(score))]


def extraction_match(pred, gold_keys):
    p = norm_ws(pred)
    return any(p == norm_ws(k) or p in norm_ws(k) or norm_ws(k) in p
               for k in gold_keys)


def run_cv(rows, encoder, blend_w):
    n = len(rows)
    rng = np.random.default_rng(13)
    idx = rng.permutation(n)
    folds = [idx[i::5] for i in range(5)]
    accs = []
    for fi, te in enumerate(folds):
        ts = set(te.tolist())
        train = [rows[i] for i in range(n) if i not in ts]
        test = [rows[i] for i in range(n) if i in ts]
        model = ESPModelV2().fit(train)
        pools = build_gold_pools(train, encoder)
        hit = tot = 0
        for r in test:
            gold = defaultdict(list)
            for s, sec in r["explanation"].items():
                gold[normalize_section(sec)].append(s)
            for sec in {normalize_section(s) for s in r["statute"]}:
                if sec not in gold:
                    continue
                tot += 1
                pick = rerank_pick(model, encoder, pools, r["fact"], sec,
                                   blend_w)
                hit += extraction_match(pick, gold[sec])
        acc = hit / tot
        accs.append(acc)
        print(f"fold {fi+1}: extraction acc = {acc:.3f}")
    print(f"\nmean extraction accuracy (embed rerank, w={blend_w}): "
          f"{np.mean(accs):.3f}")
    print("Baseline to beat (v2 verified): 0.655. "
          "Oracle ceiling with these candidates: 0.863.")
    print("ADOPT only if clearly above 0.655; also try --blend_w 0.3 / 0.7.")


def refine_submission(rows_train, sub_path, test_path, out_path, encoder,
                      blend_w):
    model = ESPModelV2().fit(rows_train)
    pools = build_gold_pools(rows_train, encoder)
    facts = {r["doc_id"]: r["fact"] for r in read_jsonl(test_path)}
    sub = read_jsonl(sub_path)
    changed = 0
    for row in sub:
        fact = facts.get(row["doc_id"])
        if not fact:
            continue
        for e in row["statute"]:
            new = rerank_pick(model, encoder, pools, fact, e["section"],
                              blend_w)
            if new and new != e["exact_fact"]:
                # keep reasoning consistent with the new evidence sentence
                from ipc_kb import build_reasoning
                e["exact_fact"] = new
                e["reasoning_trace"] = build_reasoning(e["section"], new)
                changed += 1
    write_jsonl(sub, out_path)
    print(f"exact_fact updated in {changed} entries -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="task1.jsonl")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--test", default=None)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--out", default="submission_reranked.jsonl")
    ap.add_argument("--blend_w", type=float, default=BLEND_W)
    args = ap.parse_args()

    rows = read_jsonl(args.train)
    encoder = load_encoder()

    if args.cv:
        run_cv(rows, encoder, args.blend_w)
        return

    assert args.test and args.submission, \
        "--test and --submission required when not in --cv mode"
    refine_submission(rows, args.submission, args.test, args.out, encoder,
                      args.blend_w)


if __name__ == "__main__":
    main()
