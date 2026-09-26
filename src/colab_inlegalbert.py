"""
colab_inlegalbert.py  (RUN IN GOOGLE COLAB WITH A GPU)
------------------------------------------------------
Fine-tunes law-ai/InLegalBERT as a multi-label classifier on the ESP training
set, then writes bert_probs.json: {doc_id: {label: probability}} for the test
set. Feed that file to predict_test_v2.py via --bert_probs to ensemble it with
the TF-IDF model.

NOT RUN/VERIFIED in the sandbox (no internet/GPU there) — written for Colab.
Before trusting it, VALIDATE THE ENSEMBLE with cross-validation (see
"CV mode" below): only use --bert_probs in your final submission if CV shows
it actually beats the TF-IDF-only F1 of ~0.755. On 525 examples a fine-tuned
BERT is not guaranteed to win; you have ONE submission.

Colab setup (Runtime -> Change runtime type -> T4 GPU):
    !pip install -q transformers datasets torch scikit-learn
    # upload task1.jsonl (and test.jsonl when available), then:
    !python colab_inlegalbert.py --train task1.jsonl --test test.jsonl --out bert_probs.json
    # CV mode (no test file needed): estimates ensemble gain honestly
    !python colab_inlegalbert.py --train task1.jsonl --cv
Runtime: roughly 10-20 min on a T4 for 5 epochs.
"""
import argparse
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["IPC 302", "IPC 498A", "IPC 376", "IPC 420",
          "IPC 147", "IPC 506", "IPC 201"]
L2I = {l: i for i, l in enumerate(LABELS)}
MODEL_NAME = "law-ai/InLegalBERT"
MAX_LEN = 512          # facts average ~2.5k chars; 512 wordpieces covers most
EPOCHS = 5
BATCH = 8
LR = 2e-5


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class ESPDataset(Dataset):
    def __init__(self, rows, tok, with_labels=True):
        self.rows = rows
        self.tok = tok
        self.with_labels = with_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["fact"], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.with_labels:
            y = torch.zeros(len(LABELS))
            for s in r.get("statute", []):
                s = s.upper().replace("-", "").replace("  ", " ")
                if s in L2I:
                    y[L2I[s]] = 1.0
            item["labels"] = y
        return item


def train_model(train_rows, device):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        problem_type="multi_label_classification").to(device)
    dl = DataLoader(ESPDataset(train_rows, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += out.loss.item()
        print(f"epoch {ep+1}/{EPOCHS} loss={tot/len(dl):.4f}")
    return model, tok


@torch.no_grad()
def predict_probs(model, tok, rows, device):
    model.eval()
    dl = DataLoader(ESPDataset(rows, tok, with_labels=False), batch_size=BATCH)
    probs = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs.append(torch.sigmoid(logits).cpu().numpy())
    P = np.vstack(probs)
    return {rows[i]["doc_id"]: {LABELS[j]: float(P[i, j])
                                for j in range(len(LABELS))}
            for i in range(len(rows))}


def macro_f1(gold_sets, pred_sets):
    f1s = []
    for lab in LABELS:
        tp = fp = fn = 0
        for g, p in zip(gold_sets, pred_sets):
            if lab in g and lab in p: tp += 1
            elif lab in p: fp += 1
            elif lab in g: fn += 1
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return sum(f1s) / len(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="task1.jsonl")
    ap.add_argument("--test", default=None)
    ap.add_argument("--out", default="bert_probs.json")
    ap.add_argument("--cv", action="store_true",
                    help="5-fold CV of BERT alone (honest estimate; slower)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    train_rows = read_jsonl(args.train)

    if args.cv:
        rng = np.random.default_rng(13)
        idx = rng.permutation(len(train_rows))
        folds = [idx[i::5] for i in range(5)]
        f1s = []
        for fi, te in enumerate(folds):
            ts = set(te.tolist())
            tr = [train_rows[i] for i in range(len(train_rows)) if i not in ts]
            va = [train_rows[i] for i in range(len(train_rows)) if i in ts]
            model, tok = train_model(tr, device)
            probs = predict_probs(model, tok, va, device)
            gold = [{s for s in r["statute"]} for r in va]
            pred = [{l for l, p in probs[r["doc_id"]].items() if p >= 0.5}
                    or {max(probs[r["doc_id"]], key=probs[r["doc_id"]].get)}
                    for r in va]
            f1 = macro_f1(gold, pred)
            f1s.append(f1)
            print(f"fold {fi+1}: BERT-alone macro F1 = {f1:.3f}")
        print(f"\nmean BERT-alone macro F1: {np.mean(f1s):.3f}")
        print("Compare against TF-IDF-only 0.755. Only use the ensemble in your "
              "final run if the blended CV (run locally) is clearly higher.")
        return

    model, tok = train_model(train_rows, device)
    assert args.test, "--test required when not in --cv mode"
    test_rows = read_jsonl(args.test)
    probs = predict_probs(model, tok, test_rows, device)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(probs, f)
    print(f"wrote probabilities for {len(probs)} docs -> {args.out}")
    print("Next: python predict_test_v2.py --train task1.jsonl --test test.jsonl "
          f"--out submission.jsonl --bert_probs {args.out}")


if __name__ == "__main__":
    main()
