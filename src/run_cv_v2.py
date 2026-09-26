"""
run_cv_v2.py
------------
5-fold CV for the improved pipeline. Reports the composite components AND the
extraction accuracy (does the picked exact_fact match a gold explanation
sentence for that section?), which the v1 harness did not measure.

Usage:
    python run_cv_v2.py --data task1.jsonl --folds 5
"""
import argparse
import numpy as np
from collections import defaultdict

from esp_utils import read_jsonl, normalize_section, norm_ws
from esp_pipeline_v2 import ESPModelV2
from ipc_kb import LABELS
import esp_metrics as M


def extraction_match(pred_sent, gold_keys):
    p = norm_ws(pred_sent)
    for k in gold_keys:
        kn = norm_ws(k)
        if p == kn or p in kn or kn in p:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="task1.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    n = len(rows)
    rng = np.random.default_rng(13)
    idx = rng.permutation(n)
    folds = [idx[i::args.folds] for i in range(args.folds)]

    agg = defaultdict(list)
    for fi, test_idx in enumerate(folds):
        ts = set(test_idx.tolist())
        train = [rows[i] for i in range(n) if i not in ts]
        test = [rows[i] for i in range(n) if i in ts]

        model = ESPModelV2(doc_threshold=args.threshold).fit(train)

        gold_sets, pred_sets, pred_ranked = [], [], []
        rouge_v, bleu_v, lss_v = [], [], []
        ex_hit = ex_tot = 0

        for r in test:
            gold = {normalize_section(s) for s in r.get("statute", [])}
            preds, ranked = model.predict_one(r["fact"])
            pset = {p["section"] for p in preds}
            gold_sets.append(gold)
            pred_sets.append(pset)
            pred_ranked.append(ranked)

            gold_expl = defaultdict(list)
            for sent, sec in r.get("explanation", {}).items():
                gold_expl[normalize_section(sec)].append(sent)

            for p in preds:
                sec = p["section"]
                if sec in gold_expl:
                    ex_tot += 1
                    ex_hit += extraction_match(p["exact_fact"], gold_expl[sec])
                    ref = " ".join(gold_expl[sec])
                    rouge_v.append(M.rouge_l(p["reasoning_trace"], ref))
                    bleu_v.append(M.bleu(p["reasoning_trace"], ref))
                    lss_v.append(M.lss_proxy(p["reasoning_trace"], ref))

        f1 = M.macro_f1(gold_sets, pred_sets, LABELS)
        r3 = M.recall_at_k(gold_sets, pred_ranked, 3)
        ro = float(np.mean(rouge_v)) if rouge_v else 0.0
        bl = float(np.mean(bleu_v)) if bleu_v else 0.0
        ls = float(np.mean(lss_v)) if lss_v else 0.0
        ex = ex_hit / ex_tot if ex_tot else 0.0
        comp = M.composite(f1, ro, bl, r3, ls)

        for k, v in [("macro_f1", f1), ("recall@3", r3), ("extraction_acc", ex),
                     ("rouge_l", ro), ("bleu", bl), ("lss(proxy)", ls),
                     ("composite", comp)]:
            agg[k].append(v)
        print(f"fold {fi+1}: F1={f1:.3f} R@3={r3:.3f} EXTR={ex:.3f} "
              f"ROUGE-L={ro:.3f} BLEU={bl:.3f} LSS={ls:.3f} COMP={comp:.3f}")

    print("\n=== mean over folds (v2) ===")
    for k in ["macro_f1", "recall@3", "extraction_acc", "rouge_l", "bleu",
              "lss(proxy)", "composite"]:
        print(f"  {k:15s} {np.mean(agg[k]):.3f}  (+/- {np.std(agg[k]):.3f})")
    print("\nextraction_acc is EXACT (vs gold explanation sentences); "
          "text metrics remain proxy-based.")


if __name__ == "__main__":
    main()
