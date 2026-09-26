"""
esp_metrics.py
--------------
Local re-implementation of the ESP composite metric so you can cross-validate
BEFORE the CodaBench scorer / test set is released.

Weights (from the task page; tentative):
    Macro F1   35%   exact match on section labels
    ROUGE-L    25%   reasoning text overlap (LCS F-measure)
    BLEU       20%   reasoning text overlap (n-gram)
    Recall@3   10%   gold labels present in top-3 predictions
    LSS        10%   semantic similarity of reasoning (undisclosed legal model)

IMPORTANT CAVEATS
  * ROUGE-L and BLEU here are dependency-free implementations. They track the
    official rouge-score / sacrebleu closely but may differ at the 2nd-3rd decimal.
    Treat local numbers as *directional*.
  * The organisers' gold 'reasoning_trace' is NOT in the training data, so during
    cross-validation there is no true reference. We use the gold explanation
    sentences for each section as a PROXY reference. This makes the text metrics a
    proxy for extraction+phrasing quality, not an exact predictor of leaderboard text
    scores. The label metrics (Macro F1, Recall@3) ARE exact.
  * LSS here is a TF-IDF cosine proxy for the undisclosed legal embedding model.

WEIGHTS is exposed so you can update it after the organisers finalise details.
"""
import re
import math
from collections import Counter

WEIGHTS = {
    "macro_f1": 0.35,
    "rouge_l": 0.25,
    "bleu": 0.20,
    "recall_at_3": 0.10,
    "lss": 0.10,
}

# ---------------------------------------------------------------- tokenization
_WORD = re.compile(r"[A-Za-z0-9]+")


def _tok(s: str):
    return _WORD.findall((s or "").lower())


# ---------------------------------------------------------------- label metrics
def macro_f1(gold_sets, pred_sets, labels):
    """gold_sets / pred_sets: list of sets of section strings (one per document)."""
    f1s = []
    for lab in labels:
        tp = fp = fn = 0
        for g, p in zip(gold_sets, pred_sets):
            in_g = lab in g
            in_p = lab in p
            if in_g and in_p:
                tp += 1
            elif in_p and not in_g:
                fp += 1
            elif in_g and not in_p:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def recall_at_k(gold_sets, pred_ranked, k=3):
    """pred_ranked: list of *ordered* prediction lists (highest confidence first)."""
    recs = []
    for g, p in zip(gold_sets, pred_ranked):
        if not g:
            continue
        topk = set(p[:k])
        recs.append(len(set(g) & topk) / len(g))
    return sum(recs) / len(recs) if recs else 0.0


# ---------------------------------------------------------------- ROUGE-L
def _lcs(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l(pred, ref, beta=1.2):
    p_tok, r_tok = _tok(pred), _tok(ref)
    if not p_tok or not r_tok:
        return 0.0
    l = _lcs(p_tok, r_tok)
    if l == 0:
        return 0.0
    prec = l / len(p_tok)
    rec = l / len(r_tok)
    b2 = beta * beta
    return (1 + b2) * prec * rec / (rec + b2 * prec)


# ---------------------------------------------------------------- BLEU (sentence)
def bleu(pred, ref, max_n=4):
    p_tok, r_tok = _tok(pred), _tok(ref)
    if not p_tok:
        return 0.0
    weights = [1.0 / max_n] * max_n
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_ng = Counter(tuple(p_tok[i:i + n]) for i in range(len(p_tok) - n + 1))
        r_ng = Counter(tuple(r_tok[i:i + n]) for i in range(len(r_tok) - n + 1))
        overlap = sum(min(c, r_ng[g]) for g, c in p_ng.items())
        total = max(sum(p_ng.values()), 1)
        # +1 smoothing (Chen & Cherry method 1) to avoid log(0) on short texts.
        prec = (overlap + 1e-9) / total if n == 1 else (overlap + 1.0) / (total + 1.0)
        log_sum += weights[n - 1] * math.log(max(prec, 1e-12))
    bp = 1.0 if len(p_tok) > len(r_tok) else math.exp(1 - len(r_tok) / max(len(p_tok), 1))
    return bp * math.exp(log_sum)


# ---------------------------------------------------------------- LSS proxy
def lss_proxy(pred, ref):
    """TF-IDF-free cosine over raw term counts (stand-in for the undisclosed
    legal embedding model). Directional only."""
    pc, rc = Counter(_tok(pred)), Counter(_tok(ref))
    if not pc or not rc:
        return 0.0
    common = set(pc) & set(rc)
    dot = sum(pc[t] * rc[t] for t in common)
    np_ = math.sqrt(sum(v * v for v in pc.values()))
    nr = math.sqrt(sum(v * v for v in rc.values()))
    return dot / (np_ * nr) if np_ and nr else 0.0


# ---------------------------------------------------------------- composite
def composite(macro_f1_v, rouge_v, bleu_v, recall_v, lss_v, weights=WEIGHTS):
    return (
        weights["macro_f1"] * macro_f1_v
        + weights["rouge_l"] * rouge_v
        + weights["bleu"] * bleu_v
        + weights["recall_at_3"] * recall_v
        + weights["lss"] * lss_v
    )
