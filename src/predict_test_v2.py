"""
build_submission_v3.py
----------------------
Produce a submission in the format specified by the organizers' email
(different from what the task page originally described).

Output schema per record (one JSON object per line):
  {
    "id":               "<preserved from test file>",
    "fact":             "<preserved from test file>",
    "reasoning_traces": "<free-text string, our reasoning for the whole case>",
    "explanation":      {
        "verbatim sentence 1": ["IPC XXX"],
        "verbatim sentence 2": ["IPC XXX", "IPC YYY"]
    }
  }

Notes:
  - reasoning_traces: the email says THIS is the field evaluated. We build one
    coherent paragraph per case that concatenates our templated reasoning for
    every predicted section, so all statutory-ingredient language ends up in
    the evaluated text.
  - explanation: sentence -> list of IPC sections. If two predicted sections
    happen to point at the same sentence, we merge them into one dict entry
    with both sections in the list.
  - id + fact are copied through unchanged, as the email instructs.

Usage:
    python build_submission_v3.py \
        --train task1.jsonl \
        --test task_1_statute_prediction.jsonl \
        --out task_1_statute_prediction_submission.jsonl
"""
import argparse
import json
import re

from esp_utils import read_jsonl, write_jsonl
from esp_pipeline_v2 import ESPModelV2


def to_verbatim(candidate: str, fact: str) -> str:
    """Return the exact substring of `fact` corresponding to `candidate`.

    Our extractor concatenates 2-sentence spans with a single space, but the
    original fact often has double spaces between sentences. That makes the
    candidate almost-but-not-quite a substring of fact, violating the "keys
    must be verbatim sentences copied from the fact" rule.

    Fix: match the normalized candidate against the normalized fact, then
    return the corresponding slice of the ORIGINAL fact.
    """
    if candidate in fact:
        return candidate
    # Build a whitespace-tolerant regex from the candidate
    pattern = re.compile(
        r"\s+".join(re.escape(tok) for tok in candidate.split()),
        re.DOTALL,
    )
    m = pattern.search(fact)
    if m:
        return fact[m.start():m.end()]
    return candidate  # give up, keep original


def build_reasoning_paragraph(predictions):
    """Join all per-section templated reasonings into one paragraph."""
    parts = [p["reasoning_trace"] for p in predictions]
    return " ".join(parts).strip()


def build_explanation_dict(predictions):
    """Group by exact_fact sentence -> list of sections that triggered it."""
    explanation = {}
    for p in predictions:
        sent = p["exact_fact"]
        if not sent:
            continue
        if sent in explanation:
            if p["section"] not in explanation[sent]:
                explanation[sent].append(p["section"])
        else:
            explanation[sent] = [p["section"]]
    return explanation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="task1.jsonl")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="task_1_statute_prediction_submission.jsonl")
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    train = read_jsonl(args.train)
    test = read_jsonl(args.test)
    print(f"train={len(train)} cases | test={len(test)} cases")

    # Fit the same verified v2 model
    model = ESPModelV2(doc_threshold=args.threshold).fit(train)

    out_rows = []
    for r in test:
        preds, _ = model.predict_one(r["fact"])
        # Enforce the "verbatim substring of fact" rule for each exact_fact.
        for p in preds:
            p["exact_fact"] = to_verbatim(p["exact_fact"], r["fact"])
        out = {
            "id": r["id"],
            "fact": r["fact"],
            "reasoning_traces": build_reasoning_paragraph(preds),
            "explanation": build_explanation_dict(preds),
        }
        out_rows.append(out)

    # Schema validation against the email's spec
    for row in out_rows:
        assert isinstance(row["id"], str) and row["id"], "id missing"
        assert isinstance(row["fact"], str) and row["fact"], "fact missing"
        assert isinstance(row["reasoning_traces"], str), \
            "reasoning_traces must be a string"
        assert isinstance(row["explanation"], dict), "explanation must be a dict"
        for sent, secs in row["explanation"].items():
            assert isinstance(sent, str) and sent, "sentence key must be str"
            assert sent in row["fact"], \
                f"sentence key must be verbatim substring of fact ({row['id']})"
            assert isinstance(secs, list) and all(
                isinstance(s, str) and s.startswith("IPC ") for s in secs), \
                f"explanation values must be lists of 'IPC XXX' strings; got {secs}"

    write_jsonl(out_rows, args.out)
    print(f"wrote {len(out_rows)} predictions -> {args.out}")

    # Preview
    print("\n--- first prediction preview ---")
    preview = dict(out_rows[0])
    # trim fact for readability
    if len(preview["fact"]) > 200:
        preview["fact"] = preview["fact"][:200] + " …[truncated for display]"
    print(json.dumps(preview, indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
