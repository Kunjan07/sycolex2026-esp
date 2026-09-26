"""
llm_refine.py  (OPTIONAL — needs an Anthropic API key; run in Colab or locally)
-------------------------------------------------------------------------------
Post-processes a v2 submission with an LLM:
  1. exact_fact re-selection: shows the LLM the fact's sentences (numbered) and
     asks which sentence(s) trigger the predicted section. The answer is mapped
     back to the ORIGINAL sentence text, so the output stays strictly extractive
     (a verbatim substring of the fact) — the LLM only chooses, never rewrites.
  2. reasoning polish: rewrites the templated reasoning to read naturally while
     preserving the statutory-ingredient wording (which drives ROUGE/BLEU/LSS).

NOT RUN/VERIFIED in the sandbox (no network there). Costs money per case
(~105 test cases x 2 small calls = cheap, but not free). The label predictions
are NOT changed — only exact_fact and reasoning_trace.

IMPORTANT: you have ONE submission. Before using the refined file, spot-check
it and, ideally, run the refinement on a held-out slice of TRAINING data and
confirm extraction accuracy beats the v2 extractor's 0.655 (compare picked
sentences against the gold `explanation` keys).

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=...   (or set in Colab: os.environ[...])
Usage:
    python llm_refine.py --submission submission.jsonl --test test.jsonl \
        --out submission_refined.jsonl
"""
import argparse
import json
import os
import re
import time

from esp_utils import read_jsonl, write_jsonl, split_sentences
from ipc_kb import IPC_KB

MODEL = "claude-sonnet-4-6"


def call_llm(client, prompt, max_tokens=400, retries=3):
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def pick_sentence(client, sentences, section):
    kb = IPC_KB.get(section, {})
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    prompt = (
        f"You are a legal analyst. {section} ({kb.get('offence','')}) requires: "
        f"{kb.get('ingredients','')}.\n\n"
        f"Below are the numbered sentences of a case's facts. Reply with ONLY the "
        f"number of the single sentence that most directly establishes the "
        f"ingredients of {section}. If two consecutive sentences are needed, "
        f"reply with both numbers separated by a comma (e.g. 4,5). No other text.\n\n"
        f"{numbered}"
    )
    out = call_llm(client, prompt, max_tokens=16)
    nums = [int(x) for x in re.findall(r"\d+", out or "")][:2]
    nums = [n for n in nums if 0 <= n < len(sentences)]
    if not nums:
        return None
    if len(nums) == 2 and abs(nums[0] - nums[1]) == 1:
        a, b = sorted(nums)
        return sentences[a] + " " + sentences[b]
    return sentences[nums[0]]


def polish_reasoning(client, section, exact_fact, template_reasoning):
    prompt = (
        "Rewrite the following legal reasoning so it reads as one natural, "
        "formal paragraph of 2-3 sentences. You MUST keep the statutory-"
        "ingredient phrasing intact (the clause after 'the facts establish "
        "that'). Ground it in the quoted fact. Reply with ONLY the rewritten "
        f"paragraph.\n\nSection: {section}\nFact sentence: {exact_fact}\n"
        f"Current reasoning: {template_reasoning}"
    )
    out = call_llm(client, prompt, max_tokens=300)
    return (out or "").strip() or template_reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--test", required=True, help="test JSONL with doc_id+fact")
    ap.add_argument("--out", default="submission_refined.jsonl")
    ap.add_argument("--skip_polish", action="store_true",
                    help="only re-select exact_fact; keep templated reasoning")
    args = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY first")
    client = anthropic.Anthropic()

    sub = read_jsonl(args.submission)
    facts = {r["doc_id"]: r["fact"] for r in read_jsonl(args.test)}

    changed = 0
    for i, row in enumerate(sub):
        fact = facts.get(row["doc_id"])
        if not fact:
            continue
        sentences = split_sentences(fact)
        for entry in row["statute"]:
            new_sent = pick_sentence(client, sentences, entry["section"])
            if new_sent and new_sent != entry["exact_fact"]:
                entry["exact_fact"] = new_sent
                changed += 1
            if not args.skip_polish:
                entry["reasoning_trace"] = polish_reasoning(
                    client, entry["section"], entry["exact_fact"],
                    entry["reasoning_trace"])
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(sub)} cases refined...")

    write_jsonl(sub, args.out)
    print(f"done. exact_fact changed in {changed} entries -> {args.out}")


if __name__ == "__main__":
    main()
