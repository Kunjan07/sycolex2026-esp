# KunjanKalita at SYCOLEX 2026 — Explainable Statute Prediction (FIRE 2026, Task 1)

Code for team **KunjanKalita**'s submission to Task 1 (*Explainable Statute Prediction*) of the
FIRE 2026 track **"LLM as a Judge?: From Statute Prediction to Sycophancy Detection in Law"** (SYCOLEX 2026).

Given the facts of an Indian criminal case, the system (1) predicts the applicable IPC sections,
(2) extracts the verbatim sentence(s) from the facts that trigger each section, and
(3) produces a reasoning trace grounded in the statutory ingredients of each offence.

The submitted run (Run 1) placed **9th of 16** in the official Run 1 results.

## Approach

The submitted system is a lightweight, CPU-only, three-stage pipeline (**Tier 1**):

| Stage | What it does | Method |
|---|---|---|
| 1. Label prediction | Multi-label prediction over 7 IPC sections (302, 376, 498A, 420, 147, 506, 201) | Word (1–2) + character (3–5) n-gram TF-IDF, one-vs-rest logistic regression, threshold 0.30, 1–3 labels per case |
| 2. Evidence extraction | Picks the fact sentence that triggers each predicted section | Sentence classifier trained on gold explanation sentences plus a `NONE` class of non-explanatory sentences; candidates include 2-sentence spans; outputs mapped back to verbatim substrings of the fact |
| 3. Reasoning | Generates the reasoning trace | Template built from a hand-written knowledge base of statutory ingredients (`src/ipc_kb.py`), grounded in the extracted sentence |

## Results (5-fold cross-validation on the training set, 525 cases)

| Metric | Mean (± std) |
|---|---|
| Macro F1 | 0.755 (± 0.036) |
| Recall@3 | 0.983 (± 0.013) |
| Extraction accuracy | 0.655 (± 0.081) |
| ROUGE-L* | 0.233 (± 0.008) |
| BLEU* | 0.127 (± 0.011) |
| LSS* | 0.535 (± 0.017) |
| Composite* | 0.500 (± 0.014) |

\* Proxy metrics: gold reasoning traces are not released with the training data, so text metrics are computed
against the gold explanation sentences, and LSS uses a term-count cosine in place of the organisers' undisclosed
legal embedding model. Label metrics and extraction accuracy are exact.

## Repository structure

```
├── notebooks/
│   └── FIRE2026_ESP_Pipeline.ipynb   # Self-contained Colab notebook (all tiers)
├── src/
│   ├── ipc_kb.py              # Statutory-ingredient knowledge base + reasoning template
│   ├── esp_utils.py           # JSONL IO, section normalisation, legal-aware sentence splitter
│   ├── esp_metrics.py         # Local re-implementation of the composite metric
│   ├── esp_pipeline_v2.py     # Tier 1 model (stages 1–3)
│   ├── run_cv_v2.py           # 5-fold cross-validation
│   ├── predict_test_v2.py     # Builds the submission file (submitted run)
│   ├── colab_inlegalbert.py   # Tier 2 (exploratory): InLegalBERT classifier
│   ├── embed_rerank.py        # Tier 1.5 (exploratory): sentence-embedding reranker
│   └── llm_refine.py          # Tier 3 (exploratory): LLM refinement
├── requirements.txt
└── LICENSE
```

The notebook writes the same modules to disk with `%%writefile`, so it runs on Colab without cloning the repo.

## Data

The task data is distributed by the FIRE 2026 / SYCOLEX organisers and is **not included** in this repository.
Place the training file (`task1.jsonl`) and test file in the directory you run the scripts from.

## Usage

```bash
pip install -r requirements.txt

# 5-fold cross-validation
python src/run_cv_v2.py --data task1.jsonl --folds 5

# Generate the submission (Run 1)
python src/predict_test_v2.py --train task1.jsonl --test test.jsonl \
    --out task_1_statute_prediction_submission.jsonl
```

Or open `notebooks/FIRE2026_ESP_Pipeline.ipynb` in Google Colab and run the cells top to bottom.

## Exploratory tiers (not part of the submitted run)

The notebook also contains three optional extensions. None of them was used for the submitted run, and none was
fully validated:

- **Tier 2 — InLegalBERT ensemble** (`colab_inlegalbert.py`): cross-validation was started but not completed.
- **Tier 1.5 — semantic reranker for evidence extraction** (`embed_rerank.py`): motivated by the gap between
  Tier 1 extraction accuracy (0.655) and the oracle ceiling of the candidate set (0.863); not evaluated.
- **Tier 3 — LLM refinement** (`llm_refine.py`): re-selects evidence sentences and polishes reasoning via the
  Anthropic API; not evaluated.

**Known issues** in these exploratory scripts:
- `predict_test_v2.py` does not implement the `--bert_probs` flag that the Tier 2 instructions pass to it.
- `embed_rerank.py` and `llm_refine.py` expect an earlier submission schema (`doc_id` and a `statute` list),
  whereas `predict_test_v2.py` writes the final organiser-specified schema (`id`, `fact`, `reasoning_traces`,
  `explanation`). They cannot yet consume the Tier 1 output directly.

## Citation

If you use this code, please cite the working notes paper:

```bibtex
@inproceedings{kalita2026sycolex,
  title     = {KunjanKalita at SYCOLEX 2026: KunjanKalita at SYCOLEX 2026: When Words Are Not
Enough - A Lightweight TF-IDF Baseline and Error
Analysis for Explainable Statute Prediction},
  author    = {Kalita, Kunjan},
  booktitle = {Working Notes of FIRE 2026 - Forum for Information Retrieval Evaluation},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
