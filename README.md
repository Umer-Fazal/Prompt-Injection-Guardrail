# Adaptive Multi-Layer Guardrail for Prompt Injection in LLM Applications

A three-layer detector that flags prompt injection and jailbreak attempts, hardened by an adaptive black-box attacker and adversarial retraining. Everything runs from one script, `prompt_injection_v2.py`, on a free Colab T4.

> Research and defensive use only. The attacker module generates harmless canary payloads and is meant for testing your own detector.

## Overview

| | |
|---|---|
| **Problem** | LLM applications read untrusted text (user prompts, web pages, documents). Hidden instructions in that text can hijack the model. |
| **Solution** | Three detection layers whose scores are fused into one calibrated probability, mapped to **allow / review / block**. |
| **AI models** | ModernBERT-base (main classifier), multilingual-e5-small (novelty), Qwen2.5-0.5B (perplexity), XGBoost (fusion) |
| **Main dataset** | `deepset/prompt-injections` (116-prompt test set) plus generated attacks, harmless documents and look-alike prompts |
| **Transfer datasets** | `xTRam1/safe-guard-prompt-injection`, `jackhhao/jailbreak-classification` |
| **Metrics** | F1, ROC-AUC, attack success rate (ASR), false-positive rate, TPR at fixed FPR, ECE, latency |

### Architecture

```
input -> chunking (900 chars, 200 overlap, max over chunks)
      -> Layer 1: ModernBERT-base, mean pooling, 2 heads (attack yes/no + attack family)
      -> Layer 2: multilingual-e5-small, kNN distance to benign + similarity to attack bank
      -> Layer 3: Qwen2.5-0.5B log-perplexity + 9 text-structure features
      -> fusion: XGBoost on 13 features -> Platt calibration -> two thresholds
      -> allow / review / block

adaptive attacker: score-guided fuzzer, 12 mutation operators, canary validity check,
                   bypasses added to training over 3 retraining rounds
```

### Key results (mean of 3 seeds)

| Measure | Result |
|---|---|
| Clean test F1 / ROC-AUC (fused) | 0.940 +/- 0.032 / 0.984 +/- 0.003 |
| Adaptive attack success rate, fused | 0.016 +/- 0.016 |
| Same attack on DistilBERT / DeBERTa-v3-small | 0.438 / 0.495 |
| Long documents (2k to 12k chars) | 100% detection, at most 1% false alarms |
| Harmless documents wrongly flagged, before vs after retraining | 0.356 to 0.117 |
| Latency per short input (fused) | about 88 ms (DistilBERT: 5.5 ms) |

### Honest limitations

- Clean-test accuracy is **not** significantly better than DistilBERT or DeBERTa; the gain is robustness against the adaptive attacker and long inputs.
- Layers 2 and 3 add no clean-set accuracy (ablation); Layer 1 carries the result.
- Transfer is weak on `xTRam1/safe-guard-prompt-injection` (ROC-AUC about 0.48) and good on `jackhhao` (0.957).
- The review band was empty in the measured runs, and the fused system is about 16x slower than DistilBERT.
- The test set is small (116 prompts), so confidence intervals are wide. Full discussion is in the report.

## Requirements

- Python 3.10+
- A GPU is strongly recommended. Tested on a free **Colab T4 (16 GB)**. A 8 GB laptop GPU runs out of memory.
- Internet access on first run, to download models and datasets from Hugging Face.

## Quick start (Google Colab)

1. Open a new notebook and choose **Runtime > Change runtime type > T4 GPU**.
2. Upload `prompt_injection_v2.py`.
3. Install dependencies:

```bash
!pip install -q -U transformers datasets sentence-transformers scikit-learn xgboost shap matplotlib pandas sentencepiece accelerate protobuf
```

If a tokenizer error mentioning `protobuf` or `tiktoken` appears, run `!pip install -U protobuf` and restart the runtime.

4. Run:

```bash
# offline logic test, synthetic data, no downloads (about a minute)
!python prompt_injection_v2.py --smoke --out smoke_test

# quick real run (about 25 to 40 minutes on a T4)
!python prompt_injection_v2.py --fast --out results_fast

# full run used in the report: 3 seeds, cross-dataset, pretrained guard baseline
!python prompt_injection_v2.py --out results --seeds 3 --cross --guard-model protectai/deberta-v3-base-prompt-injection-v2
```

To reproduce the report's three separate seed folders, run with `--seed 42`, `--seed 43` and `--seed 44` and different `--out` names.

## Local install

```bash
git clone https://github.com/Umer-Fazal/Prompt-Injection-Guardrail
cd https://github.com/Umer-Fazal/Prompt-Injection-Guardrail
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets sentence-transformers scikit-learn xgboost shap matplotlib pandas sentencepiece accelerate protobuf
python prompt_injection_v2.py --smoke --out smoke_test
```

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--out` | `results` | Output folder |
| `--seed` / `--seeds` | 42 / 1 | Starting seed / number of seeds (aggregated to `*_mean_std.csv`) |
| `--fast` | off | Smaller run: fewer epochs, rounds and test documents |
| `--smoke` | off | Offline logic test with synthetic data and tiny stand-in models |
| `--rounds` | 3 | Adversarial retraining rounds |
| `--main-model` | `answerdotai/ModernBERT-base` | Layer 1 backbone |
| `--compare-models` | DistilBERT, DeBERTa-v3-small | Baselines trained on the same data |
| `--embed-model` | `intfloat/multilingual-e5-small` | Layer 2 |
| `--ppl-model` | `Qwen/Qwen2.5-0.5B` | Layer 3 |
| `--epochs` / `--bs` / `--lr` | 6 / 16 / auto | Training settings |
| `--max-len` / `--infer-len` | 512 / 4096 | Training and inference token length |
| `--long-lengths` | 2000 6000 12000 | Document lengths in the long-document test |
| `--cross` | off | Evaluate on the two transfer datasets |
| `--guard-model` | none | Add a pretrained detector as a baseline |
| `--no-explain` | off | Skip SHAP token importance |

Run `python prompt_injection_v2.py --help` for the full list.

## Outputs

Each seed folder contains CSV tables and figures:

- Tables: `model_comparison.csv`, `adaptive_attack.csv`, `asr_by_family.csv`, `robustness_rounds.csv`, `long_doc.csv`, `policy.csv`, `ablation.csv`, `cross_dataset.csv`, `latency.csv`, `latency_by_length.csv`, `error_analysis.csv`, `run_config.json`
- Figures: `fig1` confusion matrices, `fig2` ROC and PR, `fig3` ASR by category, `fig4` ASR over retraining rounds, `fig5` token importance, `fig6` latency, `fig7` long documents, `fig8` calibration and policy
- With `--seeds > 1`: `*_mean_std.csv` aggregates

**Definitions.** ASR is the share of attacks scored below the operating threshold (let through). FPR is the share of benign inputs at or above it. The operating threshold gives 95% recall on out-of-fold validation scores.

## Repository layout

```
prompt_injection_v2.py          main script: data, attacker, models, fusion, evaluation, figures
prompt_injection_pipeline.py    earlier v1 baseline study (DeBERTa + GPT-2 perplexity), kept for comparison
figs/                           example figures from seed 42
results/                        add your own run outputs here
README.md
```

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | Use a T4 or larger. The script micro-batches and parks idle models, but 8 GB GPUs are too small. |
| Tokenizer or `protobuf` / `sentencepiece` error | `pip install -U protobuf sentencepiece`, then restart the runtime. A model whose tokenizer fails is skipped with a warning. |
| `Half` vs `Float` dtype error | Use the current script (weights are loaded in float32 and cast by autocast). |
| pandas 3 errors in CSV loading | Fixed in the current script. Update the file. |
| Layer 1 "collapsed" message | The script retries automatically at other learning rates. |

## References

1. F. Perez and I. Ribeiro, "Ignore Previous Prompt: Attack Techniques for Language Models," 2022. arXiv:2211.09527
2. K. Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection," 2023. arXiv:2302.12173
3. Y. Liu et al., "Formalizing and Benchmarking Prompt Injection Attacks and Defenses," USENIX Security 2024. arXiv:2310.12815
4. S. Schulhoff et al., "Ignore This Title and HackAPrompt," EMNLP 2023. arXiv:2311.16119
5. A. Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models," 2023. arXiv:2307.15043
6. P. Chao et al., "Jailbreaking Black Box Large Language Models in Twenty Queries," 2023. arXiv:2310.08419
7. J. Yi et al., "Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs," 2023. arXiv:2312.14197
8. E. Debenedetti et al., "AgentDojo," NeurIPS 2024 Datasets and Benchmarks. arXiv:2406.13352
9. G. Alon and M. Kamfonas, "Detecting Language Model Attacks with Perplexity," 2023. arXiv:2308.14132
10. B. Warner et al., "ModernBERT," 2024. arXiv:2412.13663

The full list of 18 references and the complete write-up are in the project report.

## Acknowledgements

Datasets and models are from the Hugging Face Hub: `deepset/prompt-injections`, `xTRam1/safe-guard-prompt-injection`, `jackhhao/jailbreak-classification`, `protectai/deberta-v3-base-prompt-injection-v2`, `answerdotai/ModernBERT-base`, `intfloat/multilingual-e5-small`, `Qwen/Qwen2.5-0.5B`, `microsoft/deberta-v3-small`, `distilbert-base-uncased`. Check each model and dataset card for its licence before redistributing.
