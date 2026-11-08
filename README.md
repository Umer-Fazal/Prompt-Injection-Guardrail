# Adaptive Multi-Layer Guardrail for Prompt Injection

Course project for AI for Cybersecurity (FAST NUCES Karachi). Detects prompt injection and jailbreaks with three fused layers, tested against an adaptive attacker. Everything is in `prompt_injection_v2.py`.

## Overview

- **Problem:** hidden instructions in untrusted text (prompts, web pages, documents) can hijack an LLM.
- **Solution:** three detection layers fused into one calibrated score, mapped to allow / review / block.
- **Models:** ModernBERT-base (classifier), multilingual-e5-small (novelty), Qwen2.5-0.5B (perplexity), XGBoost (fusion).
- **Data:** `deepset/prompt-injections` plus generated attacks and harmless documents. Transfer tests on `xTRam1/safe-guard-prompt-injection` and `jackhhao/jailbreak-classification`.

```
input -> chunking (900 chars, 200 overlap, max over chunks)
      -> L1 ModernBERT | L2 e5 novelty | L3 Qwen perplexity + structure
      -> XGBoost fusion -> Platt calibration -> allow / review / block
attacker: 12-operator fuzzer, bypasses added to training over 3 rounds
```

## Results (mean of 3 seeds)

| Measure | Result |
|---|---|
| Clean test F1 / ROC-AUC | 0.940 / 0.984 |
| Adaptive attack success rate | 0.016 (DistilBERT 0.438, DeBERTa-v3-small 0.495) |
| Long documents (2k to 12k chars) | 100% detection, at most 1% false alarms |
| Latency per short input | about 88 ms (DistilBERT 5.5 ms) |

Limits: clean-test accuracy is not significantly better than the baselines, Layers 2 and 3 add no clean accuracy, transfer to `xTRam1` is weak (ROC-AUC about 0.48), and the test set is small (116 prompts).

## Run (Google Colab, T4 GPU)

```bash
pip install -q -U transformers datasets sentence-transformers scikit-learn xgboost shap matplotlib pandas sentencepiece accelerate protobuf

python prompt_injection_v2.py --smoke --out smoke_test    # offline check, about 1 min
python prompt_injection_v2.py --fast --out results_fast   # quick run, about 30 min
python prompt_injection_v2.py --out results --seeds 3 --cross \
    --guard-model protectai/deberta-v3-base-prompt-injection-v2   # full run
```

Needs Python 3.10+ and a 16 GB GPU (8 GB GPUs run out of memory). If a tokenizer error mentions `protobuf`, run `pip install -U protobuf` and restart. All options: `python prompt_injection_v2.py --help`.

## Outputs

Per seed folder: CSV tables (`model_comparison`, `adaptive_attack`, `long_doc`, `ablation`, `cross_dataset`, `latency`, ...) and figures `fig1` to `fig8`. With `--seeds > 1`, `*_mean_std.csv` aggregates.

## Responsible use

For testing your own defenses. The attacker uses harmless canary payloads.
