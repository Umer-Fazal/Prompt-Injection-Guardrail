# Prompt Injection Guardrail

A three-layer detector for prompt injection and jailbreaks in LLM applications, hardened with an adaptive attacker and adversarial retraining. Course project for AI for Cybersecurity (FAST NUCES Karachi).

## How it works

```
input -> chunking (900 chars, 200 overlap, max over chunks)
      -> L1 ModernBERT-base (attack yes/no + attack family)
      -> L2 multilingual-e5-small (novelty vs benign, similarity to attack bank)
      -> L3 Qwen2.5-0.5B (log-perplexity + 9 structure features)
      -> XGBoost fusion (13 features) -> Platt calibration -> allow / review / block
```

A black-box fuzzer (12 mutation operators) probes the scores, and its bypasses are added to training over 3 retraining rounds.

## Results (mean of 3 seeds)

| Measure | Result |
|---|---|
| Clean test F1 / ROC-AUC | 0.940 / 0.984 |
| Adaptive attack success rate | 0.016 (DistilBERT 0.438, DeBERTa-v3-small 0.495) |
| Long documents (2k to 12k chars) | 100% detection, at most 1% false alarms |
| Latency per short input | about 88 ms |

Clean-test accuracy is not significantly better than the baselines; the gain is robustness. Transfer to `xTRam1/safe-guard-prompt-injection` is weak (ROC-AUC about 0.48). The test set has only 116 prompts.

## Run

Needs Python 3.10+ and a GPU (tested on a Colab T4).

```bash
pip install -q -U transformers datasets sentence-transformers scikit-learn xgboost shap matplotlib pandas sentencepiece accelerate protobuf

python prompt_injection_v2.py --smoke --out smoke_test    # offline logic test
python prompt_injection_v2.py --fast --out results_fast   # quick run, 25-40 min on a T4
python prompt_injection_v2.py --out results --seeds 3 --cross \
    --guard-model protectai/deberta-v3-base-prompt-injection-v2   # full run
```

Run `python prompt_injection_v2.py --help` for all options. Each run writes CSV tables and figures to the `--out` folder.

## Data and models

Hugging Face Hub: `deepset/prompt-injections`, `xTRam1/safe-guard-prompt-injection`, `jackhhao/jailbreak-classification`, `answerdotai/ModernBERT-base`, `intfloat/multilingual-e5-small`, `Qwen/Qwen2.5-0.5B`. Check each card for its licence.

For research and defensive testing only.
