# Model Card

## Overview
- **Base model**: bert-base-uncased
- **Dataset**: squad
- **Task**: Extractive Question Answering (SQuAD-style)
- **Validation metrics**: EM=80.3027, F1=87.8438

## Training data sizes
- train examples: 87599
- eval examples: 10570
- train features (after sliding window): 88524
- eval features (after sliding window): 10784
- train feat/example: 1.011
- eval feat/example: 1.020

## Runtime
- env: colab
- platform: Linux-6.6.105+-x86_64-with-glibc2.35

## Training config (key)
- max_length: 384
- doc_stride: 128
- learning_rate: 5e-05
- epochs: 1.0
- batch_size(train): 8
- fp16: True
- seed: 42

## Error analysis
- report path: reports/error_cases.json

## Intended use
- Course project / research experiments on SQuAD v1 style QA.
- Offline evaluation and demonstrations (not production-grade).

## Limitations
- Long context: may miss answers far from the selected window.
- Entity/number confusion when multiple similar mentions exist.
- Sensitive to paraphrases / wording differences.

## Notes
- MLflow tracking URI: file:/content/drive/MyDrive/extractive-qa-mlops/mlruns
