# Model Card

## Overview
- **Base model**: bert-base-uncased
- **Dataset**: squad
- **Task**: Extractive Question Answering (SQuAD-style)
- **Validation metrics**: EM=79.9149, F1=87.6552

## Training data sizes
- train examples: 87599
- eval examples: 10570
- train features (after sliding window): 88511
- eval features (after sliding window): 10766
- train feat/example: 1.010
- eval feat/example: 1.019

## Runtime
- env: colab
- platform: Linux-6.6.105+-x86_64-with-glibc2.35

## Training config (key)
- max_length: 384
- doc_stride: 64
- learning_rate: 3e-05
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
