# Model Card

## Overview
- **Base model**: bert-base-uncased
- **Dataset**: squad
- **Task**: Extractive Question Answering (SQuAD-style)
- **Validation metrics**: EM=-1.0000, F1=-1.0000

## Training data sizes
- train examples: 200
- eval examples: 20
- train features (after sliding window): 200
- eval features (after sliding window): 20
- train feat/example: 1.000
- eval feat/example: 1.000

## Runtime
- env: local
- platform: Windows-10-10.0.26100-SP0

## Training config (key)
- max_length: 384
- doc_stride: 128
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
- MLflow tracking URI: file:D:\projet_esilv\MLOps\extractive-qa-mlops\mlruns
