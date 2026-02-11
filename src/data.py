# src/data.py
from __future__ import annotations

from typing import Dict, Any

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_squad(dataset_name: str = "squad"):
    return load_dataset(dataset_name)


def preprocess_train_features(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Dict[str, Any]:
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map each feature back 
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        # If no answer 
        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        end_char = start_char + len(answer_text)

        sequence_ids = tokenized.sequence_ids(i)

        # Find the start and end of the context in token space
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        # If answer is not fully inside the context span, label as (0,0)
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Otherwise move token_start/token_end to the answer boundaries
        token_start = context_start
        while token_start <= context_end and offsets[token_start][0] <= start_char:
            token_start += 1
        start_positions.append(token_start - 1)

        token_end = context_end
        while token_end >= context_start and offsets[token_end][1] >= end_char:
            token_end -= 1
        end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

def preprocess_eval_features(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Dict[str, Any]:
    # Tokenize for eval (keep offsets)
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Link features -> examples
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    # Keep only context offsets
    for i in range(len(tokenized["offset_mapping"])):
        seq_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            (o if seq_ids[k] == 1 else None) for k, o in enumerate(tokenized["offset_mapping"][i])
        ]

    tokenized["example_id"] = [examples["id"][sample_mapping[i]] for i in range(len(sample_mapping))]
    return tokenized

