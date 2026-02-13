from dataclasses import dataclass


@dataclass(frozen=True)
class QASettings:
    max_context_chars: int = 3000
    max_answer_chars: int = 200
    default_top_k: int = 3
