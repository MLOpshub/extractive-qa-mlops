from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os

MODEL_DIR = "/model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR, local_files_only=True)

print("✅ QA model loaded correctly")
print(model.__class__)
