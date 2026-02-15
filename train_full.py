import torch
import librosa
import numpy as np
import io
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# -------------------------
# Check GPU
# -------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------
# 1. Load Full Dataset
# -------------------------
print("Loading dataset...")
dataset = load_dataset("abnerh/TORGO-database", split="train")
dataset = dataset.cast_column("audio", Audio(decode=False))

# Filter for dysarthric speech
print("Filtering for dysarthric speech...")
dataset = dataset.filter(lambda x: x["speech_status"] == "dysarthria")

# [STRATEGY CHANGE] Use 99% for Training
# We keep 1% (approx 55 files) just to prevent Trainer errors and track basic loss.
print("Splitting 99/1 for Maximum Training...")
dataset = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)} (Huge increase!)")
print(f"Validation samples: {len(test_dataset)} (Just for safety)")

# -------------------------
# [NEW] Log Test Files
# -------------------------
print("-" * 40)
print(f"Saving list of {len(test_dataset)} test files to 'test_files.txt'...")

with open("test_files.txt", "w") as f:
    for i, sample in enumerate(test_dataset):
        # Extract the filename from the audio path
        full_path = sample["audio"]["path"] 
        
        if full_path:
            # Handle both Linux (/) and Windows (\) separators just in case
            filename = full_path.split("/")[-1].split("\\")[-1] 
            f.write(f"{i+1}. {filename}\n")
            # Optional: Print to console so you see them immediately
            # print(f"  Reserved for Test: {filename}") 
        else:
            f.write(f"{i+1}. [Unknown Path] ID: {sample.get('id', 'N/A')}\n")

print("File list saved to 'test_files.txt'.")
print("-" * 40)

# -------------------------
# 2. Load Whisper model
# -------------------------
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# -------------------------
# 3. Apply LoRA
# -------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# 4. Preprocess
# -------------------------
def prepare(batch):
    audio_info = batch["audio"]
    try:
        if "bytes" in audio_info and audio_info["bytes"] is not None:
            array, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]))
        else:
            array, sampling_rate = librosa.load(audio_info["path"], sr=None)
    except Exception as e:
        return {"input_features": [], "labels": []}

    array = array.astype(np.float32)
    if sampling_rate != 16000:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=16000)

    inputs = processor(array, sampling_rate=16000, return_tensors="pt")
    input_text = batch["transcription"]
    labels = processor.tokenizer(input_text).input_ids
    
    return {"input_features": inputs.input_features[0], "labels": labels}

print("Preprocessing dataset...")
train_dataset = train_dataset.map(prepare, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(prepare, remove_columns=test_dataset.column_names)

# -------------------------
# 5. Data Collator
# -------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# -------------------------
# 6. Training setup
# -------------------------
training_args = TrainingArguments(
    output_dir="./whisper-lora-full",  # Saving to a NEW folder
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=6,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=200,      # Increased to save space
    save_total_limit=2,
    eval_strategy="epoch",
    remove_unused_columns=False, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator, 
)

# -------------------------
# 7. Start training
# -------------------------
print("Starting FULL training...")
trainer.train()

print("Saving FULL model...")
model.save_pretrained("./whisper-lora-full")
print("Done! Model saved to ./whisper-lora-full")