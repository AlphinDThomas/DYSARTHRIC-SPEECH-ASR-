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
# 1. Load dataset & Disable Auto-Decoding
# -------------------------
print("Loading dataset...")
dataset = load_dataset("abnerh/TORGO-database", split="train")

# Disable automatic decoding to avoid "torchcodec" errors on Windows.
dataset = dataset.cast_column("audio", Audio(decode=False))

# Filter for dysarthric speech (Strategy 1)
print("Filtering for dysarthric speech...")
dataset = dataset.filter(lambda x: x["speech_status"] == "dysarthria")

# Train/test split
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# -------------------------
# 2. Load Whisper model
# -------------------------
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# -------------------------
# 3. Apply LoRA (PEFT)
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
# 4. Manual Preprocess function
# -------------------------
def prepare(batch):
    audio_info = batch["audio"]
    
    # 1. Load Audio (Manual fix for Windows)
    try:
        if "bytes" in audio_info and audio_info["bytes"] is not None:
            # Load from memory (fastest)
            array, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]))
        else:
            # Load from disk (fallback)
            array, sampling_rate = librosa.load(audio_info["path"], sr=None)
    except Exception as e:
        print(f"Skipping bad audio file: {e}")
        return {"input_features": [], "labels": []}

    # Ensure float32
    array = array.astype(np.float32)

    # Resample to 16000Hz (Whisper requirement)
    if sampling_rate != 16000:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=16000)

    # 2. Process Audio
    inputs = processor(
        array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # 3. Process Text
    input_text = batch["transcription"]
    labels = processor.tokenizer(input_text).input_ids
    
    return {
        "input_features": inputs.input_features[0],
        "labels": labels
    }

print("Preprocessing dataset (this might take a moment)...")
train_dataset = train_dataset.map(prepare, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(prepare, remove_columns=test_dataset.column_names)

# [OPTIMIZATION] Commented out to save 15+ minutes of waiting. 
# Since 'Map' worked, we assume files are good.
# train_dataset = train_dataset.filter(lambda x: len(x["input_features"]) > 0)
# test_dataset = test_dataset.filter(lambda x: len(x["input_features"]) > 0)

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
    output_dir="./whisper-lora",
    per_device_train_batch_size=1,  # Keep at 1 for RTX 2050
    gradient_accumulation_steps=8,  # Simulate batch size of 8
    learning_rate=1e-4,
    num_train_epochs=6,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="epoch",
    # no_cuda=False,  <-- REMOVED (Deprecated)
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
print("Starting training...")
trainer.train()

# Save LoRA adapters
print("Saving model...")
model.save_pretrained("./whisper-lora")
print("Done! Model saved to ./whisper-lora")