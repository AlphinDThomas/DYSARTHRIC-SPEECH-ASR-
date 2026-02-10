import torch
import librosa
import soundfile as sf
import io
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from datasets import load_dataset, Audio
import random

# -------------------------
# 1. Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------
# 2. Load Data & Create Split
# -------------------------
print("Loading dataset...")
dataset = load_dataset("abnerh/TORGO-database", split="train") 

# [CRITICAL FIX] Disable decoding to prevent crash
dataset = dataset.cast_column("audio", Audio(decode=False))

# Filter for dysarthric speech
dataset = dataset.filter(lambda x: x["speech_status"] == "dysarthria")

# Create a random split to get a test set
# (We use seed=42 here to keep THIS test stable, even if it differs from training)
dataset = dataset.train_test_split(test_size=0.2, seed=42) 
test_dataset = dataset["test"]

print(f"Test dataset size: {len(test_dataset)}")

# -------------------------
# 3. Load the Fine-Tuned Model
# -------------------------
print("Loading model...")
peft_model_id = "./whisper-lora" 
peft_config = PeftConfig.from_pretrained(peft_model_id)

# Load Base Model
model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
# Load LoRA Adapters
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(device)

processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)

# -------------------------
# 4. Pick a Sample
# -------------------------
# Pick a random file to test
sample_index = random.randint(0, len(test_dataset) - 1)
sample = test_dataset[sample_index]

# -------------------------
# 5. Manual Audio Loading (Windows Fix)
# -------------------------
audio_info = sample["audio"]

# Load bytes or path
if "bytes" in audio_info and audio_info["bytes"] is not None:
    array, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]))
else:
    array, sampling_rate = librosa.load(audio_info["path"], sr=None)

array = array.astype(np.float32)

# Resample to 16kHz
if sampling_rate != 16000:
    array = librosa.resample(array, orig_sr=sampling_rate, target_sr=16000)

# -------------------------
# 6. Run Inference
# -------------------------
print(f"\nProcessing sample #{sample_index}...")
inputs = processor(array, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device)

with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# -------------------------
# 7. Result
# -------------------------
print("-" * 60)
print(f"REFERENCE (Actual):   {sample['transcription']}")
print(f"PREDICTION (Model):   {transcription}")
print("-" * 60)