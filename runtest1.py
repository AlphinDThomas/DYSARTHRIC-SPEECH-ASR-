import torch
import librosa
import soundfile as sf
import io
import numpy as np
import random
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from datasets import load_dataset, Audio

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Data
print("Loading dataset...")
dataset = load_dataset("abnerh/TORGO-database", split="train") 
dataset = dataset.cast_column("audio", Audio(decode=False)) # Fix for Windows
dataset = dataset.filter(lambda x: x["speech_status"] == "dysarthria")
dataset = dataset.train_test_split(test_size=0.2, seed=42) 
test_dataset = dataset["test"]

# Load Model
print("Loading model...")
peft_model_id = "./whisper-lora" 
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(device)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)

# ---------------------------------------------------------
# LOOP THROUGH 5 RANDOM SAMPLES
# ---------------------------------------------------------
print("\n" + "="*60)
print(f"TESTING 5 RANDOM SAMPLES")
print("="*60)

for i in range(5):
    # Pick random index
    idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[idx]
    
    # Load Audio
    audio_info = sample["audio"]
    if "bytes" in audio_info and audio_info["bytes"] is not None:
        array, sampling_rate = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        array, sampling_rate = librosa.load(audio_info["path"], sr=None)

    array = array.astype(np.float32)
    if sampling_rate != 16000:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=16000)

    # Inference
    inputs = processor(array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="en") # Force English to reduce warnings

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"\nSAMPLE #{idx}")
    print(f"Ref:   {sample['transcription']}")
    print(f"Pred:  {transcription}")
    print("-" * 30)