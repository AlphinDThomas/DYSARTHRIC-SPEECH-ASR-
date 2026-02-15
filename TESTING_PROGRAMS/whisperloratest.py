import torch
import librosa
import numpy as np
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig

# ==========================================
#  SETTINGS
# ==========================================
# File name of your audio (must be in the same folder as this script)
AUDIO_FILENAME = "TEST AUDIOS\\dysarthric_a.wav" 

# Automatically get the absolute path to the file in this folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(SCRIPT_DIR, AUDIO_FILENAME)

# Your trained model folder (relative to this script)
MODEL_PATH = os.path.join(SCRIPT_DIR, "whisper-lora")
# ==========================================

def transcribe_local_audio(audio_path, model_path):
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the Fine-Tuned Model
    print(f"Loading model from: {model_path}")
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_path)
        model.to(device)
        
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        print(f"Make sure the folder '{model_path}' exists and contains 'adapter_model.bin' and 'adapter_config.json'.")
        return

    # 3. Load and Preprocess Audio
    print(f"Loading audio file: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"\n[ERROR] File not found: {audio_path}")
        print(f"Please make sure '{AUDIO_FILENAME}' is in the folder: {SCRIPT_DIR}")
        return

    try:
        # librosa.load automatically resamples to 16000Hz
        array, sampling_rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"[ERROR] Error reading audio: {e}")
        return

    # Ensure valid audio array
    if len(array) == 0:
        print("[ERROR] Audio file is empty.")
        return
        
    # 4. Run Inference
    print("Transcribing...")
    inputs = processor(array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        # [FIX APPLIED HERE]
        # Force the model to use English. This fixes the "Dutch" hallucination.
        predicted_ids = model.generate(
            input_features, 
            language="en", 
            task="transcribe"
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # 5. Print Result
    print("\n" + "=" * 50)
    print(f"FILE: {AUDIO_FILENAME}")
    print("-" * 50)
    print(f"TRANSCRIPTION: {transcription}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    transcribe_local_audio(AUDIO_PATH, MODEL_PATH)