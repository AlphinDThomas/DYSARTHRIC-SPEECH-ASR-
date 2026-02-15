import torch
import librosa
import numpy as np
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig

# ==========================================
#  SETTINGS
# ==========================================
# File to test (Put your audio file here)
AUDIO_FILENAME = "../AngelaAudios\\2.wav" 

# AUTOMATIC SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(SCRIPT_DIR, AUDIO_FILENAME)

# *** IMPORTANT: Pointing to the NEW 'All-In' Model ***
MODEL_PATH = os.path.join(SCRIPT_DIR, "../whisper-lora-full")
# ==========================================

def transcribe_with_full_model(audio_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the NEW Fine-Tuned Model
    print(f"Loading FULL model from: {model_path}")
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_path)
        model.to(device)
        
        # Load processor (Offline fallback included)
        try:
            processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
        except Exception:
            print("  (!) Internet unreachable. Loading processor from local cache...")
            processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, local_files_only=True)

    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        return

    # 2. Load Audio
    print(f"Loading audio file: {audio_path}")
    if not os.path.exists(audio_path):
        print(f"\n[ERROR] File not found: {audio_path}")
        return

    try:
        array, sampling_rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"[ERROR] Error reading audio: {e}")
        return

    # 3. Run Inference
    print("Transcribing...")
    inputs = processor(array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        # Force English to prevent hallucinations
        predicted_ids = model.generate(
            input_features, 
            language="en", 
            task="transcribe"
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # 4. Result
    print("\n" + "=" * 50)
    print(f"MODEL: Whisper-LoRA-FULL (Trained on 5500 samples)")
    print(f"FILE:  {AUDIO_FILENAME}")
    print("-" * 50)
    print(f"TRANSCRIPTION:\n{transcription}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    transcribe_with_full_model(AUDIO_PATH, MODEL_PATH)