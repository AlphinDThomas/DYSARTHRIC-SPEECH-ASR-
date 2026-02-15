import torch
import os
import soundfile as sf
import librosa
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig

# ==========================================
#  SETTINGS
# ==========================================
# 1. Name of the folder with your 10 wavs + transcript.txt
TEST_FOLDER = "../ANGELA" 

# 2. Name of the transcript file
TRANSCRIPT_FILE = "transcript.txt"

# 3. Path to your trained model
MODEL_PATH = "../whisper-lora-full"
# ==========================================

def evaluate_custom_folder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from {MODEL_PATH}...")
    try:
        peft_config = PeftConfig.from_pretrained(MODEL_PATH)
        base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.to(device)
        
        try:
            processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
        except:
            processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, local_files_only=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Read Transcript File ---
    transcript_path = os.path.join(TEST_FOLDER, TRANSCRIPT_FILE)
    if not os.path.exists(transcript_path):
        print(f"Error: Cannot find {transcript_path}")
        return

    print("\nReading transcript.txt...")
    files_to_test = []

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or headers like 
            if not line or line.startswith("["): 
                continue
            
            # Split "1-Both figures..." into ["1", "Both figures..."]
            # We look for the FIRST hyphen only
            parts = line.split("-", 1)
            
            if len(parts) >= 2:
                file_id = parts[0].strip() # e.g. "1"
                text = parts[1].strip()    # e.g. "Both figures will go higher..."
                
                # Construct filename "1.wav"
                wav_name = f"{file_id}.wav"
                files_to_test.append((wav_name, text))
            else:
                print(f"Skipping weird line: {line}")

    print(f"Found {len(files_to_test)} valid audio-text pairs.")

    # --- Run Evaluation ---
    predictions = []
    references = []

    print("\n" + "="*80)
    print(f"{'FILE':<10} | {'ACTUAL TEXT':<30} | {'MODEL PREDICTION'}")
    print("="*80)

    for filename, reference_text in files_to_test:
        audio_path = os.path.join(TEST_FOLDER, filename)
        
        if not os.path.exists(audio_path):
            print(f"{filename:<10} | [FILE NOT FOUND] - Skipping...")
            continue

        # Load Audio
        try:
            # Load and resample to 16000Hz
            array, sr = sf.read(audio_path)
            if sr != 16000:
                array = librosa.resample(array, orig_sr=sr, target_sr=16000)
            
            # Transcribe
            inputs = processor(array, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features, language="en")
            
            prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            # Print result to screen
            print(f"{filename:<10} | {reference_text[:30]:<30}... | {prediction}")

            # Save for calculation
            references.append(jiwer.RemovePunctuation()(reference_text.lower()))
            predictions.append(jiwer.RemovePunctuation()(prediction.lower()))

        except Exception as e:
            print(f"Error on {filename}: {e}")

    # --- Calculate Final Score ---
    if len(references) > 0:
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)

        print("\n" + "="*40)
        print(f"FINAL SCORE ON 10 CUSTOM FILES")
        print("="*40)
        print(f"WER (Word Error Rate):      {wer:.2%}")
        print(f"CER (Character Error Rate): {cer:.2%}")
        print("="*40)
    else:
        print("\nNo files were processed.")

if __name__ == "__main__":
    evaluate_custom_folder()