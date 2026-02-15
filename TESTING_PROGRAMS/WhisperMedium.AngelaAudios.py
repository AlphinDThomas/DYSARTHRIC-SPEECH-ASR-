import whisper
import os
import jiwer
import torch

# ==========================================
#  SETTINGS
# ==========================================
# 1. Folder with 10 wavs + transcript.txt
TEST_FOLDER = "../ANGELA" 

# 2. Transcript file name
TRANSCRIPT_FILE = "transcript.txt"

# 3. Model Size (tiny, base, small, medium, large)
MODEL_SIZE = "medium"
# ==========================================

def evaluate_angela_medium():
    # --- Load Model ---
    print(f"Loading OpenAI Whisper '{MODEL_SIZE}' model...")
    try:
        # Load the model directly from OpenAI
        model = whisper.load_model(MODEL_SIZE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Read Transcript File ---
    transcript_path = os.path.join(TEST_FOLDER, TRANSCRIPT_FILE)
    if not os.path.exists(transcript_path):
        print(f"Error: Cannot find {transcript_path}")
        print(f"Make sure you moved transcript.txt inside the {TEST_FOLDER} folder!")
        return

    print("\nReading transcript.txt...")
    files_to_test = []

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["): continue
            
            # Split "1-Text..."
            parts = line.split("-", 1)
            if len(parts) >= 2:
                file_id = parts[0].strip()
                text = parts[1].strip()
                files_to_test.append((f"{file_id}.wav", text))

    print(f"Found {len(files_to_test)} valid audio-text pairs.")

    # --- Run Evaluation ---
    predictions = []
    references = []

    print("\n" + "="*80)
    print(f"{'FILE':<10} | {'ACTUAL TEXT':<30} | {'WHISPER MEDIUM PREDICTION'}")
    print("="*80)

    for filename, reference_text in files_to_test:
        audio_path = os.path.join(TEST_FOLDER, filename)
        
        if not os.path.exists(audio_path):
            print(f"{filename:<10} | [FILE NOT FOUND] - Skipping...")
            continue

        try:
            # Transcribe (Force English, No Randomness)
            result = model.transcribe(audio_path, language="en")
            prediction = result["text"].strip()
            
            # Print result
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

        print("\n" + "="*50)
        print(f"RESULTS: OpenAI Whisper ({MODEL_SIZE}) on ANGELA Folder")
        print("="*50)
        print(f"WER (Word Error Rate):      {wer:.2%}")
        print(f"CER (Character Error Rate): {cer:.2%}")
        print("="*50)
    else:
        print("\nNo files were processed.")

if __name__ == "__main__":
    evaluate_angela_medium()