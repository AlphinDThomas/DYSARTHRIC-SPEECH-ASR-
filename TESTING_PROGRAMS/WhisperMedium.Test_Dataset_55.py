import whisper
import os
import jiwer
from tqdm import tqdm

# ==========================================
#  SETTINGS
# ==========================================
# Path to your 56 test files (adjust ../ if needed)
TEST_FOLDER = "../TEST_DATASET_55" 

# Model Size (tiny, base, small, medium, large)
# You mentioned testing "medium", which is much smarter than "base"
MODEL_SIZE = "medium" 
# ==========================================

def evaluate_openai_whisper():
    # 1. Load the Model
    print(f"Loading Whisper '{MODEL_SIZE}' model...")
    try:
        model = whisper.load_model(MODEL_SIZE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Find Test Files
    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Folder '{TEST_FOLDER}' not found.")
        return

    files = [f for f in os.listdir(TEST_FOLDER) if f.endswith(".wav")]
    print(f"Found {len(files)} test files.")

    predictions = []
    references = []

    print(f"Starting evaluation with Whisper {MODEL_SIZE}...")
    
    # 3. Loop through files
    for filename in tqdm(files):
        audio_path = os.path.join(TEST_FOLDER, filename)
        text_filename = filename.replace(".wav", ".txt")
        text_path = os.path.join(TEST_FOLDER, text_filename)

        # Get Ground Truth
        if not os.path.exists(text_path):
            continue
            
        with open(text_path, "r", encoding="utf-8") as f:
            reference_text = f.read().strip()

        # Transcribe (The easy way!)
        try:
            # language="en" forces English, which helps accuracy
            # New line: Force deterministic results (no random guessing)
            result = model.transcribe(audio_path, language="en", temperature=0.0)
            prediction_text = result["text"].strip()
        except Exception as e:
            print(f"Error transcribing {filename}: {e}")
            continue

        # Normalization (Standardize for fair scoring)
        references.append(jiwer.RemovePunctuation()(reference_text.lower()))
        predictions.append(jiwer.RemovePunctuation()(prediction_text.lower()))

    # 4. Calculate Scores
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)

    print("\n" + "="*50)
    print(f"RESULTS: OpenAI Whisper ({MODEL_SIZE})")
    print("="*50)
    print(f"WER (Word Error Rate):      {wer:.2%}")
    print(f"CER (Character Error Rate): {cer:.2%}")
    print("="*50)

    # 5. Save Detailed Report
    report_file = f"results_whisper_{MODEL_SIZE}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Model: OpenAI Whisper {MODEL_SIZE}\n")
        f.write(f"WER: {wer:.2%} | CER: {cer:.2%}\n")
        f.write("="*50 + "\n")
        for i in range(len(predictions)):
            f.write(f"File: {files[i]}\n")
            f.write(f"REF:  {references[i]}\n")
            f.write(f"PRED: {predictions[i]}\n")
            f.write("-" * 20 + "\n")
            
    print(f"Full breakdown saved to '{report_file}'")

if __name__ == "__main__":
    evaluate_openai_whisper()