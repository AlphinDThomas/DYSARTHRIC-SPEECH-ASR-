import torch
import os
import io
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
import jiwer
from tqdm import tqdm  # Progress bar

# ==========================================
#  SETTINGS
# ==========================================

# Use "../" to go up one folder level to find your data
TEST_FOLDER = "../TEST_DATASET_55"
MODEL_PATH = "../whisper-lora-full"
# ==========================================

def calculate_metrics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
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

    # 2. Find Test Files
    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Folder '{TEST_FOLDER}' not found.")
        return

    files = [f for f in os.listdir(TEST_FOLDER) if f.endswith(".wav")]
    print(f"Found {len(files)} test files.")

    predictions = []
    references = []

    print("Starting evaluation...")
    
    # 3. Loop through files
    for filename in tqdm(files):
        audio_path = os.path.join(TEST_FOLDER, filename)
        text_filename = filename.replace(".wav", ".txt")
        text_path = os.path.join(TEST_FOLDER, text_filename)

        # Get Ground Truth
        if not os.path.exists(text_path):
            # Skip if no text file exists to compare against
            continue
            
        with open(text_path, "r", encoding="utf-8") as f:
            reference_text = f.read().strip()

        # Get Audio
        try:
            array, sr = sf.read(audio_path)
            # Resample if needed (Whisper needs 16000Hz)
            if sr != 16000:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Transcribe
        inputs = processor(array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features, language="en")
        
        prediction_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        # Normalization (Important for fair scoring!)
        # We convert to lowercase and remove punctuation for a fair comparison
        references.append(jiwer.RemovePunctuation()(reference_text.lower()))
        predictions.append(jiwer.RemovePunctuation()(prediction_text.lower()))

    # 4. Calculate Scores
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)

    print("\n" + "="*40)
    print(f"FINAL RESULTS ON {len(predictions)} UNSEEN FILES")
    print("="*40)
    print(f"WER (Word Error Rate):      {wer:.2%}  (Lower is better)")
    print(f"CER (Character Error Rate): {cer:.2%}  (Lower is better)")
    print("="*40)

    # 5. Show a few examples
    print("\nSample Comparisons:")
    for i in range(min(3, len(predictions))):
        print(f"Ref:  {references[i]}")
        print(f"Pred: {predictions[i]}")
        print("-" * 20)

if __name__ == "__main__":
    calculate_metrics()