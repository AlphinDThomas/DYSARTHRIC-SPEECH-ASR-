import os
import io
import soundfile as sf
from datasets import load_dataset, Audio

# ==========================================
#  SETTINGS
# ==========================================
INPUT_LIST_FILE = "test_files.txt"
OUTPUT_FOLDER = "TEST_DATASET_55"
MASTER_FILE = "ground_truth_master.txt"
# ==========================================

def save_test_data():
    # 1. Setup
    if not os.path.exists(INPUT_LIST_FILE):
        print(f"Error: Could not find {INPUT_LIST_FILE}")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. Read target list
    print(f"Reading target list from {INPUT_LIST_FILE}...")
    target_filenames = set()
    
    with open(INPUT_LIST_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 2:
                target_filenames.add(parts[-1])
    
    print(f"Looking for {len(target_filenames)} files & transcripts...")

    # 3. Load Dataset (Standard Mode + No Decode)
    print("Loading dataset...")
    dataset = load_dataset("abnerh/TORGO-database", split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # 4. Search and Save
    found_count = 0
    master_list = []

    print("Scanning dataset...")
    
    for sample in dataset:
        full_path = sample["audio"]["path"]
        filename = full_path.split("/")[-1].split("\\")[-1]

        if filename in target_filenames:
            try:
                # --- A. Save Audio ---
                audio_bytes = sample["audio"]["bytes"]
                array, sr = sf.read(io.BytesIO(audio_bytes))
                
                audio_path = os.path.join(OUTPUT_FOLDER, filename)
                sf.write(audio_path, array, sr)

                # --- B. Get Transcript ---
                # The column is usually 'transcription' or 'text'
                # (Based on your training code, it is 'transcription')
                text = sample.get("transcription", "")
                if not text:
                    text = sample.get("text", "[No text found]")

                # --- C. Save Individual Text File ---
                # e.g., "F01_Session1_005.txt"
                text_filename = filename.replace(".wav", ".txt")
                text_path = os.path.join(OUTPUT_FOLDER, text_filename)
                
                with open(text_path, "w", encoding="utf-8") as tf:
                    tf.write(text)

                # --- D. Add to Master List ---
                master_list.append(f"{filename} | {text}")

                print(f"[{found_count+1}] Saved: {filename} -> '{text}'")
                found_count += 1
                target_filenames.remove(filename)

            except Exception as e:
                print(f"Error saving {filename}: {e}")

        if len(target_filenames) == 0:
            break

    # 5. Save Master List
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(master_list))

    print("-" * 40)
    print(f"✅ Saved {found_count} audio/text pairs to '{OUTPUT_FOLDER}'")
    print(f"✅ Master list saved to '{MASTER_FILE}'")

if __name__ == "__main__":
    save_test_data()