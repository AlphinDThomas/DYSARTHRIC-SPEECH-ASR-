import whisper

# Load model (change size if needed)
model = whisper.load_model("medium")  
# options: tiny, base, small, medium, large

# Replace with your audio file path
audio_file = "audio.wav"

# Transcribe
result = model.transcribe(audio_file)

# Print result
print("\nTranscription:")
print(result["text"])
