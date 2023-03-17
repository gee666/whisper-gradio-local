import asyncio
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import os

model = whisper.load_model("small")

def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    transcription = result.text
    print(transcription)
    return transcription

async def record_audio(filename, duration, samplerate, channels):
    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        await asyncio.sleep(duration)

    audio = np.concatenate(recording, axis=0)
    sf.write(filename, audio, samplerate)

async def transcribe():
    # Parameters
    filename = "recorded_audio.wav"
    duration = 5  # recording duration in seconds
    samplerate = 16000
    channels = 1

    while True:
        # Record audio
        print("Listening...")
        await record_audio(filename, duration, samplerate, channels)

        # Transcribe the recorded audio
        print("Processing...")
        transcription = inference(filename)

        # Delete the audio file
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print(f"Error: The file {filename} does not exist.")

async def main():
    await transcribe()

# Run the main coroutine
asyncio.run(main())
