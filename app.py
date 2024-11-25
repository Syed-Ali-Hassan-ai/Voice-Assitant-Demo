import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
import openai  # For OpenAI Whisper API
import azure.cognitiveservices.speech as speechsdk  # For Azure Text-to-Speech
from dotenv import load_dotenv
import warnings
from rag.AIVoiceAssistant import AIVoiceAssistant
import time

warnings.filterwarnings("ignore")
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Azure Speech Config
speech_key = os.getenv('SPEECH_KEY')
speech_region = os.getenv('SPEECH_REGION')
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_config.speech_synthesis_voice_name = 'en-US-AvaNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# Initialize the AI assistant
ai_assistant = AIVoiceAssistant()

# Audio parameters
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 2000  # Adjust threshold to better capture silence
SILENCE_DURATION = 3.0  # Duration (in seconds) of silence to consider as end of speech

def is_silence(data, max_amplitude_threshold=SILENCE_THRESHOLD):
    """Check if audio data contains silence based on amplitude threshold."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_until_silence(audio, stream):
    """Record audio until silence is detected."""
    frames = []
    silence_start = None

    while True:
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

        # Check for silence
        audio_data = np.frombuffer(data, dtype=np.int16)
        if is_silence(audio_data):
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_DURATION:
                break  # End recording when silence duration exceeds threshold
        else:
            silence_start = None  # Reset silence timer on detecting sound

    # Save recorded frames to temp file
    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
    
    return temp_file_path

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI's Whisper API."""
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def synthesize_speech(text):
    """Synthesize speech using Azure Text-to-Speech and play the audio."""
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Assistant spoke successfully.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error and cancellation_details.error_details:
            print(f"Error details: {cancellation_details.error_details}")

def main():
    """Main function to run the voice assistant."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    try:
        while True:
            print("Listening...")
            chunk_file = record_until_silence(audio, stream)
            print("Recording stopped.")

            transcription = transcribe_audio(chunk_file)
            os.remove(chunk_file)
            if transcription.strip() == "":
                print("Could not understand audio.")
                continue
            print(f"Customer: {transcription}")

            # Get response from AI assistant
            output = ai_assistant.interact_with_llm(transcription)
            if output:
                print(f"Assistant: {output.strip()}")
                synthesize_speech(output.strip())

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
