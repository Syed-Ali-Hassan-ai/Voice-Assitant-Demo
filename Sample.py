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

warnings.filterwarnings("ignore")
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Azure Speech Config
speech_key = os.getenv('SPEECH_KEY')
speech_region = os.getenv('SPEECH_REGION')
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
# Set the voice name; you can change this to any available voice
speech_config.speech_synthesis_voice_name = 'en-US-AvaNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# Initialize the AI assistant
ai_assistant = AIVoiceAssistant()

DEFAULT_CHUNK_LENGTH = 10  # Length of each audio recording chunk in seconds

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence based on amplitude threshold."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    """Record an audio chunk and save it to a temporary WAV file."""
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check for silence in the recorded chunk
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True  # Indicates silence
        else:
            return False  # Indicates voice detected
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI's Whisper API."""
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def synthesize_speech(text):
    """Synthesize speech using Azure Text-to-Speech and play the audio."""
    # Perform text-to-speech
    result = speech_synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Assistant spoke successfully.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error and cancellation_details.error_details:
            print(f"Error details: {cancellation_details.error_details}")
            print("Did you set the speech resource key and region values?")

def main():
    """Main function to run the voice assistant."""
    audio = pyaudio.PyAudio()
    # Open microphone stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=1024)
    customer_input_transcription = ""

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"

            # Record audio chunk
            print("Listening...")
            if not record_audio_chunk(audio, stream):
                # Transcribe audio using OpenAI Whisper API
                transcription = transcribe_audio(chunk_file)
                os.remove(chunk_file)
                if transcription.strip() == "":
                    print("Could not understand audio.")
                    continue
                print(f"Customer: {transcription}")

                # Process customer input and get response from AI assistant
                output = ai_assistant.interact_with_llm(transcription)
                if output:
                    output = output.strip()
                    print(f"Assistant: {output}")
                    # Synthesize speech using Azure Text-to-Speech
                    synthesize_speech(output)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Close streams and terminate PyAudio
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
