import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Azure Speech Config
speech_key = os.getenv('SPEECH_KEY')
speech_region = os.getenv('SPEECH_REGION')
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
speech_config.speech_synthesis_voice_name = 'en-US-JennyNeural'

# Initialize speech synthesizer
audio_config_synthesizer = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config_synthesizer)

def test_tts():
    """Function to test if Azure TTS is working properly."""
    test_text = "Hello, this is a test of the text-to-speech system."
    result = speech_synthesizer.speak_text_async(test_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesis succeeded.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error and cancellation_details.error_details:
            print(f"Error details: {cancellation_details.error_details}")

# Run the TTS test function
test_tts()
