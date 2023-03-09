
import streamlit as st
import pytube
import io
import os
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types

# Set up the Google Cloud Speech-to-Text client
client = speech_v1.SpeechClient()

# Define a function to transcribe the audio file and return the text
def transcribe_audio(audio_file):
    # Convert the audio data to a format recognized by the Google Cloud API
    audio = types.RecognitionAudio(content=audio_file.read())

    # Set the configuration for the transcription
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US')

    # Perform the transcription and get the text
    response = client.recognize(config, audio)
    text = ''.join(result.alternatives[0].transcript for result in response.results)

    return text

# Define the Streamlit app
def main():
    st.title("Convert YouTube Audio to Text Transcription")

    # Get the YouTube link from the user
    video_url = st.text_input("Enter a YouTube link:")

    if video_url:
        try:
            # Download the audio from the YouTube video
            video = pytube.YouTube(video_url)
            audio = video.streams.filter(only_audio=True).first()
            audio_file = io.BytesIO()
            audio.stream_to_buffer(audio_file)
            audio_file.seek(0)

            # Transcribe the audio file and show the text
            text = transcribe_audio(audio_file)
            st.header("Transcription")
            st.write(text)
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
