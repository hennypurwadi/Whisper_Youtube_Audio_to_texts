
import os
import streamlit as st
import pytube
import whisper
import nltk
import tempfile

# Download the NLTK tokenizer models
nltk.download('punkt')

# Load the WhisPer speech recognition model
model = whisper.load_model("base")

# Define a function to transcribe the audio file and return the text
def transcribe_audio(audio_path):
    txt = model.transcribe(audio_path)
    return txt['text']

# Define the Streamlit app
def main():
    st.title("Convert YouTube Audio to Text Transcription")

    # Get the YouTube link from the user
    video_url = st.text_input("Enter a YouTube link:")

    if video_url:
        try:
            # Download the audio from the YouTube video to a temporary file
            video = pytube.YouTube(video_url)
            audio = video.streams.filter(only_audio=True).first()
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                audio.stream_to_buffer(temp_file)
                audio_path = temp_file.name

            # Transcribe the audio file and show the text
            text = transcribe_audio(audio_path)
            st.header("Transcription")
            st.write(text)
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
