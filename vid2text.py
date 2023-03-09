
import io
import subprocess
import numpy as np
import streamlit as st
import pytube
import whisper
import nltk

# Download the NLTK tokenizer models
nltk.download('punkt')

# Load the WhisPer speech recognition model
model = whisper.load_model("base")

# Define a function to transcribe the audio file and return the text
def transcribe_audio(audio_data):
    # Convert the audio data to a WAV file using ffmpeg
    ffmpeg_command = ['ffmpeg', '-y', '-i', '-', '-f', 'wav', '-']
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    wav_data, stderr = ffmpeg_process.communicate(audio_data.getbuffer())

    # Convert the WAV data to a NumPy array
    audio_array = np.frombuffer(wav_data, dtype=np.int16)

    # Transcribe the audio array and return the text
    txt = model.transcribe(audio_array)
    return txt['text']

# Define the Streamlit app
def main():
    st.title("Convert YouTube Audio to Text Transcription")

    # Get the YouTube link from the user
    video_url = st.text_input("Enter a YouTube link:")

    if video_url:
        try:
            # Download the audio from the YouTube video to memory
            video = pytube.YouTube(video_url)
            audio = video.streams.filter(only_audio=True).first()
            audio_data = io.BytesIO()
            audio.stream_to_buffer(audio_data)
            audio_data.seek(0)

            # Transcribe the audio file and show the text
            text = transcribe_audio(audio_data)
            st.header("Transcription")
            st.write(text)
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
