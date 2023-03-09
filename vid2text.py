
import io
import wave
import numpy as np
import streamlit as st
import pytube
import whisper
import nltk
from moviepy.video.io.VideoFileClip import VideoFileClip

# Download the NLTK tokenizer models
nltk.download('punkt')

# Load the WhisPer speech recognition model
model = whisper.load_model("base")

# Define a function to transcribe the audio file and return the text
def transcribe_audio(audio_data):
    # Read the audio data using the wave module
    with wave.open(audio_data, 'rb') as audio_file:
        audio_params = audio_file.getparams()
        audio_frames = audio_file.readframes(audio_params.nframes)

    # Convert the audio data to a NumPy array
    audio_array = np.frombuffer(audio_frames, dtype=np.int16)

    # Transcribe the audio array and return the text
    txt = model.transcribe(audio_array)
    return txt['text']

# Define the Streamlit app
def main():
    st.title("Convert YouTube MP4 Video Audio to Text Transcription")

    # Get the YouTube link from the user
    video_url = st.text_input("Enter a YouTube link:")

    if video_url:
        try:
            # Download the video from the YouTube link
            video = pytube.YouTube(video_url)
            video_stream = video.streams.filter(adaptive=True).first()
            video_data = io.BytesIO()
            chunk_size = 4096
            while True:
                chunk = video_stream.read(chunk_size)
                if not chunk:
                    break
                video_data.write(chunk)
            video_data.seek(0)

            # Extract the audio from the video and convert to a NumPy array
            video_clip = VideoFileClip(video_data)
            audio_clip = video_clip.audio
            audio_data = io.BytesIO()
            audio_clip.write_audiofile(audio_data, fps=16000, nbytes=2, codec='pcm_s16le')
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
