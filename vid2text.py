
import io
import wave
import numpy as np
import streamlit as st
import requests
import whisper
import nltk
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

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
            headers = {"Range": "bytes=0-"}
            response = requests.get(video_url, headers=headers, stream=True)
            video_data = io.BytesIO()
            for chunk in response.iter_content(chunk_size=4096):
                video_data.write(chunk)
            video_data.seek(0)

            # Check the duration of the video file
            video_clip = VideoFileClip(video_data)
            duration = video_clip.duration
            if duration is None:
                raise ValueError("Invalid video file: cannot read duration")
            st.write(f"Video duration: {duration} seconds")

            # Extract the audio from the video and save to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
                reader = FFMPEG_VideoReader(filename=video_data, inputdict={}, outputdict={'-f': 'wav'})
                audio_frames = reader.read_audio_frames()
                audio_data = np.concatenate(audio_frames)
                audio_file.writeframes(audio_data.tobytes())

                # Transcribe the audio file and show the text
                audio_file.seek(0)
                text = transcribe_audio(audio_file)
                st.header("Transcription")
                st.write(text)
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
