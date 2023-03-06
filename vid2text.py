
import streamlit as st
import pytube
import nltk
from pydub import AudioSegment
from pydub.utils import which
import whisper

# Download the NLTK tokenizer models
nltk.download('punkt')

# Load the WhisPer speech recognition model
model = whisper.load_model("base")

# Define a function to transcribe the audio file and return the text
def transcribe_audio(audio_path):
    text = model.transcribe(audio_path)
    return text['text']

# Define the Streamlit app
def main():
    st.title("Convert YouTube Audio to Text Transcription")
    
    # Get the YouTube link from the user
    video_url = st.text_input("Enter a YouTube link:")
    
    if video_url:
        try:
            # Download the audio from the YouTube video
            video = pytube.YouTube(video_url)
            audio_stream = video.streams.filter(only_audio=True).first()
            audio = AudioSegment.from_file(audio_stream.download(), format="mp4")
            
            # Set the path to the ffmpeg and ffprobe executables
            ffmpeg_path = which("ffmpeg")
            ffprobe_path = which("ffprobe")
            
            # Transcribe the audio file and show the text
            text = transcribe_audio(audio.set_channels(1).set_frame_rate(16000), ffmpeg=ffmpeg_path, ffprobe=ffprobe_path)
            st.header("Transcription")
            st.write(text)
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
