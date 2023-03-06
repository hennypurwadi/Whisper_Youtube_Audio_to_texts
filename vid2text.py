
import streamlit as st
import pytube
import whisper
import nltk

# Download the NLTK tokenizer models
nltk.download('punkt')

# Load the WhisPer speech recognition model
model = whisper.load_model("base")

# Define a function to transcribe the audio stream and return the text
def transcribe_audio(audio):
    text = model.transcribe(audio)
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
            audio = video.streams.filter(only_audio=True, progressive=False).order_by('abr').desc().first()
            
            # Transcribe the audio stream and show the text
            text = transcribe_audio(audio.stream())
            st.header("Transcription")
            st.write(text)
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
