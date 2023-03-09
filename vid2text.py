
import streamlit as st
import pytube
import whisper
import nltk

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

    # Prompt the user to enter a YouTube link and a description
    video_url = st.text_input("Enter a YouTube link:")
    description = st.text_input("Describe the YouTube video (optional):")

    if video_url:
        try:
            # Download the audio from the YouTube video
            video = pytube.YouTube(video_url)
            audio = video.streams.filter(only_audio=True).first()
            audio_path = audio.download()

            # Transcribe the audio file and show the text
            text = transcribe_audio(audio_path)
            if description:
                st.write("Description:", description)
            st.header("Transcription")
            st.write(text)
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
