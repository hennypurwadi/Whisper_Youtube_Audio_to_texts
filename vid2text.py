
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
            audio = video.streams.filter(only_audio=True).first()
            audio_path = audio.download()
            
            # Transcribe the audio file and show the text
            text = transcribe_audio(audio_path)
            st.header("Transcription")
            st.write(text)
            
            # Create a download button for the text file
            if st.button("Download Text File"):
                with open("transcription.txt", "w") as f:
                    f.write(text)
                st.download_button(
                    label="Download Transcription",
                    data=open("transcription.txt", "rb").read(),
                    file_name="transcription.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(str(e))

# Run the Streamlit app
if __name__ == '__main__':
    main()
