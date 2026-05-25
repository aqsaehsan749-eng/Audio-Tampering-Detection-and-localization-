import streamlit as st
import matplotlib.pyplot as plt
import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Audio Tampering Detection", layout="wide")
st.title("🎵 Audio Tampering Detection")
st.write("Upload audio file to check for tampering")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.success("File uploaded successfully!")
    
    with st.spinner("Analyzing audio..."):
        # Load audio
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Waveform
        st.subheader("1. Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set(title="Waveform - Look for sudden jumps or cuts")
        st.pyplot(fig)
        
        # 2. Spectrogram
        st.subheader("2. Spectrogram Analysis")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img=librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        ax2.set(title="Spectrogram - Look for unnatural breaks")
        plt.colorbar(img,format='%+2.0f dB', ax=ax2)
        st.pyplot(fig2)
        
        # 3. Basic Tampering Check
        st.subheader("3. Basic Tampering Check")
        intervals = librosa.effects.split(y, top_db=20)
        
        if len(intervals) > 1:
            st.warning(f"⚠️ Possible Tampering Detected: Found {len(intervals)} audio segments")
            st.write("Multiple cuts/silence gaps found in audio")
        else:
            st.success("✅ Audio looks authentic - No cuts detected")
        
        # 4. Audio Info
        duration = len(y) / sr
        st.info(f"**Audio Info:** Duration: {duration:.2f} sec | Sample Rate: {sr} Hz")

else:
    st.info("👆 Upload an audio file to start analysis")
