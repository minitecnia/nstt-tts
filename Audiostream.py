#-------------------------------------------------------------------------
# CTEET
# Iberian Noisy Speech To Text software v2.0 
# @Author: Francisco Jose Ochando Terreros
# @Test: 
# @Debugging: 
# version 2.0
# Uses Vosk, Kaldi recognizer
#-------------------------------------------------------------------------

import streamlit as st  # type:ignore
import numpy as np      # type:ignore
import os
import settings         # type:ignore
from Transcriber import Transcriber # type:ignore
from voskengine import read, load, recognizer # type: ignore
from vosk import Model, KaldiRecognizer, SetLogLevel # type:ignore
from Noisereducer import noisereducer # type: ignore
from Convert import convert # type: ignore
from Habla import play # type: ignore


#st.audio(audio_bytes, format="audio/ogg")

sample_rate = 44100  # 44100 samples per second
seconds = 1  # Note duration of 2 seconds
frequency_la = 440  # Our played note will be 440 Hz


path_audio = 'audio/'
path_models = "models/"


def nota(frequency):
    t = np.linspace(0, seconds, seconds * sample_rate, False)
    note = np.sin(frequency * t * 2 * np.pi)
    return note

def noise_options():
    denoise = st.radio("Denoise", ("Yes", "No"))
    level = float(st.slider("Denoising level", 0.25, 1.0, 0.75, 0.05)) # Denoising level
    return denoise, level


def dashboard(transcription = None):

    """Runs real-time object detection on video input using Ultralytics YOLOv8 in a Streamlit application."""
    #check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds

    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
    selected_audio = ""
    audio = ""

    # Set html page configuration
    st.set_page_config(page_title="Voice Streamlit App", layout="wide", initial_sidebar_state="auto", page_icon=settings.im)
   
    # Set Background image
    st.markdown(settings.background_image, unsafe_allow_html=True)
    st.markdown(settings.input_style, unsafe_allow_html=True)
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    # Set Title and subtitle
    st.markdown(settings.main_title_cfg, unsafe_allow_html=True)
    st.markdown(settings.sub_title_cfg, unsafe_allow_html=True)

    audio_file = open(path_audio + "Apollo_11_launch_day_communication_relayed_through_Canary_Station.wav", "rb")
    audio_bytes = audio_file.read()

    
    with st.sidebar:
        # Add logos in sidebar
        logo1 = "logos/speechgpt.png"
        logo2 = "logos/Coat_of_Arms_of_the_Spanish_Army_CIS_Command.png"

        # Display them in a 2x2 grid using Streamlit columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(logo1, use_column_width=True, width=80)
        
        with col2:
            st.image(logo2, use_column_width=True, width=80)

        # Add elements to vertical setting menu
        st.sidebar.title("Configuración NSTT")

    # Paneles laterales
    tab1, tab2, tab3 = st.sidebar.tabs(["Model", "Audio", "Noise"])

    with tab1:
        st.header("Model")
        available_models = []
        if os.path.exists(path_models) and os.path.isdir(path_models):
            local_models = os.listdir(path_models)
            available_models.extend(local_models)  # add local files to the available file list
        selected_model = st.selectbox("Models", available_models)
        mimodelo = path_models + selected_model
        if st.button("Load"):
            with st.spinner("Model " + selected_model + " is loading..."):
                #print(mimodelo)
                transcriber = Transcriber(mimodelo)
    
    with tab2:
        st.header("Audio")
        # Add dropdown menu for audio selection
        # List audios from the path
        available_audio = []
        if os.path.exists(path_audio) and os.path.isdir(path_audio):
            local_audio = os.listdir(path_audio)
            available_audio.extend(local_audio)  # add local files to the available file list
        
        # Add audio source selection dropdown
        source = st.selectbox(
            "Audio source", ("Mic", "Speaker", "File", "Streaming", "RTSP"),
        )

        if source == "Mic":
            #st.text_input("Device", "Micro")
            if st.button("Listen"):
                with st.spinner("Inferring text from microphone..."):
                    #modelo = load(path_models + selected_model)
                    rec = recognizer(modelo)
                    read(0, rec, "prueba.txt")
        if source == "Speaker":
            st.warning('Activar mezclador audio en modo grabacion', icon="⚠️")
            if st.button("Listen"):
                with st.spinner("Inferring text from microphone..."):
                    modelo = load(path_models + selected_model)
                    rec = recognizer(modelo)
                    read(2, rec, "prueba.txt")
        elif source == "File":
            with st.spinner("Audio is loading..."):
                selected_audio = st.selectbox("Audio files", available_audio)
                #st.sidebar.audio(path_audio + selected_audio, format="audio/wav", loop=False, sample_rate=sample_rate)
                audio = path_audio + selected_audio
                st.success("Audio loaded successfully!")
                if selected_audio.endswith(".wav"):
                    st.audio(audio, format="audio/wav", sample_rate=sample_rate)
                elif selected_audio.endswith(".mp3"):
                    st.audio(audio, format="audio/mpeg", sample_rate=sample_rate)
                elif selected_audio.endswith(".ogg"):
                    st.audio(audio, format="audio/ogg", loop=False, sample_rate=sample_rate)
            
            if st.button("Start"):
                with st.spinner("Inferring text from audio file..."):
                    print(audio)
                    transcriber = Transcriber(mimodelo)
                    transcription = transcriber.transcribe(audio)
                st.success("File transcript successfully!")
        elif source == "Streaming":
            st.text_input("URL", "https://www.youtube.com")
        elif source == "RTSP":
            st.text_input("URL", "https://www.youtube.com")
            if st.button("Start"):
                with st.spinner("Inferring text from audio file..."):
                    print(audio)
                    trans = Transcriber(path_models + selected_model)
                    transcription = trans.transcribe(audio)
                st.success("File transcript successfully!")

    with tab3:
        denoise, level = noise_options()
        if denoise:
            if audio.endswith(".mp3"):
                st.warning('Detectado MP3. Solo ficheros WAV', icon="⚠️")
                if st.button("Convert"):
                    audio_data = convert.mp3towav(audio)
                    denoiseaudio = audio[:-4]
                    print(denoiseaudio)
                    audio_data.export(denoiseaudio + "_denoised.wav", format="wav")
            else:
                if st.button("Denoise"):
                    reducer = noisereducer()
                    reducer.reduce_ruido(audio, path_audio + 'Denoised_' + selected_audio, True, prop_decrease=level)
                    denoised = path_audio + 'Denoised_' + selected_audio
                    st.success("Noise reduced in file:" + denoised)
        

    # Paneles principales
    tab1, tab2, tab3 = st.tabs(["Audio", "Transcription", "Text to speech"])

    with tab1:
        st.header("Audio original")
        if selected_audio != "":
            st.audio(audio)
        else:
            st.success("File not selected")
        
        st.header("Audio modificado")
        if path_audio + 'Denoised_' + selected_audio != path_audio + 'Denoised_':
            if audio.endswith(".wav"):
                st.audio(path_audio + 'Denoised_' + selected_audio)
            else:
                st.audio(audio)
            #st.audio(path_audio + 'Denoised_' + selected_audio, sample_rate=sample_rate)
        else:
            st.success("File not selected")

    with tab2:
        st.header("Transcripción y diarización")
        if transcription:
            st.text_input("Transcription", transcription['transcription'])
            #st.write(title)
            for x in transcription['transcription']:
                st.text(x)
        else:
            st.text_input("Transcription", "Sin transcripción")
        st.audio(nota(400), sample_rate=sample_rate)
    
    with tab3:
        subtab1, subtab2, subtab3 = st.tabs(["Saludos", "Comentarios", "Frases"])
        with subtab1:
            if st.button("Hola buenos días"):
                play("Hola, buenos días")
            if st.button("Buenas tardes"):
                play("Buenas tardes")
        with subtab2:
            if st.button("Quiero hacer un comentario"):
                play("Quiero hacer un comentario")
            if st.button("¿Puedo comentar algo?"):
                play("¿Puedo comentar algo?")
        with subtab3:
            Frase = st.text_input("Frase", "Escribe")
            if st.button("Habla"):
                play(Frase)



# Main function call
if __name__ == "__main__":
    dashboard()