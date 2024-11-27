import warnings
import pyaudio # type: ignore
import json
import sys
from datetime import datetime
from vosk import Model, KaldiRecognizer, SetLogLevel # type: ignore

warnings.filterwarnings('ignore')
SetLogLevel(-1) # type: ignore

def read(DEVICE, recognizer, output_file):

    p = pyaudio.PyAudio()
    chunks = 16000
    stream = listen(p, inputdev=DEVICE)

    while True:
        f = open (output_file,'a')
        data = stream.read(chunks) #read in chunks of bytes
        if recognizer.AcceptWaveform(data): #accept waveform of input voice
            # Parse the JSON result and get the recognized text
            result = json.loads(recognizer.Result())
            #print(result)
            recognized_text = result['text']
            # Write recognized text to the file
            print(recognized_text)
            f.write(recognized_text + "\n")
            f.close()
            if end(recognized_text, " roller "):
                break
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio object
    p.terminate()

def load(model):
    modelo = Model(model)
    return modelo

def recognizer(model, rate=16000):
    rec = KaldiRecognizer(model, rate)
    return rec

def transcribe_file(modelo, fichero, idioma='es'):
    result = modelo.transcribe(fichero, language= idioma)
    return result

def transcribe(stream, recognizer, output_file_path):
    with open(output_file_path, "w") as output_file:
        print("Listening for speech. Say 'Termina' to stop.")
        # Start streaming and recognize speech
        while True:
            data = stream.read(4096)#read in chunks of 4096 bytes
            if recognizer.AcceptWaveform(data):#accept waveform of input voice
                # Parse the JSON result and get the recognized text
                result = json.loads(recognizer.Result())
                recognized_text = result['text']
            
                # Write recognized text to the file
                output_file.write(recognized_text + "\n")
                print(recognized_text)
            
                # Check for the termination keyword
                if "roller" in recognized_text.lower():
                    print("Termination keyword detected. Stopping...")
                    break
    return stream

def listen(p, channels=1, rate=16000, frames_buffer=8192, inputdev=1):
    stream = p.open(format=pyaudio.paInt16, 
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=inputdev,
                    frames_per_buffer=frames_buffer)
    return stream

def end(text, word):
    termina = False
    if word in text.lower():
        print("Termination keyword detected. Stopping...")
        return True
    else:
        return termina