##-------------------------------------------------------------------------
# CTEET
# Prueba de concepto: Instrucción por voz con whisper
# @Author: Alejandro Gomez Sierra, Antonio Mejías Vello
# @Test: Antonio Mejías Vello, Francisco Jose Ochando Terreros
# @Debugging: Francisco Jose Ochando Terreros
# version 0.1
#-------------------------------------------------------------------------
import pyaudio
import wave
from scipy.io import wavfile # type: ignore
from pydub import AudioSegment # type: ignore
import librosa # type: ignore
import soundfile as sf
import noisereduce as nr # https://pypi.org/project/noisereduce/
import threading
import os
import time
import xml.etree.ElementTree as et
import numpy as np

class Record:

    def __init__(self, formato: str = pyaudio.paInt16, canales: int = 1, frecuencia: int = 44100, entrada: bool = True, chunk: int = 1024, path_grabacion = "./"):
        '''
        Inicia la instancia de la clase Record

        Parameters
        ----------
        formato : str, optional
            DESCRIPTION. The default is pyaudio.paInt16.
        canales : int, optional
            Número de canales del micrófono. The default is 1.
        frecuencia : int, optional
            Frecuencia de audio del micrófono. The default is 44100.
        entrada : bool, optional
            Indica si se utiliza un dispositivo de entrada. The default is True.
        chunk : int, optional
            Número de frames incluidos en cada buffer. The default is 1024.
        path_grabacion : str, optional
            Ruta en la que se guarda el archivo WAV de audio. The default is "C:/Users/Zelenza/IberIAn/Audio/".

        Returns
        -------
        None.
        '''
        self.frames = []
        self.p = None
        self.stream = None
        self.formato = formato
        self.canales = canales
        self.frecuencia = frecuencia
        self.entrada = entrada
        self.chunk = chunk
        self.path_grabacion = path_grabacion
        self.name = time.strftime("%Y%m%d-%H%M%S")

        # generar ruta de carpetas
        if not os.path.isdir(path_grabacion + "original_samples/"):
            os.mkdir(path_grabacion + "original_samples/")
        if not os.path.isdir(path_grabacion + "clean_samples/"):
            os.mkdir(path_grabacion + "clean_samples/")
        if not os.path.isdir(path_grabacion + "tmp/"):
            os.mkdir(path_grabacion + "tmp/")

    def start(self, device):
        '''
        Inicia la grabación de un audio

        Returns
        -------
        None.
        '''
        self.p = pyaudio.PyAudio()
        #device = 0
        self.name = time.strftime("%Y%m%d-%H%M%S")
        self.stream = self.p.open(format=self.formato, channels=self.canales, rate=self.frecuencia, input=self.entrada, input_device_index=device, frames_per_buffer=self.chunk)

    def stop(self):
        '''
        Detiene la grabación de un audio

        Returns
        -------
        None.
        '''
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
        
    def get(self, stop):
        while True:
            
            if stop():
                print("Cierre hilo de audio")
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                break
            else:
                data = self.stream.read(self.chunk)
                self.frames.append(data)

    def read(self, file):
        rate, data = wavfile.read(file)
        return rate, data
    
    def write(self, audio, rate, file_path):
        wavfile.write(file_path, rate, audio)
            
    def save(self, path):
        '''
        Guarda el audio en un archivo

        Returns
        -------
        None.
        '''
        wf = wave.open(path, "wb")
        wf.setnchannels(self.canales)
        wf.setsampwidth(self.p.get_sample_size(self.formato))
        wf.setframerate(self.frecuencia)
        wf.writeframes(b"".join(self.frames))
        wf.close()
        
        self.frames = []
            
    def delete(self, path):
        os.remove(path)

    def clean(self,
              original_file_path: str = "./grabacion.wav", 
              clean_file_path: str = "./clean_path/clean_file.wav",
              stationary: bool = False,
              prop_decrease: float = 1.0, 
              time_constant_s: float = 2.0,
              freq_mask_smooth_hz: int = 500,
              time_mask_smooth_ms: int = 50,
              thresh_n_mult_nonstationary: int = 1,
              sigmoid_slope_nonstationary: int = 10,
              n_std_thresh_stationary: float = 1.5,
              tmp_folder: [type] = None,
              chunk_size: int = 60000,
              padding: int = 30000,
              n_fft: int = 1024,
              win_length: [type] = None,
              hop_length: [type] = None,
              clip_noise_stationary: bool = True, 
              use_tqdm: bool = False, 
              n_jobs: int = 1,
              use_torch: bool = False,
              device: bool = "cuda"):
        '''
        Reduce el ruido de un audio y lo guarda en un archivo en formato WAV

        Parameters
        ----------
        stationary: bool, optional
            Whether to perform stationary, or non-stationary noise reduction, by default False
            
        prop_decrease: float, optional
            The proportion to reduce the noise by (1.0 = 100%), by default 1.0
            
        time_constant_s: float, optional
            The time constant, in seconds, to compute the noise floor in the non-stationary algorithm, by default 2.0
            
        freq_mask_smooth_hz: int, optional
            The frequency range to smooth the mask over in Hz, by default 500
            
        time_mask_smooth_ms: int, optional
            The time range to smooth the mask over in milliseconds, by default 50
            
        thresh_n_mult_nonstationary: int, optional
            Only used in nonstationary noise reduction., by default 1
            
        sigmoid_slope_nonstationary: int, optional
            Only used in nonstationary noise reduction., by default 10
            
        n_std_thresh_stationary: int, optional
            Number of standard deviations above mean to place the threshold between signal and noise., by default 1.5
            
        tmp_folder: [type], optional
            Temp folder to write waveform to during parallel processing. Defaults to default temp folder for python., by default None
        
        chunk_size: int, optional
            Size of signal chunks to reduce noise over. Larger sizes will take more space in memory, smaller sizes can take longer to compute. , by default 60000 padding : int, optional How much to pad each chunk of signal by. Larger pads are needed for larger time constants., by default 30000
            
        padding : int, optional 
            How much to pad each chunk of signal by. Larger pads are needed for larger time constants., by default 30000
            
        n_fft: int, optional
            length of the windowed signal after padding with zeros. The number of rows in the STFT matrix D is (1 + n_fft/2). The default value, n_fft=2048 samples, corresponds to a physical duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the default sample rate in librosa. This value is well adapted for music signals. However, in speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm., by default 1024
            
        win_length: [type], optional
            Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft. Smaller values improve the temporal resolution of the STFT (i.e. the ability to discriminate impulses that are closely spaced in time) at the expense of frequency resolution (i.e. the ability to discriminate pure tones that are closely spaced in frequency). This effect is known as the time-frequency localization trade-off and needs to be adjusted according to the properties of the input signal y. If unspecified, defaults to win_length = n_fft., by default None
            
        hop_length: [type], optional
            number of audio samples between adjacent STFT columns. Smaller values increase the number of columns in D without affecting the frequency resolution of the STFT. If unspecified, defaults to win_length // 4 (see below)., by default None
            
        use_tqdm: bool, optional
            Whether to show tqdm progress bar, by default False
            
        n_jobs: int, optional
            Number of parallel jobs to run. Set at -1 to use all CPU cores, by default 1
            
        use_torch: bool, optional
            Whether to use the torch version of spectral gating, by default False
            
        device: str, optional
            A device to run the torch spectral gating on, by default "cuda"
        Returns
        -------
        None.

        '''
    
        # load data
        rate, data = self.read(original_file_path)

        # Instantiate noise reduction open source algorithm
        reduced_noise = nr.reduce_noise(y = data, 
                                        sr = rate, 
                                        stationary = stationary,
                                        prop_decrease = prop_decrease, 
                                        time_constant_s = time_constant_s,
                                        freq_mask_smooth_hz = freq_mask_smooth_hz,
                                        time_mask_smooth_ms = time_mask_smooth_ms,
                                        thresh_n_mult_nonstationary = thresh_n_mult_nonstationary,
                                        sigmoid_slope_nonstationary = sigmoid_slope_nonstationary,
                                        n_std_thresh_stationary = n_std_thresh_stationary,
                                        tmp_folder = tmp_folder,
                                        chunk_size = chunk_size,
                                        padding = padding,
                                        n_fft = n_fft,
                                        win_length = win_length,
                                        hop_length = hop_length,
                                        clip_noise_stationary = clip_noise_stationary, 
                                        use_tqdm = use_tqdm, 
                                        n_jobs = n_jobs,
                                        use_torch = use_torch,
                                        device = device)
        
        #clean_file_path = self.clean_path + audiofile
        wavfile.write(clean_file_path, rate, reduced_noise)
    
    def config_noisereduction(self, configfile):
        tree = et.parse(configfile)
        root = tree.getroot()
        for reducer in root.findall('reducer'):
            freq_mask_smooth_hz = reducer.find('freq_mask_smooth_hz').text
            #freq_mask_smooth_ms = reducer.find('time_mask_smooth_ms').text
            #time_constant_s = reducer.find('time_constant_s').text
            print(freq_mask_smooth_hz)


class SplitAudio():

    def __init__(self):

        self.split_path = './tmp/'
    
    # Advanced function to split audio in chunks
    def split_audio_on_silence(self, input_file="", min_silence_duration=1, silence_threshold=25, max_chunk_duration=15):
        
        # Load audio file
        y, sr = librosa.load(input_file)
        
        # Detectar los intervalos de silencio
        intervals = librosa.effects.split(y, top_db=silence_threshold, frame_length=2048, hop_length=2000)
        print("librosa splitter detected intervals: " + str(len(intervals)))

        # Check if there is chunks larger than 15 seconds
        for i, (start, end) in enumerate(intervals):
            start = max(0, start)
            end = min(len(y), end )
            if (end-start)/sr>max_chunk_duration/1000:
                return False
        
        # Divide the audio sample in fragments based on silence interval
        if len(intervals) !=1 :
            for i, (start, end) in enumerate(intervals):
                start = max(0, start - sr * min_silence_duration)
                end = min(len(y), end + sr * min_silence_duration)

                # Returns the start and end time (in samples) of non-silent interval
                segment = y[start:end]
                
                # Save audio segment as a file
                output_file = os.path.join(self.split_path, f"segment_{i+1}.wav")
                sf.write(output_file, segment, sr)
                
                if ((end-start)/sr > 15):
                    return False
                    #break
                print(f"Segmento {i+1} guardado como {output_file}")
            return True
        
    # Function to split audio in chunks
    def split_audio_silence(input_file, output_dir, min_silence_duration=1, silence_threshold=25):
        
        # Load audio file
        y, sr = librosa.load(input_file)
        
        # Detectar los intervalos de silencio
        intervals = librosa.effects.split(y, top_db=silence_threshold, frame_length=2048, hop_length=512)
        
        print(intervals)
        # Dividir el audio en fragmentos basados en los intervalos de silencio
        
        for i, (start, end) in enumerate(intervals):
            start = max(0, start - sr * min_silence_duration)
            end = min(len(y), end + sr * min_silence_duration)
            segment = y[start:end]
            
            # Guardar el segmento de audio como un archivo
            output_file = os.path.join(output_dir, f"segment_{i+1}.wav")
            librosa.output.write_wav(output_file, segment, sr)
            print(f"Segmento {i+1} guardado como {output_file}")
    
    def forced_split_audio(self, input_file, segment_duration_ms, overlap_time=1000):
        
        sound = AudioSegment.from_file(input_file)
        
        # Gets total length of the audio in milliseconds
        total_duration_ms = len(sound)
        
        # Calculate the number of segments required
        num_segments = total_duration_ms // segment_duration_ms
        
        # Iterate through the audio and export segments
        for i in range(num_segments):
            start_time = i * segment_duration_ms
            if i!=0:
                start_time=start_time-overlap_time
            end_time = (i + 1) * segment_duration_ms
            segment = sound[start_time:end_time]
            output_file = f"{self.split_path}/segment_{i + 1}.wav"
            segment.export(output_file, format="wav")
        
    def on_segment_split_audio(self, input_file, output_file, time_start, interval):
        sound = AudioSegment.from_file(input_file)
        segment = sound[time_start:time_start+interval]
        segment.export(output_file, format = "wav")
    
    def get_duration(input_file):
        sound = AudioSegment.from_file(input_file)
        duration = len(sound)
        return duration

class reduce_noise():

    # Class for noise reducing. Configuring noise reducer parameters
    def __init__(self, configfile):
        tree = et.parse(configfile)
        root = tree.getroot()
        for reducer in root.findall('reducer'):
            freq_mask_smooth_hz = reducer.find('freq_mask_smooth_hz').text
            #freq_mask_smooth_ms = reducer.find('freq_mask_smooth_ms').text
            #time_constant_s = reducer.find('time_constant_s').text
            print('Frequency mask Smooth: ' + freq_mask_smooth_hz)

    # Function for reducing noise
    def clean(self,
              file_path: str = "./grabacion.wav", 
              clean_file_path: str = "./clean_path/clean_file.wav",
              stationary: bool = False,
              prop_decrease: float = 1.0, 
              time_constant_s: float = 2.0,
              freq_mask_smooth_hz: int = 500,
              time_mask_smooth_ms: int = 50,
              thresh_n_mult_nonstationary: int = 1,
              sigmoid_slope_nonstationary: int = 10,
              n_std_thresh_stationary: float = 1.5,
              tmp_folder: [type] = None,
              chunk_size: int = 60000,
              padding: int = 30000,
              n_fft: int = 1024,
              win_length: [type] = None,
              hop_length: [type] = None,
              clip_noise_stationary: bool = True, 
              use_tqdm: bool = False, 
              n_jobs: int = 1,
              use_torch: bool = False,
              device: bool = "cuda"):
        '''
        Parameters
        ----------
        stationary: bool, optional
            Whether to perform stationary, or non-stationary noise reduction, by default False
            
        prop_decrease: float, optional
            The proportion to reduce the noise by (1.0 = 100%), by default 1.0
            
        time_constant_s: float, optional
            The time constant, in seconds, to compute the noise floor in the non-stationary algorithm, by default 2.0
            
        freq_mask_smooth_hz: int, optional
            The frequency range to smooth the mask over in Hz, by default 500
            
        time_mask_smooth_ms: int, optional
            The time range to smooth the mask over in milliseconds, by default 50
            
        thresh_n_mult_nonstationary: int, optional
            Only used in nonstationary noise reduction., by default 1
            
        sigmoid_slope_nonstationary: int, optional
            Only used in nonstationary noise reduction., by default 10
            
        n_std_thresh_stationary: int, optional
            Number of standard deviations above mean to place the threshold between signal and noise., by default 1.5
            
        tmp_folder: [type], optional
            Temp folder to write waveform to during parallel processing. Defaults to default temp folder for python., by default None
        
        chunk_size: int, optional
            Size of signal chunks to reduce noise over. Larger sizes will take more space in memory, smaller sizes can take longer to compute. , by default 60000 padding : int, optional How much to pad each chunk of signal by. Larger pads are needed for larger time constants., by default 30000
            
        padding : int, optional 
            How much to pad each chunk of signal by. Larger pads are needed for larger time constants., by default 30000
            
        n_fft: int, optional
            length of the windowed signal after padding with zeros. The number of rows in the STFT matrix D is (1 + n_fft/2). The default value, n_fft=2048 samples, corresponds to a physical duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the default sample rate in librosa. This value is well adapted for music signals. However, in speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm., by default 1024
            
        win_length: [type], optional
            Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft. Smaller values improve the temporal resolution of the STFT (i.e. the ability to discriminate impulses that are closely spaced in time) at the expense of frequency resolution (i.e. the ability to discriminate pure tones that are closely spaced in frequency). This effect is known as the time-frequency localization trade-off and needs to be adjusted according to the properties of the input signal y. If unspecified, defaults to win_length = n_fft., by default None
            
        hop_length: [type], optional
            number of audio samples between adjacent STFT columns. Smaller values increase the number of columns in D without affecting the frequency resolution of the STFT. If unspecified, defaults to win_length // 4 (see below)., by default None
            
        use_tqdm: bool, optional
            Whether to show tqdm progress bar, by default False
            
        n_jobs: int, optional
            Number of parallel jobs to run. Set at -1 to use all CPU cores, by default 1
            
        use_torch: bool, optional
            Whether to use the torch version of spectral gating, by default False
            
        device: str, optional
            A device to run the torch spectral gating on, by default "cuda"
        Returns
        -------
        None.
        '''

        # load data
        rate, data = wavfile.read(file_path)

        # Instantiate noise reduction open source algorithm
        reduced_noise = nr.reduce_noise(y=data, 
                                        sr=rate, 
                                        stationary = stationary,
                                        prop_decrease = prop_decrease, 
                                        time_constant_s = time_constant_s,
                                        freq_mask_smooth_hz = freq_mask_smooth_hz,
                                        time_mask_smooth_ms = time_mask_smooth_ms,
                                        thresh_n_mult_nonstationary = thresh_n_mult_nonstationary,
                                        sigmoid_slope_nonstationary = sigmoid_slope_nonstationary,
                                        n_std_thresh_stationary = n_std_thresh_stationary,
                                        tmp_folder = tmp_folder,
                                        chunk_size = chunk_size,
                                        padding = padding,
                                        n_fft = n_fft,
                                        win_length = win_length,
                                        hop_length = hop_length,
                                        clip_noise_stationary = clip_noise_stationary, 
                                        use_tqdm = use_tqdm, 
                                        n_jobs = n_jobs,
                                        use_torch = use_torch,
                                        device = device)
        
        wavfile.write(clean_file_path, rate, reduced_noise)

class convert():

    def mp3towav(src):
        # convert wav to mp3
        sound = AudioSegment.from_mp3(src)
        if sound.channels > 1:
            # Convert stereo to mono channel
            sound = sound.set_channels(1)
        return sound
    
    def resample(audio, framerate):
        # framerate could be an integer (8000, 16000, etc.)
        resample_audio = audio.set_frame_rate(framerate)
        return resample_audio

    def exportowav(audio, destiny):
        audio.export(destiny, format='wav')


if __name__ == "__main__":
    # Testing software or command line use
    g = Record()
    n = reduce_noise('config.xml')
    mydir = os.getcwd()
    midir = "./"

    g.start()
    print("Grabando...")
    stop_threads=False
    t = threading.Thread(target=g.get, args =(lambda : stop_threads, )).start()
    time.sleep(10)
    print("Fin grabacion")
    stop_threads=True
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = midir + timestr + '_record.wav'
    print('File saved: ' + filename)
    g.save(filename)
