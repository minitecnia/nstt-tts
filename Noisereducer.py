##-------------------------------------------------------------------------
# Centro de Tecnologías Emergentes del Ejército de Tierra (CTEET)
# Prueba de concepto: Instrucción por voz con whisper
# @Author: Francisco Jose Ochando Terreros
# @Test: Antonio Mejías Vello, Alejandro Gomez Sierra
# @Debugging: Francisco Jose Ochando Terreros
# version 0.1
#-------------------------------------------------------------------------
import wave
from scipy.io import wavfile # type: ignore
from Audio import Record as rec
import noisereduce as nr # type: ignore # https://pypi.org/project/noisereduce/
import xml.etree.ElementTree as et
import warnings
import argparse

warnings.filterwarnings('ignore')


class noisereducer():

    def __init__(self):
        self.stationary = True
        self.prop_decrease = float(0.9)
        self.freq_mask_smooth_hz = int(500)
        self.time_mask_smooth_ms = int(50)
        self.thresh_n_mult_nonstationary = int(1)
        self.sigmoid_slope_nonstationary = int(10)
        self.n_std_thresh_stationary = float(1.5)
        self.time_constant_s = float(2.0)
        self.n_fft = int(1024)
        self.chunk_size = int(60000)
        self.padding = int(30000)

    def setparams(self, config):
        tree = et.parse(config)
        root = tree.getroot()
        for reducer in root.findall('reducer'):
            self.stationary = reducer.find('stationary').text
            self.prop_decrease = reducer.find('prop_decrease').text
            self.freq_mask_smooth_hz = reducer.find('freq_mask_smooth_hz').text
            self.time_mask_smooth_ms = reducer.find('time_mask_smooth_ms').text
            self.time_constant_s = reducer.find('time_constant_s').text
            self.n_fft = reducer.find('n_fft').text
    
    def getparams(self, config):
        tree = et.parse(config)
        root = tree.getroot()
        for reducer in root.findall('reducer'):
            stationary = reducer.find('stationary').text
            prop_decrease = reducer.find('prop_decrease').text
            hz = reducer.find('freq_mask_smooth_hz').text
            ms = reducer.find('time_mask_smooth_ms').text
            s = reducer.find('time_constant_s').text
            fft = reducer.find('n_fft').text
        return stationary, prop_decrease, hz, ms, s, fft

    def read_noisyfile(path):
        rate, data = rec.read_audio(path)
        return rate, data

    def write_cleanfile(self, reduced_noise, rate, clean_file_path):
        # save audio file
        rec.write(self, reduced_noise, rate, clean_file_path)

    def reduce_ruido(self,
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
    
        # load audio data
        rate, data = wavfile.read(file_path)
        #print(data)

        # perform noise reduction
        reduced_noise = nr.reduce_noise(y = data, 
                                        sr = rate, 
                                        stationary = self.stationary,
                                        prop_decrease = self.prop_decrease, 
                                        time_constant_s = self.time_constant_s,
                                        freq_mask_smooth_hz = self.freq_mask_smooth_hz,
                                        time_mask_smooth_ms = self.time_mask_smooth_ms,
                                        thresh_n_mult_nonstationary = self.thresh_n_mult_nonstationary,
                                        sigmoid_slope_nonstationary = self.sigmoid_slope_nonstationary,
                                        n_std_thresh_stationary = self.n_std_thresh_stationary,
                                        tmp_folder = tmp_folder,
                                        chunk_size = chunk_size,
                                        padding = padding,
                                        n_fft = self.n_fft,
                                        win_length = win_length,
                                        hop_length = hop_length,
                                        clip_noise_stationary = clip_noise_stationary, 
                                        use_tqdm = use_tqdm, 
                                        n_jobs = n_jobs,
                                        use_torch = use_torch,
                                        device = device)
        
        print(reduced_noise)
        # save audio file
        rec.write(self, reduced_noise, rate, clean_file_path)

if __name__=='__main__':

    parser=argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("stationarynoise")
    path = "/Users/tucan/Documents/notebooks/audio2text/audio/"
    entrada = path + "websdr_recording_2025-01-11T11_27_15Z_7116.9kHz.wav"
    #salida = '../clean_samples/clean_Apollo_11_launch_day_communication_relayed_through_Canary_Station.wav'
    salida = path + "Denoised" + "websdr_recording_2025-01-11T11_27_15Z_7116.9kHz.wav"


    args=parser.parse_args()
    if args.stationarynoise=="Yes":
        reducer = noisereducer()
        reducer.reduce_ruido(entrada, salida, True, prop_decrease=0.9,n_std_thresh_stationary=2.0)
        denoised = salida
    else:
        reducer = noisereducer()
        reducer.reduce_ruido(entrada, salida, False, prop_decrease=0.8)
        denoised = salida