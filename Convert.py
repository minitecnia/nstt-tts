from os import path
from pydub import AudioSegment # type: ignore
import sys

class convert():

    def mp3towav(src):
        # convert wav to mp3
        try:
            sound = AudioSegment.from_file(src, "mp3")
        except:
            sound = AudioSegment.from_file(src, format="mp4")
        #sound = AudioSegment.from_mp3(src)
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

# files
file = '2-3B-ENG-NS_cutted.mp3'
test_path = 'testaudio/Conversations/'
converted_path = 'testaudio/ConvertAudios/'  
path = './audio/'                                                                      
src = test_path + file
dst = converted_path + '2-3B-ENG-NS_cutted.wav'

if __name__=='__main__':

    if len(sys.argv) < 2:
        print("python convert.py <file mp3>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        src = path + str(sys.argv[1])
        dst = path + str(sys.argv[1]) + '.wav'

    audio = convert.mp3towav(src)
    audio.export(dst, format="wav")