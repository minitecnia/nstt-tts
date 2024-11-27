#-------------------------------------------------------------------------
# CTEET
# Habla; Programa de habla 
# @Author: Francisco Jose Ochando Terreros
# @Test: 
# @Debugging: Francisco Jose Ochando Terreros
# version 0.1
# Usa el modulo tts
#-------------------------------------------------------------------------

from tts import tts # type: ignore
import win32com.client as w32client # type: ignore
import pythoncom # type: ignore

pythoncom.CoInitialize()

# make request to google to get synthesis

def play(texto, language='es'):
    voz_sp = 'Microsoft Helena Desktop'
    voz_en = 'Microsoft Zira Desktop'
        
    speaker = w32client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 1
    if (language == 'en'):
        speaker.Voice = speaker.GetVoices('Name='+ voz_en).Item(0)
    else:
        speaker.Voice = speaker.GetVoices('Name='+ voz_sp).Item(0)
        
    #voz = speaker.GetVoices().Item(1).GetDescription()
    speaker.Speak(texto)
