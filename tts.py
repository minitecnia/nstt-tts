#-------------------------------------------------------------------------
# CTEET
# Class TTS; text to speech using w32com 
# @Author: Francisco Jose Ochando Terreros
# @Test: 
# @Debugging: Francisco Jose Ochando Terreros
# version 0.1
#
# 
#-------------------------------------------------------------------------

#import subprocess
import win32com.client as w32client # type: ignore


class tts:

    def playvoice(respuesta, language='en', rate=1):
        
        voz_sp = 'Microsoft Helena Desktop'
        voz_en = 'Microsoft Zira Desktop'
        
        speaker = w32client.Dispatch("SAPI.SpVoice")
        speaker.Rate = rate
        if (language == 'en'):
            speaker.Voice = speaker.GetVoices('Name='+ voz_en).Item(0)
        else:
            speaker.Voice = speaker.GetVoices('Name='+ voz_sp).Item(0)
        
        #voz = speaker.GetVoices().Item(1).GetDescription()
        speaker.Speak(respuesta)

    def listvoices(speaker):
        numKEY=[] # numero de voz
        nombreVALUE=[] # nombre de voz
        try:
            contador = 0
            while True:

                tipo_voz=speaker.GetVoices().Item(contador).GetDescription()
                numKEY.append(contador)
                nombreVALUE.append(tipo_voz[0:tipo_voz.index("-")-1])
                contador+=1
        except:
            pass

        return numKEY, nombreVALUE


if __name__=='__main__':

    KEY, VALUE = tts.listvoices(w32client.Dispatch("SAPI.SpVoice"))
    voxIdiomas=dict(zip(KEY,VALUE))
    print("\nvoxIdiomas =",voxIdiomas)

    test = "This is Apollo Control, the Canary Island Station has acquisition of Apollo 11 now. We'll continue to stand by live for any air-to-ground communication. We're showing an orbital weight of the combined vehicles of 297,914 pounds."

    resultado = tts.playvoice(test, 'en')
    
    

