import numpy as np
import scipy.io.wavfile as wawe
import matplotlib.pyplot as plt
import os    
import librosa                  # Python library to process audio files for ML and AI apps
from scipy import signal


'''
    The main objective of this code is to: generate a "Spectrogram tensor", where Each spectrogram is normalized in frequency and amplitude.

    The spectrograms are generated from the monophonic .wav files in the "samples" folder (raw audio data taken from the sea measurements).

    The resulting tensor has dimentions: [files number, X , Y]

    The audio samples were obtained from different measurements ( sources ), then the following code is supposed to:
        - Normalize the duration of the audio samples.
        - Generate the spectrogram associated to the audio sample.
        - Store the Spectrograms tensor in a .py file.
'''

#---------------------------------------------------------------------------------
# DATA INGESTION
# Read and process the data stored in the  Samples folder.
#---------------------------------------------------------------------------------

print('Enter the Samples folder path')
folder          =   input()
file_list       =   os.listdir(folder)  
files_num       =   len(file_list)  #List the number of files within the selected folder

#---------------------------------------------------------------------------------
#  ETL ° PASO
#   Data re-sampling: Normalization of the sampling frequency
#   Time Normalization
## ---------------------------------------------------------------------------------

print('Enter the wished sampling frequency to normalize the audio data.')
freq_obj    =   int(input())
file_size   =   []           # To store the final size of each file
steps_int   =   1
steps       =   []
print('Enter the wished time duration of the final audio samples [Seg]')
time_duration       =   int(input())
max_samples         =   time_duration   *   freq_obj
list_trunc          =   []
size_list_trunc     =   0
final_sound         =   []

# Scan for files with time duration longer than "time_duration"

for j  in range(files_num): 
  data      =   os.path.join(folder, file_list[j]) 
  sound,f   = librosa.load( data,
                            sr      =   44100   ,
                            mono    =   True    ,
                            offset  =   0.0     ,
                            res_type='kaiser_best'
                        ) #sr freq_original
  sound_length = np.shape(sound)
  file_size.append(sound_length)  
  
  if sound_length[0]    >   max_samples:     # Check desired audio sample duration
    steps_int       =sound_length[0]//max_samples
    steps.append(steps_int)
    list_trunc.append(file_list[j])         #List of audio samples longer than the desired duration
    size_list_trunc =len(list_trunc)        


# Normalization and segmentation
z=0
t=0
Pxx=[]
if   size_list_trunc != 0 :     
  for x in range (size_list_trunc):
      data      =   os.path.join(   folder, 
                                    list_trunc[x]
                                ) #path to find the audio sample to process
      sonido,f  =   librosa.load(
                                    data                    ,
                                    sr      =   freq_obj    , 
                                    mono    =   True        , 
                                    offset  =   0.0         , 
                                    res_type='kaiser_best' 
                                )   #sr freq_obj
      #absoluto=
      #maximo=np.max(np.abs(sound))
      sound_normalized    =   (   np.abs(sound)   )   /   np.max(np.abs(sound))


      for i in range (steps[x]):
        
        r   =   i       *max_samples
        p   =   (i+1)   *max_samples 
        final_sound.append([])
        final_sound[i]      =   sound_normalized[r:p]
        freq, time, Sxx      =   signal.spectrogram( final_sound[i]          ,
                                                    fs      =   freq_obj    ,
                                                    nfft    =   256   
                                                    )
        Sxx =   10*np.log10(Sxx)        #   Data compression
        Sxx =   np.clip(Sxx, -150, 50)  #   Deal with -inf problem when 0amplitude is found
        Pxx =   np.append(Pxx,Sxx)
        t=t+1


  s=len(freq)
  h=len(time)
  print()
  Pxx=np.reshape(Pxx,(s,h,t))
  print (np.shape(Pxx))
   
else:
    print('All files are shorter than the specified duration') 

print('Enter the name for the Spectrogram tensor file')
e=input()
np.save(e,Pxx)          # La matriz de espectrograma logaritmica
np.save('Freq',freq)    # y axis for future plotting
np.save('Time',time)    # x axis for future plotting
