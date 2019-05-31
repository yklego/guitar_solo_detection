import numpy as np 
import librosa
import os

filedir = "/home/bill317996/Guitar-Solo-Detection/Dataset/mp3/"
targetdir = "/home/lego/guitar_solo_detection/feature/mel/"
num = 0
for root , dirs, files in os.walk(filedir):
	files.sort()
	for f in files:
		if ".mp3" in f:
			print (f , ".....")
			path = os.path.join(root,f)
			num = int(f.split("_")[0])
			y, sr = librosa.load(path,sr = 22050)
			mel = librosa.feature.melspectrogram(y, sr = sr, hop_length=512)
			np.save(targetdir+"x"+str(num)+".npy",np.transpose(mel,(1,0)))
			print (f , "is saved")
