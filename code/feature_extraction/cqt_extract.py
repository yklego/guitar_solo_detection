import numpy as np 
import librosa
import os

filedir = "/home/bill317996/Guitar-Solo-Detection/Dataset/mp3/"
targetdir = "/home/lego/guitar_solo_detection/feature/cqt_wider/"
num = 0
for root , dirs, files in os.walk(filedir):
	files.sort()
	for f in files:
		if ".mp3" in f:
			print (f , ".....")
			path = os.path.join(root,f)
			num = int(f.split("_")[0])
			y, sr = librosa.load(path,sr = 22050)
			cqt = librosa.cqt(y, sr = sr, hop_length=512, n_bins=168, bins_per_octave=24)
			np.save(targetdir+"x"+str(num)+".npy",cqt)
			print (f , "is saved")
