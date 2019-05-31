import cfp
import numpy as np
import os

for root, dirr, file in os.walk('/home/bill317996/Guitar-Solo-Detection/Dataset/mp3/'):
	file.sort()
        for filename in file:
                if '.mp3' in filename:
                	song_num = filename.split('_')[0]
                	filepath = os.path.join(root,filename)
                	x, sr  = cfp.load_audio(filepath, sr=22050, mono=True, dtype='float32')
                	print('Loading song number: '+song_num+' ...')
                	Z,time,CenFreq,l0,l1,l2 = cfp.feature_extraction(x, sr, Hop=320, Window=2049, StartFreq=80.0, StopFreq=1350.0, NumPerOct=48) #E6
                	f= 'feature/' + filename.split('.')[0] + '.npy'
                	np.save(f,Z)
                	print('complete! the feature shape:')
                	print(np.load(f).shape)


"""
import matplotlib.pyplot as plt
def ticks(convert_array, n=5):
	l = int(len(convert_array)/(n+1))
	x = np.zeros(n)
	y = np.zeros(n)
	for i in range(n+1):
		if i == 0:
			continue
		x[i-1] = i*l
		y[i-1] = round(convert_array[(i*l)],2)
		print(y[i-1])
	return x, y
"""


"""plt.imshow(Z, cmap='terrain', origin='lower',aspect='auto')
rawT, newT = ticks(time)

rawF, newF = ticks(CenFreq,n=3)
plt.xticks(rawT, newT)
plt.xlabel('Time (Sec)')
plt.yticks(rawF, newF)
plt.ylabel('Freq (Hz)')
plt.show()"""
# f=file('feature/prayer_test.npy','w')
