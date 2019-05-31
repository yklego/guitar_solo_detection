import numpy as np 

datadir = '/home/bill317996/Guitar-Solo-Detection/feature/basic/'
newdir = '/home/lego/guitar_solo_detection/feature/seg_basic/'
frame_num = 215
num_t = 0

for i in range (60):
	data_num = i+1
	xname = 'x' + str(data_num) + '.npy'
	yname = 'y' + str(data_num) + '.npy'
	xdir = datadir + xname
	ydir = datadir + yname
	xnewdir = newdir + xname
	ynewdir = newdir + yname

	x = np.transpose(np.load(xdir),(1,0))
	y = np.load(ydir)

	j = 0
	num_s = 0
	while j + frame_num <= len(y) :
		if  y[j:j+frame_num].all() == 0 or y[j:j+frame_num].all() == 1:
			num_t += 1
			num_s += 1
			x_seg = x[j:j+frame_num][:]
			if y[j:j+frame_num].all() == 0:
				y_seg = np.array([[0.]])
			elif y[j:j+frame_num].all() == 1:
				y_seg = np.array([[1.]])
			
			if data_num <=40 :
				np.save(newdir + 'train/' + 'x' + str(data_num) + '_' + str(num_s) + '.npy', x_seg)
				np.save(newdir + 'train/' + 'y' + str(data_num) + '_' + str(num_s) + '.npy', y_seg)

			elif data_num <= 50:
				np.save(newdir + 'val/' + 'x' + str(data_num) + '_' + str(num_s) + '.npy', x_seg)
				np.save(newdir + 'val/' + 'y' + str(data_num) + '_' + str(num_s) + '.npy', y_seg)

			elif data_num <= 60:
				np.save(newdir + 'test/' + 'x' + str(data_num) + '_' + str(num_s) + '.npy', x_seg)
				np.save(newdir + 'test/' + 'y' + str(data_num) + '_' + str(num_s) + '.npy', y_seg)
			print ([data_num,num_s])
			j+=frame_num
print('total:' + str(num_t))
		




