import numpy as np 

datadir = '/home/lego/guitar_solo_detection/feature/cqt_wider/'
targetdir = '/home/bill317996/Guitar-Solo-Detection/feature/basic/'
newdir = '/home/lego/guitar_solo_detection/feature/cqt_wider_seg/'
frame_num = 215

train_0, train_1, val_0, val_1 = 0, 0, 0, 0

for i in range (60):
	data_num = i+1
	xname = 'x' + str(data_num) + '.npy'
	yname = 'y' + str(data_num) + '.npy'
	xdir = datadir + xname
	ydir = targetdir + yname
	xnewdir = newdir + xname
	ynewdir = newdir + yname

	x = np.load(xdir)
	y = np.load(ydir)

	j = 0
	while j + frame_num <= len(y):
		if y[j:j+frame_num].all() == 0:
			y_seg = np.array([0.])
			x_seg = x[j:j+frame_num][:]
			tag = '0/'
			if data_num <= 40 :
				train_0 += 1
				np.save(newdir + 'train/' + tag + 'x' + str(train_0) + '.npy', x_seg)
				np.save(newdir + 'train/' + tag + 'y' + str(train_0) + '.npy', y_seg)
				print ('Song #'+str(data_num)+" transfer as train 0 seg #" + str(train_0))
			elif data_num <= 50 :
				val_0 += 1
				np.save(newdir + 'val/' + tag + 'x' + str(val_0) + '.npy', x_seg)
				np.save(newdir + 'val/' + tag + 'y' + str(val_0) + '.npy', y_seg)
				print ('Song #'+str(data_num)+" transfer as val 0 seg #" + str(val_0))
		elif y[j:j+frame_num].all() == 1:
			y_seg = np.array([1.])
			x_seg = x[j:j+frame_num][:]
			tag = '1/'
			if data_num <= 40 :
				train_1 += 1
				np.save(newdir + 'train/' + tag + 'x' + str(train_1) + '.npy', x_seg)
				np.save(newdir + 'train/' + tag + 'y' + str(train_1) + '.npy', y_seg)
				print ('Song #'+str(data_num)+" transfer as train 1 seg #" + str(train_1))
			elif data_num <= 50 :
				val_1 += 1
				np.save(newdir + 'val/' + tag + 'x' + str(val_1) + '.npy', x_seg)
				np.save(newdir + 'val/' + tag + 'y' + str(val_1) + '.npy', y_seg)
				print ('Song #'+str(data_num)+" transfer as val 1 seg #" + str(val_1))
		"""elif data_num <= 60:
			np.save(newdir + 'test/' + 'x' + str(data_num) + '_' + str(num_s) + '.npy', x_seg)
			np.save(newdir + 'test/' + 'y' + str(data_num) + '_' + str(num_s) + '.npy', y_seg)"""
		j+=frame_num