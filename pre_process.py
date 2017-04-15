
import os
import time
import scipy.misc
import numpy as np
from multiprocessing.dummy import Pool as Parallel
import cPickle as pickle

user = '/home/avisek/arnav/'
data = 'DIV2K' #DIV2K
HR_train_dir = user+'SuperResolution/data/' + data + '_train_HR/'
LR_train_dir = user+'SuperResolution/data/' + data+ '_train_LR_unknown/X4/'
#LR_valid_dir = '/users/TeamVideoSummarization/SuperResolution/DIV2K_valid_LR_unknown/X2/'

save_path_lr = user+'SuperResolution/patches/' + data + '_train_LR_unknown/X4/'
save_path_hr = user+'SuperResolution/patches/' + data + '_train_HR/X4/'

count, g_count = 0, 0
max_file_size = 3000
im_size = 96
scale = 4#96 / im_size
stride = 32

image_list_hr = np.zeros((max_file_size, im_size*scale, im_size*scale, 3))
image_list_lr = np.zeros((max_file_size, im_size, im_size, 3))

def stack_images(image_id):
	global count, g_count
	# patch dimensions for images
	patch_h = im_size * scale
	patch_w = im_size * scale

	stride_h = stride * scale
	stride_w = stride * scale
	path = image_id.split('.')[0]
	# _id = image_id.split('/')[-1]
	# path = os.path.join(path, _id)
	image_hr = scipy.misc.imread(image_id)
	image_lr = scipy.misc.imread(image_id.replace(data+'_train_HR/', data+'_train_LR_unknown/X4/').replace('.png', 'x4.png'))
	print count, image_hr.shape, image_lr.shape
	h, w = image_hr.shape[:2]

	max_rows = h // patch_h #2 * (h // patch_h) + (h%stride_h)/stride_h
	max_cols = w // patch_w #2 *(w // patch_w) + (w%stride_h)/stride_h
	for row in range(max_rows):
		for col in range(max_cols):

			image_list_hr[count, :, :, :] = image_hr[row * stride_h: row * stride_h + patch_h, col * stride_w: col * stride_w + patch_w, :]
 			image_list_lr[count, :, :, :] = image_lr[row * stride : row * stride + im_size, col * stride : col * stride + im_size, :]

			count = count + 1
			if count == max_file_size:				 
				count = 0
				np.save(save_path_hr + 'data_' + str(g_count) + '.npy', image_list_hr)
				np.save(save_path_lr + 'data_' + str(g_count) + '.npy', image_list_lr)
				print 'saving file', g_count
				g_count += 1
			#new_img_name = path+'_{}_{}.png'.format(row, col)
			#scipy.misc.imsave(new_img_name, new_img)
	return


hr_train_images = [os.path.join(HR_train_dir, img) for img in os.listdir(HR_train_dir)]
lr_train_images = [os.path.join(LR_train_dir, img) for img in os.listdir(LR_train_dir)]
#lr_valid_images = [os.path.join(LR_valid_dir, img) for img in os.listdir(LR_valid_dir)]

total_hr_train_images = len(hr_train_images)
total_lr_train_images = len(lr_train_images)
#total_lr_valid_images = len(lr_valid_images)

for image in hr_train_images:
	stack_images(image)

if count < max_file_size:
	np.save(save_path_hr+'data_'+str(g_count)+'.npy', image_list_hr[:count])
	np.save(save_path_lr+'data_'+str(g_count)+'.npy', image_list_lr[:count])
	print "Saving Last file", g_count

print 'Pre processing High Resolution training images : '
start = time.time()

