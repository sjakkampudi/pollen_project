import os
import csv

path1_seg = '/home/agtrivedi/.keras/datasets/train/images/1/train_SEGM'
path2_seg = '/home/agtrivedi/.keras/datasets/train/images/2/train_SEGM'
path3_seg = '/home/agtrivedi/.keras/datasets/train/images/3/train_SEGM'
path4_seg = '/home/agtrivedi/.keras/datasets/train/images/4/train_SEGM'

if os.path.isfile('./segm_ims.csv') == True:
	os.remove('segm_ims.csv')

with open('segm_ims.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file name', 'label'])
    for root, dirs, files in os.walk(path1_seg):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '1'])
    
    for root, dirs, files in os.walk(path2_seg):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '2'])

    for root, dirs, files in os.walk(path3_seg):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '3'])

    for root, dirs, files in os.walk(path4_seg):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '4'])

path1_obj = '/home/agtrivedi/.keras/datasets/train/images/1/train_OBJ'
path2_obj = '/home/agtrivedi/.keras/datasets/train/images/2/train_OBJ'
path3_obj = '/home/agtrivedi/.keras/datasets/train/images/3/train_OBJ'
path4_obj = '/home/agtrivedi/.keras/datasets/train/images/4/train_OBJ'

if os.path.isfile('./obj_ims.csv') == True:
        os.remove('obj_ims.csv')

with open('obj_ims.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file name', 'label'])
    for root, dirs, files in os.walk(path1_obj):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '1'])

    for root, dirs, files in os.walk(path2_obj):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '2'])

    for root, dirs, files in os.walk(path3_obj):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '3'])

    for root, dirs, files in os.walk(path4_obj):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '4'])

path1_mask = '/home/agtrivedi/.keras/datasets/train/images/1/train_MASK'
path2_maks = '/home/agtrivedi/.keras/datasets/train/images/2/train_MASK'
path3_mask = '/home/agtrivedi/.keras/datasets/train/images/3/train_MASK'
path4_mask = '/home/agtrivedi/.keras/datasets/train/images/4/train_MASK'

if os.path.isfile('./mask_ims.csv') == True:
        os.remove('mask_ims.csv')

with open('mask_ims.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file name', 'label'])
    for root, dirs, files in os.walk(path1_mask):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '1'])

    for root, dirs, files in os.walk(path2_mask):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '2'])

    for root, dirs, files in os.walk(path3_mask):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '3'])

    for root, dirs, files in os.walk(path4_mask):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '4'])


