import os
import csv

path1 = '/home/agtrivedi/.keras/datasets/train/images/1/train_SEGM'
path2 = '/home/agtrivedi/.keras/datasets/train/images/2/train_SEGM'
path3 = '/home/agtrivedi/.keras/datasets/train/images/3/train_SEGM'
path4 = '/home/agtrivedi/.keras/datasets/train/images/4/train_SEGM'

if os.path.isfile('./ims.csv') == True:
	os.remove('ims.csv')

with open('ims.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file name', 'label'])
    for root, dirs, files in os.walk(path1):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '1'])
    
    for root, dirs, files in os.walk(path2):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '2'])

    for root, dirs, files in os.walk(path3):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '3'])

    for root, dirs, files in os.walk(path4):
        for filename in files:
            writer.writerow([os.path.join(root,filename), '4'])
