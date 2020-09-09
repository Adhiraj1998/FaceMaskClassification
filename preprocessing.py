
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# create directories
dataset_home = 'dataset/'
subdirs = ['train/', 'val/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['yes/', 'no/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'train'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train'
    if random() < val_ratio:
        dst_dir = 'val/'
    if file.startswith('y'):
        dst = dataset_home + dst_dir + 'yes/'  + file
        copyfile(src, dst)
    elif file.startswith('n'):
        dst = dataset_home + dst_dir + 'no/'  + file
        copyfile(src, dst)
        
dataset_home = 'dataset/'
subdirs = ['test']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['yes/', 'no/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
src_directory = 'test'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'test/'
    if file.startswith('y'):
        dst = dataset_home + dst_dir + 'yes/'  + file
        copyfile(src, dst)
    elif file.startswith('n'):
        dst = dataset_home + dst_dir + 'no/'  + file
        copyfile(src, dst)
