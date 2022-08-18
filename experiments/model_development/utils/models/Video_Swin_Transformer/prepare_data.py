import os, random
import pickle5 as pickle

# TODO: import translator(s) from ../../../translators.py
def SHAPE(label):
    return {
        -1: 0,
        1: 1
    }[label]

# TODO: link parameters to config
seed = 22
random.seed(seed)
data_root = '../../../data/'
train_split = 0.8
ann_file_train = os.path.join(data_root, 'ann_file_train.txt')
ann_file_val = os.path.join(data_root, 'ann_file_val.txt')

open(ann_file_train, 'w').close()
open(ann_file_val, 'w').close()

iteration_subdirectories = [subdirectory for subdirectory in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, subdirectory))]
num_iterations = len(iteration_subdirectories)
train_iterations = random.sample(range(num_iterations), int(train_split * num_iterations))

for iteration in iteration_subdirectories:
    frames_directory = os.path.join(data_root, iteration, 'human_readable', 'frames')
    num_frames = len([frame for frame in os.listdir(frames_directory) if os.path.isfile(os.path.join(frames_directory, frame))])

    with open(os.path.join(data_root, iteration, 'machine_readable', 'iteration_data.pickle'), 'rb') as f:
        iteration_data = pickle.load(f)
    annotation_label = SHAPE(iteration_data['label'])

    if int(iteration) in train_iterations:
        ann_file = ann_file_train
    else:
        ann_file = ann_file_val
    with open(ann_file, 'a') as f:
        f.write(' '.join([frames_directory, str(num_frames), str(annotation_label)]))
        f.write('\n')
