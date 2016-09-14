__author__ = 'jpatdalton'

import pickle
import numpy as np
from PIL import Image

train_data_rows = 33402
test_data_rows = 13068
extra_data_rows = 202353

test_path = 'data/test/'
test_file_1 = 'test_bbox_data.pk1'
test_file_2 = 'test_bbox_labels.pickle'
test_grey_file = 'test_images_greyscale.pickle'


train_path = 'data/train/'
train_file_1 = 'bbox_data.pk1'
train_file_2 = 'train_bbox_labels.pickle'
train_grey_file = 'train_images_greyscale.pickle'


def make_image_greyscale(path_to_files, num_images, out_file):
    """Function to convert testing/training data into greyscale numpy arrays for model to train on

    Args:
        path_to_files: File path with images.
        num_images: Number of images in dataset.
        out_file: Output file to write array to.
    """

    arr = list()
    for i in range(1, num_images+1):
        img = Image.open(path_to_files + str(i) + '.png')
        n = img.convert('L')
        norm_n = n - np.mean(n)
        arr.append(np.array(norm_n))
        if i % 5000 == 0:
            print 'processed ' + str(i) + ' images'

    with open(out_file, 'wb') as input_data_file:
        pickle.dump(arr, input_data_file)
        print 'pickle dump success to file ' + out_file


def create_bounding_box_labels(file1, file2, num_rows):
    """This function creates labels for the top, left, bottom, and right positions of the bounding boxes for the data

    Args:
        file1: The file that contains individual bounding box info.
        file2: The file that will be written to
    """

    with open(file1, 'rb') as pickled_data:
        data_dict = pickle.load(pickled_data)

    input_data = np.zeros((num_rows, 4))
    for key,value in data_dict.iteritems():
        input_data[key-1, 0] = (min(value['top']))
        input_data[key-1, 1] = (min(value['left']))
        input_data[key-1, 2] = (max([sum(x) for x in zip(value['top'], value['height'])]))
        input_data[key-1, 3] = (max([sum(x) for x in zip(value['left'], value['width'])]))

    with open(file2, 'wb') as input_data_file:
        pickle.dump(input_data, input_data_file)
        print 'pickle dump success'


def run():
    make_image_greyscale(train_path, train_data_rows, train_grey_file)
    make_image_greyscale(test_path, test_data_rows, test_grey_file)
    create_bounding_box_labels(test_file_1, test_file_2, test_data_rows)
    create_bounding_box_labels(train_file_1, train_file_2, train_data_rows)
