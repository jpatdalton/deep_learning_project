__author__ = 'jpatdalton'

'''
This file processes format #1 from http://ufldl.stanford.edu/housenumbers/
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import h5py
import random
import pickle


def extract_value(f, obj, label, count = 0):
    """Function to extract values from HDF5 object references and groups

    Args:
        f: h5py file object
        obj: object to extract
        label: label of feature
        count: counter for number of images

    Returns:
        Extracted value.
    """
    ref = obj[label]
    if len(ref) == 1:
        return [ref[0][0]]
    elif len(ref) < 6:
        return [f[ref[i][0]][0][0] for i in range(len(ref))]
    else:
        print 'Encountered length greater than 5 to extract, too long of a street number! Length is ' + str(len(ref)) + ' count is ' + str(count)
        return [f[ref[i][0]][0][0] for i in range(len(ref))]

# Can't load this file with scipy, need h5py to load.
def load_matlab_file(ipath, out_file_1):
    """Function to load .mat bounding box file provided from SVHN dataset and convert it into Python dictionaries

    Args:
        ipath: relative path to file
        out_file_1: file to save Python dictionaries to

    """
    f = h5py.File(ipath+'digitStruct.mat','r')

    count = 1
    data = dict()

    # Loop over elements in bbox dataset and create python dictionary from them.
    for bbox in f['digitStruct/bbox']:
        obj = f[bbox[0]]
        temp_dict = dict()
        temp_dict['height'] = extract_value(f, obj, 'height', count=count)
        temp_dict['label'] = extract_value(f, obj, 'label')
        temp_dict['left'] = extract_value(f, obj, 'left')
        temp_dict['top'] = extract_value(f, obj, 'top')
        temp_dict['width'] = extract_value(f, obj, 'width')
        # Check that we are getting equal amounts of data for all 5 attributes on each image
        if not(len(temp_dict['height']) == len(temp_dict['label']) == len(temp_dict['left']) == len(temp_dict['top']) == len(temp_dict['width'])):
            raise Exception('Not all sizes for png #' + count + " are equal, look into it")
        data[count] = temp_dict
        count += 1
        if count % 5000 == 0:
            print 'Translated ' + str(count) + ' bounding box entries for file ' + out_file_1

    # Write to pickle file so we have a python dictionary ready to load later, and don't have to deal with this .mat file
    try:
        with open(out_file_1,'wb') as out_file:
            pickle.dump(data, out_file)
    except Exception as e:
            print 'Save pickle data exception to file: ' + str(out_file) + ' - ' + str(e)
            raise e

def preprocess_images(ipath, out_file_1, out_file_2, data_rows):
    """Function to perform preprocessing of images.

    Imports training data from a Pickle file, and crops images based on bounding box data. Converts to greyscale and 40x40 pixel resolution.
    Saves the preprocessed data to out_file_2.

    There is commented out functionality to save the intermediate data dictionary to a file.  This is only useful if you want to use the
    intermediate bounding box dictionaries created.

    Args:
        ipath: relative path to images from SVHN dataset
        out_file_1: Pickled python dictionary data of bounding boxes
        out_file_2: File to save preprocessed data
        data_rows: Number of images in dataset to preprocess

    """
    with open(out_file_1, 'rb') as pickled_data:
        data_dict = pickle.load(pickled_data)

    '''
    Here we take the pickle created before and reload the data, then get coordinates for left, right, top and bottom of -entire- image.
    Our model will train on the full image, having 5 outputs for each digit and an added output for the length of the full number
    '''
    input_data = {}
    for key,value in data_dict.iteritems():
        temp_dict = {}
        temp_dict['label'] = [0 if int(l) == 10 else int(l) for l in value['label']]
        temp_dict['length'] = len(temp_dict['label'])
        temp_dict['top'] = min(value['top'])
        temp_dict['left'] = min(value['left'])
        temp_dict['bottom'] = max([sum(x) for x in zip(value['top'], value['height'])])
        temp_dict['right'] = max([sum(x) for x in zip(value['left'], value['width'])])
        input_data[key] = temp_dict

    # check to see 5 images are okay with these specs
    img_num = str(int(random.random()*data_rows))
    ida = input_data[int(img_num)]
    print 'Image #' + img_num + ' value is = ' + str(ida['label'])
    img=mpimg.imread(ipath + str(img_num) + '.png')
    plt.imshow(img[ida['top']:ida['bottom'],ida['left']:ida['right']])

    '''
    #OPTIONAL: Save this data here (uncomment code) for different preprocessing of your choice
    try:
        with open('test1.pickle', 'wb') as input_data_file:
            pickle.dump(input_data, input_data_file)
            print 'pickle dump success'
    except Exception as e:
        print 'Save pickle data exception to file: ' + str(input_data_file) + ' - ' + str(e)
        raise e

    with open('test1.pickle', 'rb') as pickled_data:
        data_dict = pickle.load(pickled_data)

    assert(len(data_dict) == train_data_rows)
    print 'Successfully imported image location data'
    '''

    # Crop images to 40x40, greyscale, and save to pickle.
    arr = list()
    labels = list()
    for k, val in input_data.iteritems():
        img=Image.open(ipath + str(k) + '.png')
        cim = img.crop((val['left'],val['top'],val['right'],val['bottom']))
        cim_resized = cim.resize((40,40), resample=Image.LANCZOS)
        n = cim_resized.convert('L')
        cropped = np.array(n).astype(np.float64)
        normalized_cropped_image = cropped - np.mean(cropped)
        arr.append(normalized_cropped_image)
        length = len(val['label'])
        labels.append(val['label'] + [10 for i in range(5 - length)])
        if k%5000 == 0:
            print 'processed ' + str(k) + ' images from file: ' + out_file_1

    p_item = {}
    p_item["dataset"] = np.array(arr)
    p_item["labels"] = np.array(labels)

    # Save data to file for model.py
    try:
        with open(out_file_2, 'wb') as pickle_file:
            pickle.dump(p_item,pickle_file)
            print 'pickle dump success to file ' + str(pickle_file)
    except Exception as e:
        print 'Save pickle data exception to file: ' + str(pickle_file) + ' - ' + str(e)
        raise e
