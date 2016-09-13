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


'''
######  YOU NEED TO CHANGE THE BELOW PARAMETERS BEFORE RUNNING THIS SCRIPT  ######
'''

train_data_rows = 10000#202353#13068
ipath = 'data/extra/'
out_file_1 = 'extra_bbox_data.pk1'
out_file_2 = 'extra_cropped_images_greyscale.pickle'
'''
ipath = 'data/test/'
out_file_1 = 'test_bbox_data.pk1'
out_file_2 = 'test_cropped_images_greyscale.pickle'

ipath = 'data/train/'
out_file_1 = 'bbox_data.pk1'
out_file_2 = 'cropped_images_greyscale.pickle'
'''

def extract_value(f, obj, label, count = 0):
    '''Function to extract values from HDF5 object references and groups

    Args:
        f: h5py file object
        obj: object to extract
        label: label of feature
    Returns:
        Extracted value.
    '''
    ref = obj[label]
    if len(ref) == 1:
        return [ref[0][0]]
    elif len(ref) < 6:
        return [f[ref[i][0]][0][0] for i in range(len(ref))]
    else:
        print 'Encountered length greater than 5 to extract, too long of a street number! Length is ' + str(len(ref)) + ' count is ' + str(count)
        return [f[ref[i][0]][0][0] for i in range(len(ref))]

# Can't load this file with scipy, need h5py to load.
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
        print 'Count is ' + str(count)

# Write to pickle file so we have a python dictionary ready to load later, and don't have to deal with this .mat file
try:
    with open(out_file_1,'wb') as out_file:
        pickle.dump(data, out_file)
except Exception as e:
        print 'Save pickle data exception to file: ' + str(out_file) + ' - ' + str(e)
        raise e

'''
Importing training data that we already have preprocessed to a pickle file
33402 rows with attributes of left, top, height, width, label
'''
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
img_num = str(int(random.random()*train_data_rows))
ida = input_data[int(img_num)]
print 'Image #' + img_num + ' value is = ' + str(ida['label'])
img=mpimg.imread(ipath + str(img_num) + '.png')
plt.imshow(img[ida['top']:ida['bottom'],ida['left']:ida['right']])

'''
OPTIONAL: Save this data here (uncomment code) for different preprocessing of your choice
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
crop_data = {}
arr = list()
labels = list()
for k, val in input_data.iteritems():
    img=Image.open(ipath + str(k) + '.png')
    cim = img.crop((val['left'],val['top'],val['right'],val['bottom']))
    cim_resized = cim.resize((40,40), resample=Image.LANCZOS)
    n = cim_resized.convert('L')
    cropped = np.array(n).astype(np.float64)
    normalized_cropped_image = cropped - np.mean(cropped)#np.divide(cropped,255.0)#
    temp_dict = {}
    arr.append(normalized_cropped_image)
    length = len(val['label'])
    labels.append(val['label'] + [10 for i in range(5 - length)])
    if k%5000 == 0:
        print 'processed ' + str(k) + ' images'

p_item = {}
p_item["dataset"] = np.array(arr)
p_item["labels"] = np.array(labels)

# Save data to file for model.py
try:
    with open(out_file_2, 'wb') as pickle_file:
        pickle.dump(p_item,pickle_file)
        print 'pickle dump success'
except Exception as e:
    print 'Save pickle data exception to file: ' + str(pickle_file) + ' - ' + str(e)
    raise e

'''
data exploration to try to find out how to crop, rotate, resize data
- found that about half the images have higher height than width
- 2785 images have height > 2*width or vice versa
- 326 images have height > 3*width or vice versa

average height, width (respectively)
37.0859229986
36.2566014011
'''
