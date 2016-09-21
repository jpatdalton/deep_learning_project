__author__ = 'jpatdalton'

import pickle_image_data
import predictor
import model

train_data_rows = 33402
test_data_rows = 13068
extra_data_rows = 202353

test_path = 'data/test/'
test_file_1 = 'test_bbox_data.pk1'
test_file_2 = 'test_bbox_labels.pickle'
test_grey_file = 'test_cropped_images_greyscale.pickle'


train_path = 'data/train/'
train_file_1 = 'bbox_data.pk1'
train_file_2 = 'train_bbox_labels.pickle'
train_grey_file = 'train_cropped_images_greyscale.pickle'

load_model = "saved_models/model12.ckpt"
save_model = "saved_models/model13.ckpt"

def execute(num_steps=5000):
    pickle_image_data.load_matlab_file(train_path, train_file_1)
    pickle_image_data.preprocess_images(train_path, train_file_1, train_grey_file, train_data_rows)

    pickle_image_data.load_matlab_file(test_path, test_file_1)
    pickle_image_data.preprocess_images(test_path, test_file_1, test_grey_file, test_data_rows)

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = model.load_pickles(test_grey_file, train_grey_file)
    model.create_and_run_model(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, load_model, save_model, restore_model=False, num_steps=num_steps)
