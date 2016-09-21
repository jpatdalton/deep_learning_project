__author__ = 'jpatdalton'

'''This contains the model and training data for the number recognizer.

Values can be 0 through 10.  10 represents no value, all other digits represent their own number.

I used https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb as template
for model and training execution.

The training, test, and extra data is from 
'''

from PIL import Image
import tensorflow as tf
import numpy as np
import random
import timeit
import pickle

# Constants
training = False    # Change this if you want to train on more data!
coverage_confidence_level = .885
train_data_rows = 33402
test_data_rows = 13068
extra_data_rows = 202353
train_path = 'data/train/'
batch_size = 128
num_channels = 1    # Images are all greyscale
depth1 = 16
depth2 = 25
depth3 = 36
filter1_size = 3
filter2_size = 3
filter3_size = 3
filter4_size = 3
filter5_size = 3
image_size = 40
num_hidden = 256
num_labels = 11

'''
Importing training data that we already have preprocessed to a pickle file
33402 rows with two entries in a dictionary: 'dataset' and 'labels'
'''
# Test data
def load_pickles(test_file, train_file):
    with open(test_file, 'rb') as pickled_data:
        test_data_dict = pickle.load(pickled_data)

    # Train data
    with open(train_file, 'rb') as pickled_data_train:
        train_data_dict = pickle.load(pickled_data_train)

    # Extra data
    #with open('extra_cropped_images_greyscale.pickle', 'rb') as pickled_data_train:
    #    extra_data_dict = pickle.load(pickled_data_train)

    test_ds = test_data_dict['dataset']
    test_labels = test_data_dict['labels']
    train_ds = train_data_dict['dataset']
    train_labels = train_data_dict['labels']
    #extra_ds = extra_data_dict['dataset']
    #extra_labels = extra_data_dict['labels']

    image_size = train_ds.shape[1]

    # Open up random image from all datasets to make sure everything is ok
    def display_random_image(ds, labels):
        img_num = int(random.random()*10000)
        ida = ds[int(img_num)]
        print 'Image #' + str(img_num) + ' value is = ' + str(labels[img_num])
        # images are greyscale and have mean subtracted, so just add 100 to each value
        im = Image.fromarray(np.add(ida, 100))
        im.show()

    display_random_image(test_ds, test_labels)
    display_random_image(train_ds, train_labels)
    #display_random_image(extra_ds, extra_labels)

    assert len(train_ds) == train_data_rows
    assert len(test_ds) == test_data_rows
    #assert len(extra_ds) == extra_data_rows

    # If there is a label with more than 5 digits we trim the end digit(s).  This model is only for up to 5 digit numbers.
    t_labels = list()
    for i in range(len(train_labels)):
        if len(train_labels[i]) != 5:
            print i, train_labels[i]
            t_labels.append(np.array(train_labels[i][:5]))
        else:
            t_labels.append(np.array(train_labels[i]))
    t_labels = np.array(t_labels)
    del train_labels
    train_labels = t_labels
    del t_labels

    assert train_labels.shape[1] == test_labels.shape[1] #== extra_labels.shape[1]

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        return dataset, labels

    size_dict = dict()
    for i in range(1,7):
        size_dict[i] = 0
    for label in train_labels:
        size_dict[len(''.join(['' if int(x) == 10 else str(x) for x in list(label)]))] += 1


    train_dataset, train_labels = reformat(train_ds, train_labels)
    test_dataset, test_labels = reformat(test_ds, test_labels)
    #extra_dataset, extra_labels = reformat(extra_ds, extra_labels)

    # Validation data comes from last 4000 training instances.  We delete these instances from training set.
    valid_dataset = train_dataset[-4000:,:,:]
    valid_labels = train_labels[-4000:,:]
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    train_dataset = np.delete(train_dataset, np.s_[-4000:], axis=0)
    train_labels = np.delete(train_labels, np.s_[-4000:], axis=0)

    # OPTIONAL - add extra data to training and validation data.
    '''
    train_dataset = np.concatenate(train_dataset, extra_dataset[:40000])
    valid_dataset = np.concatenate(valid_dataset,extra_dataset[40000:45000])
    train_labels = np.concatenate(train_labels, extra_labels[:40000])
    valid_labels = np.concatenate(valid_labels, extra_labels[40000:45000])
    '''

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Valid set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    #print('Extra set', extra_dataset.shape, extra_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def accuracy(predictions, labels):
    '''Calculates per digit accuracy.

    Each number has 5 digits that are judged, meaning we count the correctness of 10 - no digit.

    Ex: prediction argmax: [9 10 10 10 10] and label: [8 10 10 10 10] would yield 4/5 correct or 80%

    Args:
        predictions: 3D numpy array of softmax probabilities for each of 5 digit predictors.
        labels: The given label for the image.

    Returns:
        A float calculation of percent multiplied by 100 for readability
    '''
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels)) / predictions.shape[1] / predictions.shape[0]


def per_seq_accuracy(predictions, labels):
    '''Calculates per number (sequence) accuracy.

    Each house number is judged whether it is completely correct or not.

    Ex: prediction argmax: [8 9 10 10 10] and label: [8 10 10 10 10] would yield an incorrect value - 0%

    Args:
        predictions: 3D numpy array of softmax probabilities for each of 5 digit predictors.
        labels: The given label for the image.

    Returns:
        A float calculation of percent multiplied by 100 for readability
    '''
    return 100.0 * (len(np.where(np.sum(np.argmax(predictions, 2).T == labels,1) == 5)[0])) / predictions.shape[1]


def coverage_accuracy(predictions, labels):
    '''Calculates percent of data covered at a specific accuracy

    Constant:
        coverage_confidence_level: The minimum softmax calculation required to attempt to judge correctness of number

    Args:
        predictions: 3D numpy array of softmax probabilities for each of 5 digit predictors.
        labels: The given label for the image.

    Returns:
        A float of the percent of data covered and the accuracy of the covered data
    '''
    counted = 0
    correct = 0
    size = predictions.shape[1]
    for i in range(size):
        if len(np.where(np.max(predictions[:,i,:],1) > coverage_confidence_level)[0]) < 5:
            pass
        elif np.sum(np.argmax(predictions[:,i,:], 1) == labels[i]) < 5:
            counted+=1
        else:
            counted+=1
            correct+=1
    print 'counted: ' + str(counted) + ' predictions: ' + str(size) + ' correct: ' + str(correct)
    return 100.0*counted/size, 100.0*correct/counted


def create_and_run_model(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, load_model, save_model, restore_model=True, num_steps=10000):

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, 5))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Convolutional layer variables
        layer1_filter = tf.Variable(tf.truncated_normal([filter1_size, filter1_size, num_channels, depth1], stddev=0.05))
        layer1_biases = tf.Variable(tf.constant(0.001, shape=[depth1]))
        layer2_filter = tf.Variable(tf.truncated_normal([filter2_size, filter2_size, depth1, depth2], stddev=0.05))
        layer2_biases = tf.Variable(tf.constant(0.001, shape=[depth2]))
        layer3_filter = tf.Variable(tf.truncated_normal([filter3_size, filter3_size, depth2, depth3], stddev=0.05))
        layer3_biases = tf.Variable(tf.constant(0.001, shape=[depth3]))
        layer6_filter = tf.Variable(tf.truncated_normal([filter4_size, filter4_size, depth3, depth3], stddev=0.05))
        layer6_biases = tf.Variable(tf.constant(0.001, shape=[depth3]))
        layer7_filter = tf.Variable(tf.truncated_normal([filter5_size, filter5_size, depth3, depth3], stddev=0.05))
        layer7_biases = tf.Variable(tf.constant(0.001, shape=[depth3]))
        layer8_filter = tf.Variable(tf.truncated_normal([filter5_size, filter5_size, depth3, depth2], stddev=0.05))
        layer8_biases = tf.Variable(tf.constant(0.001, shape=[depth2]))

        # Fully connected layer variables
        layer4_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 16 * depth2, num_hidden], stddev=0.05))
        layer4_biases = tf.Variable(tf.zeros([num_hidden]))
        layer5_weights1 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.05))
        layer5_biases1 = tf.Variable(tf.constant(0.001, shape=[num_labels]))
        layer5_weights2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.05))
        layer5_biases2 = tf.Variable(tf.constant(0.001, shape=[num_labels]))
        layer5_weights3 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.05))
        layer5_biases3 = tf.Variable(tf.constant(0.001, shape=[num_labels]))
        layer5_weights4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.05))
        layer5_biases4 = tf.Variable(tf.constant(0.001, shape=[num_labels]))
        layer5_weights5 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.05))
        layer5_biases5 = tf.Variable(tf.constant(0.001, shape=[num_labels]))

        def model(data, train=None):
            '''Builds the 8 layer deep convolutional network.

            The first 6 layers are convolutional, and the last two are fully connected.

            The first 7 layers have one classifier, which turns into 5 classifiers before the last layer.

            Args:
                data: The data to run through the model
                train: Value that decides whether there is dropout (only used for training).  Defaults to no dropout

            Returns:
                5 logits calculated from data running through network
            '''

            conv = tf.nn.conv2d(data, layer1_filter, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,1,1,1], 'SAME')
            if train:
                pool = tf.nn.dropout(pool,.6)

            conv = tf.nn.conv2d(pool, layer2_filter, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], 'SAME')
            if train:
                pool = tf.nn.dropout(pool,.6)

            conv = tf.nn.conv2d(pool, layer3_filter, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer3_biases)
            pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,1,1,1], 'SAME')
            if train:
                pool = tf.nn.dropout(pool,.6)

            conv = tf.nn.conv2d(pool, layer6_filter, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer6_biases)
            pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], 'SAME')
            if train:
                 pool = tf.nn.dropout(pool,.6)

            conv = tf.nn.conv2d(pool, layer7_filter, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer7_biases)
            pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,1,1,1], 'SAME')
            if train:
                pool = tf.nn.dropout(pool,.6)

            conv = tf.nn.conv2d(pool, layer8_filter, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer8_biases)
            pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], 'SAME')
            if train:
                pool = tf.nn.dropout(pool,.6)

            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)

            return (tf.matmul(hidden, layer5_weights1) + layer5_biases1), (tf.matmul(hidden, layer5_weights2) + layer5_biases2), \
                (tf.matmul(hidden, layer5_weights3) + layer5_biases3), (tf.matmul(hidden, layer5_weights4) + layer5_biases4), \
                (tf.matmul(hidden, layer5_weights5) + layer5_biases5)

        # Training computation to get 5 classifiers
        logits1,logits2,logits3,logits4,logits5 = model(tf_train_dataset, train=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, tf_train_labels[:,0])) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, tf_train_labels[:,1])) \
                              + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits3, tf_train_labels[:,2])) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits4, tf_train_labels[:,3])) \
                              + tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits5, tf_train_labels[:,4])))

        # Adam Optimizer is regarded as the most efficient optimizer for CNNs.
        optimizer = tf.train.AdamOptimizer(.0005).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.pack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),tf.nn.softmax(logits4),tf.nn.softmax(logits5)])
        logits1,logits2,logits3,logits4,logits5 = model(tf_valid_dataset, train=False)
        valid_prediction = tf.pack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),tf.nn.softmax(logits4),tf.nn.softmax(logits5)])
        logits1,logits2,logits3,logits4,logits5 = model(tf_test_dataset, train=False)
        test_prediction = tf.pack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),tf.nn.softmax(logits4),tf.nn.softmax(logits5)])

    start_time = timeit.default_timer()
    with tf.Session(graph=graph) as session:
        if restore_model:
            saver = tf.train.Saver()
            saver.restore(session, load_model)
            print("Model restored.")
        else:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            print('Initialized')

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
        preds = test_prediction.eval()
        print('Test accuracy per digit: %.1f%%' % accuracy(preds, test_labels))
        print('Test accuracy per sequence: %.1f%%' % per_seq_accuracy(preds, test_labels))
        print('Test accuracy coverage: %.1f%% at %.1f%% percent' % coverage_accuracy(preds, test_labels))

        if not restore_model:
            save_path = saver.save(session, save_model)
            print("Model saved in file: %s" % save_path)

    elapsed = timeit.default_timer() - start_time
    print ('TIME TO RUN: ' + str(elapsed))




