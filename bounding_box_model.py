__author__ = 'jpatdalton'

import tensorflow as tf
import numpy as np
import timeit
import pickle
import matplotlib.pyplot as plt

training = True
train_data_rows = 33402
test_data_rows = 13068
extra_data_rows = 202353
batch_size = 64
num_channels = 1 # images are all greyscale
depth1 = 16
depth2 = 25
depth3 = 40
image_size = 40
filter1_size = 3
filter2_size = 3
filter3_size = 3
filter4_size = 3
filter5_size = 3
num_hidden = 128
num_labels = 1

test_path = 'data/test/'
test_file_1 = 'test_bbox_data.pk1'
test_file_2 = 'test_bbox_labels.pickle'
test_grey_file = 'test_images_greyscale.pickle'


train_path = 'data/train/'
train_file_1 = 'bbox_data.pk1'
train_file_2 = 'train_bbox_labels.pickle'
train_grey_file = 'train_images_greyscale.pickle'


with open(train_file_2, 'rb') as out_file:
    train_labels = pickle.load(out_file)
    print 'Train labels have been loaded.'

with open(test_file_2, 'rb') as out_file:
    test_labels = pickle.load(out_file)
    print 'Test labels have been loaded.'

with open(train_grey_file, 'rb') as out_file:
    train_dataset = pickle.load(out_file)
    print 'Train dataset has been loaded.'

with open(test_grey_file, 'rb') as out_file:
    test_dataset = pickle.load(out_file)
    print 'Test dataset has been loaded.'


count = 0
trimmed_ds = list()
trimmed_labels = list()
for ds, label in zip(train_dataset,train_labels):
    if ds.shape[0] <= 200 and ds.shape[1] <= 400:
        height = 200 - ds.shape[0]
        width = 400 - ds.shape[1]
        longer = np.concatenate((ds,np.zeros((height, ds.shape[1]))), axis=0)
        wider = np.concatenate((longer, np.zeros((200, width))), axis=1)
        trimmed_labels.append(label)
        trimmed_ds.append(wider)
    else:
        np.delete(train_dataset,1)
print count

x = list()
y = list()
for ds in train_dataset:
    x.append(ds.shape[1])
    y.append(ds.shape[0])

plt.scatter(x,y)
plt.xlabel('Width of Image')
plt.ylabel('Height of Image')
plt.title('Training data size')

train_dataset = np.array(train_dataset)
test_dataset = np.array(test_dataset)

print('Training set', train_dataset.shape, train_labels.shape)
print('Valid set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

'''
valid_dataset = train_dataset[-4000:,:,:]
valid_labels = train_labels[-4000:,:]
train_dataset = np.delete(train_dataset, np.s_[-4000:], axis=0)
train_labels = np.delete(train_labels, np.s_[-4000:], axis=0)
'''
graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32)
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 4))


    # Convolutional layer variables
    layer1_filter = tf.Variable(tf.truncated_normal([filter1_size, filter1_size, num_channels, depth1], stddev=0.05))
    layer1_biases = tf.Variable(tf.constant(0.001, shape=[depth1]))
    layer2_filter = tf.Variable(tf.truncated_normal([filter2_size, filter2_size, depth1, depth2], stddev=0.05))
    layer2_biases = tf.Variable(tf.constant(0.001, shape=[depth2]))
    layer3_filter = tf.Variable(tf.truncated_normal([filter3_size, filter3_size, depth2, depth3], stddev=0.05))
    layer3_biases = tf.Variable(tf.constant(0.001, shape=[depth3]))

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

        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [64, -1])
        print 'pool shape', pool.get_shape()
        print 'reshape', reshape.get_shape()
        hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)

        return (tf.matmul(hidden, layer5_weights1) + layer5_biases1), (tf.matmul(hidden, layer5_weights2) + layer5_biases2), \
            (tf.matmul(hidden, layer5_weights3) + layer5_biases3), (tf.matmul(hidden, layer5_weights4) + layer5_biases4)

    # Training computation to get 5 classifiers
    top_logit,left_logit,bottom_logit,right_logit = model(tf_train_dataset, train=True)
    print 'logit shape: ', top_logit.get_shape()
    print 'labels shape: ', tf_train_labels.get_shape()
    print 'label shape: ', tf_train_labels[:,0].get_shape()
    loss = tf.sqrt(tf.reduce_mean(tf.square(top_logit - tf_train_labels[:,0])) + tf.reduce_mean(tf.square(left_logit - tf_train_labels[:,1])) \
                          + tf.reduce_mean(tf.square(bottom_logit - tf_train_labels[:,2])) + tf.reduce_mean(tf.square(right_logit - tf_train_labels[:,3])))

    # Adam Optimizer is regarded as the most efficient optimizer for CNNs.
    optimizer = tf.train.AdamOptimizer(.001).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.pack([top_logit,left_logit,bottom_logit,right_logit])


num_steps = 10000
start_time = timeit.default_timer()
with tf.Session(graph=graph) as session:
    if training:
        tf.initialize_all_variables().run()
        print('Initialized')
    else:
        saver = tf.train.Saver()
        saver.restore(session, "saved_models/model22.ckpt")
        print("Model restored.")

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 250 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            #print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            #print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    #preds = test_prediction.eval()
    #print('Test accuracy: %.1f%%' % accuracy(preds, test_labels))
    #print('Test accuracy blocked: %.1f%%' % per_seq_accuracy(preds, test_labels))
    #print('Test accuracy coverage: %.1f%% at %.1f%% percent' % coverage_accuracy(preds, test_labels))
    if training:
        save_path = saver.save(session, "saved_models/model23.ckpt")
        print("Model saved in file: %s" % save_path)

elapsed = timeit.default_timer() - start_time
print ('TIME TO RUN: ' + str(elapsed))


