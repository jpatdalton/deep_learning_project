__author__ = 'jpatdalton'

import tensorflow as tf
from PIL import Image
import numpy as np
import random
import timeit
import pickle

# Constants
train_data_rows = 33402
test_data_rows = 13068
extra_data_rows = 202353
train_path = 'data/train/'
batch_size = 48
num_channels = 1 # images are all greyscale
depth1 = 12
depth2 = 16
depth3 = 24
filter1_size = 5
filter2_size = 5
filter3_size = 3
filter4_size = 3
filter5_size = 3
num_hidden = 64
num_labels = 11
'''
values can be -1 (no digit) through 9.  0 represents 10
'''

'''
Importing training data that we already have preprocessed to a pickle file
33402 rows with two entries in a dictionary: dataset and labels
'''
with open('test_cropped_images_greyscale.pickle', 'rb') as pickled_data:
    test_data_dict = pickle.load(pickled_data)

#assert(len(test_data_dict) == test_data_rows)
print 'Successfully imported train data'

with open('cropped_images_greyscale.pickle', 'rb') as pickled_data_train:
    train_data_dict = pickle.load(pickled_data_train)

#assert(len(train_data_dict) == train_data_rows)
print 'Successfully imported image location data'

test_ds = test_data_dict['dataset']
test_labels = test_data_dict['labels']
train_ds = train_data_dict['dataset']
train_labels = train_data_dict['labels']
image_size = train_ds.shape[1]
'''
# open up random image from both datasets to make sure everything is ok
def display_random_image(ds, labels):
    img_num = int(random.random()*test_data_rows)
    ida = ds[int(img_num)]
    print 'Image #' + str(img_num) + ' value is = ' + str(labels[img_num])
    # images are greyscale normalized to between 0 and 1, so we multiply by max(greyscale_value) = 255
    im = Image.fromarray(np.add(ida, 100))

    im.show()

display_random_image(test_ds, test_labels)
display_random_image(train_ds, train_labels)
'''

assert len(train_ds) == train_data_rows
assert len(test_ds) == test_data_rows

# if there is one with more than 5 digits we trim
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
#train_labels.resize(train_data_rows, 5)
assert train_labels.shape[1] == test_labels.shape[1]


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    tf.placeholder(tf.int32, shape=(batch_size, 5))
    return dataset, labels


train_dataset, train_labels = reformat(train_ds, train_labels)
test_dataset, test_labels = reformat(test_ds, test_labels)


valid_dataset = train_dataset[-4000:,:,:]
valid_labels= train_labels[-4000:,:]
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
train_dataset = np.delete(train_dataset, np.s_[-4000:], axis=0)
train_labels = np.delete(train_labels, np.s_[-4000:], axis=0)


print('Training set', train_dataset.shape, train_labels.shape)
print('Valid set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
# Start of Model...



def accuracy(predictions, labels):
    #print predictions[0], labels[0]
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])

#  Used code from 4_convolutions as template for the model

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, 5))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    layer1_filter = tf.Variable(tf.truncated_normal([filter1_size, filter1_size, num_channels, depth1], stddev=0.05))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    layer2_filter = tf.Variable(tf.truncated_normal([filter2_size, filter2_size, depth1, depth2], stddev=0.05))
    layer2_biases = tf.Variable(tf.constant(0, shape=[depth2]))
    layer3_filter = tf.Variable(tf.truncated_normal([filter3_size, filter3_size, depth2, depth3], stddev=0.05))
    layer3_biases = tf.Variable(tf.constant(0, shape=[depth3]))
    layer6_filter = tf.Variable(tf.truncated_normal([filter4_size, filter4_size, depth3, depth3], stddev=0.05))
    layer6_biases = tf.Variable(tf.constant(0, shape=[depth3]))


    layer4_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth3, num_hidden], stddev=0.05))
    layer4_biases = tf.Variable(tf.zeros([num_hidden]))
    layer5_weights1 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.08))
    layer5_biases1 = tf.Variable(tf.constant(0, shape=[num_labels]))
    layer5_weights2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.08))
    layer5_biases2 = tf.Variable(tf.constant(0, shape=[num_labels]))
    layer5_weights3 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.08))
    layer5_biases3 = tf.Variable(tf.constant(0, shape=[num_labels]))
    layer5_weights4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.08))
    layer5_biases4 = tf.Variable(tf.constant(0, shape=[num_labels]))
    layer5_weights5 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.08))
    layer5_biases5 = tf.Variable(tf.constant(0, shape=[num_labels]))

    def model(data, train=None):
        conv = tf.nn.conv2d(data, layer1_filter, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,1,1,1], 'SAME')
        #if train:
        #    tf.nn.dropout(pool,.5)

        conv = tf.nn.conv2d(pool, layer2_filter, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], 'SAME')
        #if train:
        #    tf.nn.dropout(pool,.5)

        conv = tf.nn.conv2d(pool, layer3_filter, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer3_biases)
        pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,1,1,1], 'SAME')
        #if train:
        #    tf.nn.dropout(pool,.5)

        conv = tf.nn.conv2d(pool, layer6_filter, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer6_biases)
        pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], 'SAME')

        #if train:
        #    tf.nn.dropout(pool,.5)
        shape = pool.get_shape().as_list()

        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        #print 'reshape shape + ' + str(reshape.get_shape())
        #print 'layer4weights shape + ' + str(layer4_weights.get_shape())
        hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)


        return (tf.matmul(hidden, layer5_weights1) + layer5_biases1), (tf.matmul(hidden, layer5_weights2) + layer5_biases2), \
            (tf.matmul(hidden, layer5_weights3) + layer5_biases3), (tf.matmul(hidden, layer5_weights4) + layer5_biases4), \
            (tf.matmul(hidden, layer5_weights5) + layer5_biases5)

    # Training computation.
    logits1,logits2,logits3,logits4,logits5 = model(tf_train_dataset, train=True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, tf_train_labels[:,0])) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, tf_train_labels[:,1])) \
                          + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits3, tf_train_labels[:,2])) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits4, tf_train_labels[:,3])) \
                          + tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits5, tf_train_labels[:,4])))


    global_step = tf.Variable(0, trainable=False)
    #starter_learning_rate = 0.03
    #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                      1000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(.05).minimize(loss)#, global_step=global_step)


    # Predictions for the training, validation, and test data.
    train_prediction = tf.pack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),tf.nn.softmax(logits4),tf.nn.softmax(logits5)])
    logits1,logits2,logits3,logits4,logits5 = model(tf_valid_dataset, train=False)
    valid_prediction = tf.pack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),tf.nn.softmax(logits4),tf.nn.softmax(logits5)])
    logits1,logits2,logits3,logits4,logits5 = model(tf_test_dataset, train=False)
    test_prediction = tf.pack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),tf.nn.softmax(logits4),tf.nn.softmax(logits5)])



# Add ops to save and restore all the variables.



num_steps = 5000
start_time = timeit.default_timer()
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    #saver.restore(session, "saved_models/model.ckpt")
    #print("Model restored.")
    print('Initialized')
    for step in range(num_steps):
        tf.initialize_all_variables().run()
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        #print feed_dict
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    save_path = saver.save(session, "saved_models/model.ckpt")
    print("Model saved in file: %s" % save_path)
elapsed = timeit.default_timer() - start_time
print ('TIME: ' + str(elapsed))
