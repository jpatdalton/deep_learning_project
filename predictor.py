__author__ = 'jpatdalton'

'''This file contains the model and code to predict the digits of a house number.

It has methods to produce a prediction on image arrays and image files.

'''

import tensorflow as tf
from PIL import Image
import numpy as np

ipath='data/sample/'        # path to image
image_size=40
num_channels=1
batch_size=1
batch_size = 1
num_channels = 1    # images are all greyscale
depth1 = 16
depth2 = 25
depth3 = 36
filter1_size = 3
filter2_size = 3
filter3_size = 3
filter4_size = 3
filter5_size = 3
num_hidden = 256
num_labels = 11

# Start of tensorflow graph
graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_sample_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

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
        """Builds the 8 layer deep convolutional network.

        The first 6 layers are convolutional, and the last two are fully connected.

        The first 7 layers have one classifier, which turns into 5 classifiers before the last layer.

        Args:
            data: The data to run through the model
            train: Value that decides whether there is dropout (only used for training).  Defaults to no dropout

        Returns:
            5 logits calculated from data running through network
        """

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

    logits1,logits2,logits3,logits4,logits5 = model(tf_sample_dataset, train=False)
    train_prediction = tf.pack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),tf.nn.softmax(logits4),tf.nn.softmax(logits5)])


def process(file_name):
    """Opens an image file, resizes to 40x40 pixels, converts to greyscale, subtracts mean value of image's pixels and prints the predicted number.

    Args:
        name: The name of the file (relative to your current path).

    """
    img=Image.open(str(file_name))
    cim_resized = img.resize((40,40), resample=Image.LANCZOS)
    n = cim_resized.convert('L')
    cropped = np.array(n).astype(np.float64)
    im=Image.fromarray(cropped)
    im.show()
    normalized_cropped_image = cropped - np.mean(cropped)
    normalized_cropped_image = normalized_cropped_image.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    predicted_arr = predict(normalized_cropped_image)
    label = ''.join(['' if int(x[0]) == 10 else str(x[0]) for x in list(predicted_arr)])
    print 'LABEL: ' + label


def process_array(arr, session):
    """Takes in an image NumPy array and predicts the number in the image.

    This is the function used for predicting images captured with a computer camera.
    It converts the prediction array into a string printed to the terminal.

    Args:
        name: The image NumPy array.
        session: The active Tensorflow session

    """
    img = Image.fromarray(arr)
    cim_resized = img.resize((40,40), resample=Image.LANCZOS)
    n = cim_resized.convert('L')
    cropped = np.array(n).astype(np.float64)
    normalized_cropped_image = cropped - np.mean(cropped)
    normalized_cropped_image = normalized_cropped_image.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    predicted_arr = predict_live(normalized_cropped_image, session)
    label = ''.join(['' if int(x[0]) == 10 else str(x[0]) for x in list(predicted_arr)])
    print 'NUMBER IS : ' + label


def predict(image):
    """Loads a trained model and gets predicted value for image.

    Args:
        image: A numpy array of a 40x40 greyscale image with mean subtraction.

    Returns:
        An array of predicted values (10 means no value).
    """
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, "saved_models/model12.ckpt")
        print("Model restored.")
        feed_dict = {tf_sample_dataset : image}
        predictions = session.run(train_prediction, feed_dict=feed_dict)
        # Prints an array of softmax probabilities for each digit in the number
        print str(predictions)
        return np.argmax(predictions, 2)


def predict_live(image, session):
    """Takes an image array and produces a prediction from a loaded model.

    This is used in the Live Camera App.

    Args:
        image: A numpy array of a 40x40 greyscale image with mean subtraction.
        session: An active Tensorflow session.

    Returns:
        An array of predicted values (10 means no value).
    """

    feed_dict = {tf_sample_dataset : image}
    predictions = session.run(train_prediction, feed_dict=feed_dict)

    return np.argmax(predictions, 2)




