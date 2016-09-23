__author__ = 'jpatdalton'
'''
This file is a simple camera application that can be run on Mac OS X (and perhaps other operating systems).
'''

import time
import cv2
import predictor
import tensorflow as tf


def execute(model_file):
    """This function creates a OpenCV video capture on a computer's camera.  It then loads the predictor's graph and takes images every 4 seconds.
    Predictions for each image are printed to the terminal

    Args:
        model_file: This is the file that contains a trained model on the SVHN dataset.
    """

    time.sleep(2)  # give camera time to get setup - it will throw an error if not
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    with tf.Session(graph=predictor.graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, model_file)
        print("Model restored.")

        for i in range(100):
            # Capture frame-by-frame
            time.sleep(4)
            ret, frame = cap.read()
            predictor.process_array(frame, session)

    cap.release()