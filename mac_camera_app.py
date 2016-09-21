__author__ = 'jpatdalton'


import time
import cv2
import predictor
import tensorflow as tf


def execute(model_file):
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