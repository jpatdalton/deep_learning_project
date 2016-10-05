# deep_learning_project

This project is centered around using a Convolutional Neural Network in TensorFlow Python to recognize house numbers from the SVHN dataset: http://ufldl.stanford.edu/housenumbers.  There are several funtionalities.  You may train the model in the code with your own data, you may predict images with an included trained model, and you can run a live camera app on your computer (only tested on OS X).

## Dependencies

### Libraries

This project was built with Python 2.7 and relies on the following libraries:

- numpy
- tensorflow
- PIL
- cv2
- matplotlib
- h5py

You should ensure they are available in your environment before running any of the code.  <br>
FYI: I had a few issues with PIL when testing, but uninstalling PIL and pillow then reinstalling just pillow fixed them.

### Data

You must first download the SVHN dataset to begin.  Download the training and test set:

training set: http://ufldl.stanford.edu/housenumbers/train.tar.gz <br>
test set: http://ufldl.stanford.edu/housenumbers/test.tar.gz

Save them in an easily accessible place.  Relative to my working directory, I saved them in 'data/train' and 'data/test'.  I recommend you do the same.

Once you have downloaded the source code from this repository, and the datasets as described above, you may perform any of the following functionality.

## Create and Train a Model

```
import master_script
master_script.execute()
```

There is a keyword argument you may pass in to the execute() function that is the number of steps to train on.
Default is 5000, but change it like such:

master_script.execute(num_steps=2000)

## Predict Single Images

*Note:* The image should be cropped around the number.  There is no localizer, so prediction will not work well on numbers that take up a small part of the image<br>

```
import predictor
predictor.process(file_name)
```

## Live Camera Application

*(Only tested on Mac OS X 10.11)*<br>
*Note:* This application also requires numbers to take up a majority of the image.  There should be little extra space in the image outside of the number.  It will not work on numbers far away from your camera.  
I'd suggest opening up photo booth to see what you are showing the camera.  

```
import mac_camera_app
model_file = 'saved_models/model12.ckpt'
mac_camera_app.execute(model_file)
```

Example model_file value is "saved_models/model12.ckpt"  That is the trained model from this repo.
