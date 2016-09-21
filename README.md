# deep_learning_project

Make sure that this directory is on your path.. this did it for me
sys.path.append('~/Desktop/udacity_ml/machine-learning/projects/deep_learning')

TO CREATE AND TRAIN A MODEL:
import master_script
master_script.execute()
 
There is a keyword argument you may pass in to the execute() function that is the number of steps to train on.
Default is 5000, but change it like such:
master_script.execute(num_steps=4000)

TO PREDICT IMAGES:
import predictor
predictor.process(file_name)


LIVE CAMERA APPLICATION:
import mac_camera_app
mac_camera_app.execute(model_file)

Example model_file value is "saved_models/model12.ckpt"