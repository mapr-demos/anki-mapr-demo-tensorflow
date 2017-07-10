# anki-mapr-demo-tensorflow
Traffic sign recognition with TensorFlow

A simple example of using Machine Learning to recognise images.
This optionally integrates with the anki-mapr-demo - when the sign is recognised (e.g. "STOP") the appropriate command is send the Anki cars.

The master for anki-mapr-demo is here: https://github.com/tgrall/anki-mapr-demo

The TensorFlow programs are based on this post: https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6

## High Level Desription

learn.py uses TensorFlow to create a model that recognises different images. If the loss of the model is below a given threshold then the model is saved.

predict_camera.py uses a saved model to give a prediction of the image. The program can optionally control the capture of the image using a webcam and, again optionally, send commands to the anki-mapr-demo controller so that when an image is recognised (e.g. "GO") then the appropriate command is sent to the Anki cars.

A model trained on a set of six images with a low loss value is included along with the 120 training images used.

The parameters used by the programs can be displayed with "python ...py --help". The main ones are:
- train_dir:      Where the training images and labels are located
- trains:         The number of training iterations to peform
- learning_rate:  Learning rate for the optimizer
- model_path:     Where to save/load models from
- models:         Number of models to produce
- min_loss:       Threshold for a model to be considered as "good"
- test_dir:       Location of images for prediction (this contains labels and the image is assumed to be in directory 0)
- use_camera:     Whether to get the image using fswebcam
- poll_frq:       How often to take or check for an image
- use_anki:       Whether or not to send commands to the Anki Controller
- anki_address:   URL of the Anki Controller

Two bash shell scripts are provided with default parameters - runPredict.sh and runPredictUseCamera.sh

A PowerPoint presentation with timed slide changes is also provided (see below).

## Software Used

- TensorFlow
- Python
- fswebcam (a different package could easily be used)

## Assumed Directroy Structure

Beneath the directories for both the training images and the images fed into the prediction, the programs assume that the label directories (0,1,2,3,4,5) exist. For training, these correspond to the 6 different images (STOP, GO, CONNECT, SCAN, FAST, NONE). For prediction, the directories exist only to populate the labels correctly (there is presumably a more elegant way to do this) and directory "0" is the location searched for an image.

## Usage
Assumes a working TensorFlow environment.

### Learning:
Create a directory (DIR) and beneath it six subdirectories (0,1,2,3,4,5) and populate the relevant image into the appropriate subdirectory (so for example put images for STOP signs into directory 0). Then use learn.py to create a model

e.g. python learn.py --train_dir=DIR --trains=2000 --learning_rate=0.01 --models=10 --min_loss=0.1

### Predicting (Manually Controlling Images)
Alter runPredict.sh as necessary and run the script (or modifiy for your environment). Place images in the "0" subdirectory of the testing directory. The program will detect the image, predict what the image is, and then delete the image.

### Predicting (Using the Camera Directly)
As above, but include the "use_camera" parameter as in ruPredictUseCamera.sh

### Using the PowerPoint
The supplied PowerPoint presentation includes slides with the six images in the centre, and also links to other images around the side. Using this, if you focus the webcam on the PowerPoint such that the image is correctly identified, you can then control the image the webcam is seeing by using the links around the side. The PowerPoint returns to "NONE" - i.e. no command - after a short delay so that when using with the Anki controller, commands are not being constantly sent to the cars (generally this is only an issue for SCAN and CONNECT).
