
Data used: 
1. 5000 interior images to train planner model
2. 3000 images to train Yolo model. This has 4 classes viz Hardhats, Vests, Masks, Boots


Capstone project consists of three models, In this MiDas is encoder freezed and connected to the following decoders. These decoders are trained:
1. Midas Decoder - this decoder estimates the depth of the image.
2. Planner  Decoder -  used to get the segmented planes of the image, planner decoder also give many other outputs, ex: depth estimation, but here we will concentrate only on planner segmentation output
3. YoloV3 Decoder - this is used for object detection. this model was already trained on 4 classes. detailed explanation is given below.

File descriptions:
train.txt: This file contains all the training images.
test.txt: This file contains all the testing images.
classes.txt: This file contains classes that are used for training viz Hardhats, Vests, Masks, Boots

Training our model:
1. The logs as pdf are attached which shows that the model has been setup and able to train
2. Observation about decreasing loss
  2.1 Yolo total loss can be seen decreasing.
  2.2 Midas loss can be found more or less constant because of model freeze
  2.3 PlannerCnn loss initially decreased, but after few epochs maintained constant value.


More updates to the loss function is being explored and will be updated soon!
