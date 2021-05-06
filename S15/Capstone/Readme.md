**Capstone Logs : https://github.com/UrmilaAirsang/Pytorch/blob/main/S15/Capstone/CapstoneLogs.pdf**

Data used: 
1. 5000 interior images to train planer model
2. 3000 images to train Yolo model. This has 4 classes viz Hardhats, Vests, Masks, Boots


Capstone project consists of three models, In this MiDas is encoder freezed and connected to the following decoders. These decoders are trained:
1. Midas Decoder - this decoder estimates the depth of the image.
2. Planer  Decoder -  used to get the segmented planes of the image, planer decoder also give many other outputs, ex: depth estimation, but here we will concentrate only on planer segmentation output
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
  2.3 PlanerCnn loss initially decreased, but after few epochs maintained constant value.


More updates to the loss function is being explored and will be updated soon!

Below are the links:
MiDaS link: https://github.com/intel-isl/MiDaS
Planercnn: https://github.com/NVlabs/planercnn
1. weight files 
    1.  Midas: https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
    2.  PlanerCNN: https://www.dropbox.com/s/yjcg6s57n581sk0/checkpoint.zip?dl=0
2. Assignment15 py file: https://github.com/UrmilaAirsang/Pytorch/blob/main/S15/Capstone/mainfolder/Assignment15.py
3. Capstone Logs : https://github.com/UrmilaAirsang/Pytorch/blob/main/S15/Capstone/CapstoneLogs.pdf 
4. Images, Label are stored locally and in drive 
