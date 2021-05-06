"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

from torchvision.transforms import Compose
from .transforms import Resize, NormalizeImage, PrepareForNet
from .utils import read_image, write_depth


def run(model, size, input_path, output_path):
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w, net_h = size, size
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio = False,
                ensure_multiple_of=32,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.eval()



    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input

        img = read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction, _, _, _ = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        trained_filename = os.path.join(
            output_path, 'trained_' + os.path.splitext(os.path.basename(img_name))[0]
        )
        write_depth(trained_filename, prediction, bits=2)


    print("finished")