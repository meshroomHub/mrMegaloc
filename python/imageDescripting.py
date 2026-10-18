from pyalicevision import image as avimage
from pyalicevision import sfmData as avsfmData
from pyalicevision import sfmDataIO as avsfmDataIO
from pyalicevision import system as avsys

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_views_from_sfmdata(inputSfMData):
    """
    Parse a sfmData and return a dict (view uid -> image path), sorted by frame id.
    """
    data = avsfmData.SfMData()
    if not avsfmDataIO.load(data, inputSfMData, avsfmDataIO.VIEWS):
        raise RuntimeError("Error with sfm data")

    views = sorted(data.getViews().items(), key=lambda v: v[1].getFrameId())
    return {key: view.getImage().getImagePath() for (key, view) in views}

def load_model():
    """
    Create the MegaLoc model from the code and weights provided by megaloc_nnModels.
    """
    sys.path.insert(0, os.environ["MEGALOC_REPO"])
    from megaloc_model import MegaLoc

    model = MegaLoc()
    model.load_state_dict(torch.load(os.environ["MEGALOC_WEIGHTS"], map_location="cpu", weights_only=True))
    return model.eval().to(device)

def load_image_tensor(path, imageSize):
    """
    Load an image using pyalicevision and apply the MegaLoc preprocessing
    (ImageNet normalization, square resize).
    """
    image = avimage.Image_RGBColor()
    optRead = avimage.ImageReadOptions(avimage.EImageColorSpace_SRGB)
    avimage.readImage(path, image, optRead)

    array = np.asarray(image.getNumpyArray(), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return F.interpolate(tensor, size=(imageSize, imageSize), mode="bilinear", antialias=True, align_corners=False)

def compute_descriptors(inputSfMData, imageSize, outputFolder, rangeIteration, rangeBlocksCount):
    """
    Compute the MegaLoc descriptor of each view in the assigned range
    and save it as <outputFolder>/<viewId>.megaloc.npy.
    """
    views = get_views_from_sfmdata(inputSfMData)
    viewIds = list(views.keys())

    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(viewIds))
    if not valid:
        logging.error("Range is out of bounds.")
        return

    viewIds = viewIds[rangeStart:rangeEnd]
    logging.info(f"Processing views {rangeStart} to {rangeEnd}")

    model = load_model()

    for viewId in viewIds:
        logging.info(f"Computing descriptor for view {viewId} ({views[viewId]})")
        with torch.inference_mode():
            descriptor = model(load_image_tensor(views[viewId], imageSize))[0]

        np.save(os.path.join(outputFolder, f"{viewId}.megaloc.npy"), descriptor.float().cpu().numpy())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Compute MegaLoc global descriptors for each view')
    parser.add_argument('--inputSfMData', type=str, required=True, help='Input sfmData')
    parser.add_argument('--imageSize', type=int, default=322, help='Square size images are resized to')
    parser.add_argument('--output', type=str, required=True, help='Output descriptors folder')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    args, _ = parser.parse_known_args()

    compute_descriptors(inputSfMData=args.inputSfMData,
                        imageSize=args.imageSize,
                        outputFolder=args.output,
                        rangeIteration=args.rangeIteration,
                        rangeBlocksCount=args.rangeBlocksCount)
