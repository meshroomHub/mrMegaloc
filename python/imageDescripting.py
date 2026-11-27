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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


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
    (square resize, ImageNet normalization).
    imageSize is rounded to a multiple of 14 (MegaLoc patch size), as MegaLoc would resize again otherwise.
    """
    image = avimage.Image_RGBColor()
    optRead = avimage.ImageReadOptions(avimage.EImageColorSpace_SRGB)
    avimage.readImage(path, image, optRead)

    array = np.asarray(image.getNumpyArray(), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)

    imageSize = max(14, round(imageSize / 14) * 14)
    # Resizing first is equivalent (the normalization is per-channel affine) and cheaper
    tensor = F.interpolate(tensor, size=(imageSize, imageSize), mode="bilinear", antialias=True, align_corners=False)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD

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
    parser = argparse.ArgumentParser(description='Compute MegaLoc global descriptors for each view')
    parser.add_argument('--inputSfMData', type=str, required=True, help='Input sfmData')
    parser.add_argument('--imageSize', type=int, default=322, help='Square size images are resized to')
    parser.add_argument('--output', type=str, required=True, help='Output descriptors folder')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.add_argument('--verboseLevel', type=str, default='info', help='Verbosity level (fatal, error, warning, info, debug, trace)')
    args, _ = parser.parse_known_args()

    # Python logging has no trace level
    logging.basicConfig(level="DEBUG" if args.verboseLevel == "trace" else args.verboseLevel.upper())

    compute_descriptors(inputSfMData=args.inputSfMData,
                        imageSize=args.imageSize,
                        outputFolder=args.output,
                        rangeIteration=args.rangeIteration,
                        rangeBlocksCount=args.rangeBlocksCount)
