__version__ = "1.0"

from pathlib import Path
from meshroom.core import desc
from pyalicevision import parallelization as avpar

class MegalocImageDescripting(desc.CommandLineNode):

    category = "MegaLoc"
    documentation = """
    Compute a MegaLoc global image descriptor for each view of the sfmData.
    Descriptors are L2-normalized float32 vectors saved as <viewId>.megaloc.npy.
    """
    size = avpar.DynamicViewsSize('inputSfMData')
    gpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=50)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "imageDescripting.py").as_posix()

    commandLine = "python " + exePath + " {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.IntParam(
            name="imageSize",
            label="Image Size",
            description="Images are resized to imageSize x imageSize before computing the descriptor (rounded to a multiple of 14).",
            value=322,
            range=(112, 1036, 14),
            advanced=True,
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Descriptors Folder",
            description="Output folder for descriptors.",
            value="{nodeCacheFolder}",
        ),
    ]
