__version__ = "2.0"

from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class MegalocImageMatching(desc.CommandLineNode):

    category = "MegaLoc"
    documentation = """
    Build an image pairs list between the views of two sfmData, using MegaLoc descriptors (output of MegalocImageDescripting).
    Each view of sfmData A is paired with its most similar views of sfmData B.
    Use the same sfmData twice to match a set of views against itself.
    Descriptors are looked up in the features folders, which may contain descriptors of other views.
    """
    gpu = desc.Level.INTENSIVE

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "imageMatching.py").as_posix()

    commandLine = "python " + exePath + " {allParams}"

    inputs = [
        desc.File(
            name="inputSfMDataA",
            label="SfMData A",
            description="SfMData of the query views.",
            value="",
        ),
        desc.File(
            name="inputSfMDataB",
            label="SfMData B",
            description="SfMData of the views to match against.",
            value="",
        ),
        desc.ListAttribute(
            elementDesc=desc.File(
                name="featuresFolder",
                label="Features Folder",
                description="Folder of MegaLoc descriptors (output of MegalocImageDescripting).",
                value="",
            ),
            name="featuresFolders",
            label="Features Folders",
            description="Folders of MegaLoc descriptors. They must contain the descriptors of all the views of both sfmData, and may contain others.",
        ),
        desc.IntParam(
            name="nbNeighbors",
            label="Nb Neighbors",
            description="Number of most similar views of B paired with each view of A.",
            value=20,
            range=(1, 200, 1),
        ),
        desc.FloatParam(
            name="minSimilarity",
            label="Min Similarity",
            description="Minimal cosine similarity between two views to keep a pair (applied on the nbNeighbors best views).",
            value=0.0,
            range=(-1.0, 1.0, 0.01),
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Image Pairs",
            description="Output file containing the list of image pairs.",
            value="{nodeCacheFolder}/imagePairs.txt",
        ),
    ]
