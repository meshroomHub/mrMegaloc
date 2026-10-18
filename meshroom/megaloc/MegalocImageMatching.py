__version__ = "1.0"

from pathlib import Path
from meshroom.core import desc

class MegalocImageMatching(desc.CommandLineNode):

    category = "MegaLoc"
    documentation = """
    Build an image pairs list from two folders of MegaLoc descriptors (output of MegalocImageDescripting).
    Each view of the first folder is paired with its most similar views of the second folder.
    Use the same folder twice to match a set of views against itself.
    """
    gpu = desc.Level.NORMAL

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "imageMatching.py").as_posix()

    commandLine = "python " + exePath + " {allParams}"

    inputs = [
        desc.File(
            name="descriptorsFolderA",
            label="Descriptors Folder A",
            description="Folder of MegaLoc descriptors of the query views.",
            value="",
        ),
        desc.File(
            name="descriptorsFolderB",
            label="Descriptors Folder B",
            description="Folder of MegaLoc descriptors of the views to match against.",
            value="",
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
    ]

    outputs = [
        desc.File(
            name="output",
            label="Image Pairs",
            description="Output file containing the list of image pairs.",
            value="{nodeCacheFolder}/imagePairs.txt",
        ),
    ]
