from pyalicevision import matchingImageCollection as avmic
from pyalicevision import sfmData as avsfmData
from pyalicevision import sfmDataIO as avsfmDataIO

import argparse
import logging
import os

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUFFIX = ".megaloc.npy"


def get_view_ids(inputSfMData):
    """
    Return the sorted view ids of a sfmData.
    """
    data = avsfmData.SfMData()
    if not avsfmDataIO.load(data, inputSfMData, avsfmDataIO.VIEWS):
        raise RuntimeError(f"Unable to load sfmData {inputSfMData}")
    return sorted(data.getViews().keys())

def load_descriptors(viewIds, folders):
    """
    Load the <viewId>.megaloc.npy descriptor of each view, from the first folder containing it.
    Folders may contain descriptors of other views, which are ignored.

    Returns:
    (N, D) descriptors tensor, in the order of viewIds
    """
    descriptors = []
    missing = []
    for viewId in viewIds:
        path = next((p for p in (os.path.join(f, f"{viewId}{SUFFIX}") for f in folders) if os.path.isfile(p)), None)
        if path is None:
            missing.append(viewId)
        else:
            descriptors.append(np.load(path))

    if missing:
        raise RuntimeError(f"No descriptor found for {len(missing)} views: {missing}")
    if not descriptors:
        return torch.empty(0, 0, device=device)

    return torch.from_numpy(np.stack(descriptors)).float().to(device)

def compute_pairs(viewIdsA, descriptorsA, viewIdsB, descriptorsB, nbNeighbors, minSimilarity):
    """
    Pair each view of A with its nbNeighbors most similar views of B,
    keeping only pairs with a similarity of at least minSimilarity.
    A view is never paired with itself.

    Returns:
    a PairSet of (smallest id, largest id) pairs
    """
    pairs = avmic.PairSet()
    if not viewIdsA or not viewIdsB:
        return pairs

    similarities = descriptorsA @ descriptorsB.T

    # Exclude self pairs (views present in both folders)
    indexB = {viewId: j for j, viewId in enumerate(viewIdsB)}
    for i, viewId in enumerate(viewIdsA):
        if viewId in indexB:
            similarities[i, indexB[viewId]] = -float("inf")

    k = min(nbNeighbors, len(viewIdsB))
    values, neighbors = similarities.topk(k, dim=1)
    values = values.cpu().numpy()
    neighbors = neighbors.cpu().numpy()

    for i, row in enumerate(neighbors):
        for value, j in zip(values[i], row):
            if value < minSimilarity:
                continue
            a, b = viewIdsA[i], viewIdsB[int(j)]
            pairs.add((min(a, b), max(a, b)))

    return pairs

def image_matching(inputSfMDataA, inputSfMDataB, featuresFolders, nbNeighbors, minSimilarity, output):
    viewIdsA = get_view_ids(inputSfMDataA)
    viewIdsB = get_view_ids(inputSfMDataB)
    descriptorsA = load_descriptors(viewIdsA, featuresFolders)
    descriptorsB = load_descriptors(viewIdsB, featuresFolders)
    logging.info(f"Loaded {len(viewIdsA)} descriptors for {inputSfMDataA}")
    logging.info(f"Loaded {len(viewIdsB)} descriptors for {inputSfMDataB}")

    pairs = compute_pairs(viewIdsA, descriptorsA, viewIdsB, descriptorsB, nbNeighbors, minSimilarity)
    logging.info(f"Writing {len(pairs)} pairs to {output}")

    if not avmic.savePairsToFile(output, pairs):
        raise RuntimeError(f"Unable to write pairs file {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute image pairs between the views of two sfmData using MegaLoc descriptors')
    parser.add_argument('--inputSfMDataA', type=str, required=True, help='SfMData of the query views')
    parser.add_argument('--inputSfMDataB', type=str, required=True, help='SfMData of the views to match against')
    parser.add_argument('--featuresFolders', type=str, nargs='+', required=True, help='Folders of MegaLoc descriptors')
    parser.add_argument('--nbNeighbors', type=int, default=20, help='Number of neighbors in B per view of A')
    parser.add_argument('--minSimilarity', type=float, default=0.0, help='Minimal cosine similarity to keep a pair')
    parser.add_argument('--output', type=str, required=True, help='Output image pairs file')
    parser.add_argument('--verboseLevel', type=str, default='info', help='Verbosity level (fatal, error, warning, info, debug, trace)')
    args, _ = parser.parse_known_args()

    # Python logging has no trace level
    logging.basicConfig(level="DEBUG" if args.verboseLevel == "trace" else args.verboseLevel.upper())

    image_matching(inputSfMDataA=args.inputSfMDataA,
                   inputSfMDataB=args.inputSfMDataB,
                   featuresFolders=args.featuresFolders,
                   nbNeighbors=args.nbNeighbors,
                   minSimilarity=args.minSimilarity,
                   output=args.output)
