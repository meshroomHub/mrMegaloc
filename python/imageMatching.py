from pyalicevision import matchingImageCollection as avmic

import argparse
import glob
import logging
import os

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUFFIX = ".megaloc.npy"


def load_descriptors(folder):
    """
    Load all the <viewId>.megaloc.npy files of a folder.

    Returns:
    (list of view ids, (N, D) descriptors tensor)
    """
    paths = sorted(glob.glob(os.path.join(folder, "*" + SUFFIX)))
    viewIds = [int(os.path.basename(path)[:-len(SUFFIX)]) for path in paths]
    if not paths:
        return viewIds, torch.empty(0, 0, device=device)

    descriptors = np.stack([np.load(path) for path in paths])
    return viewIds, torch.from_numpy(descriptors).float().to(device)

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

def image_matching(descriptorsFolderA, descriptorsFolderB, nbNeighbors, minSimilarity, output):
    viewIdsA, descriptorsA = load_descriptors(descriptorsFolderA)
    viewIdsB, descriptorsB = load_descriptors(descriptorsFolderB)
    logging.info(f"Loaded {len(viewIdsA)} descriptors from {descriptorsFolderA}")
    logging.info(f"Loaded {len(viewIdsB)} descriptors from {descriptorsFolderB}")

    pairs = compute_pairs(viewIdsA, descriptorsA, viewIdsB, descriptorsB, nbNeighbors, minSimilarity)
    logging.info(f"Writing {len(pairs)} pairs to {output}")

    if not avmic.savePairsToFile(output, pairs):
        raise RuntimeError(f"Unable to write pairs file {output}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Compute image pairs from two folders of MegaLoc descriptors')
    parser.add_argument('--descriptorsFolderA', type=str, required=True, help='Descriptors folder of the query views')
    parser.add_argument('--descriptorsFolderB', type=str, required=True, help='Descriptors folder of the views to match against')
    parser.add_argument('--nbNeighbors', type=int, default=20, help='Number of neighbors in B per view of A')
    parser.add_argument('--minSimilarity', type=float, default=0.0, help='Minimal cosine similarity to keep a pair')
    parser.add_argument('--output', type=str, required=True, help='Output image pairs file')
    args, _ = parser.parse_known_args()

    image_matching(descriptorsFolderA=args.descriptorsFolderA,
                   descriptorsFolderB=args.descriptorsFolderB,
                   nbNeighbors=args.nbNeighbors,
                   minSimilarity=args.minSimilarity,
                   output=args.output)
