# mrMegaloc

Meshroom plugin using [MegaLoc](https://github.com/gmberton/MegaLoc) global image descriptors.

## Node

**MegalocImageDescripting** (category `MegaLoc`)
- `inputSfMData`: input SfMData.
- `imageSize`: square size images are resized to before computing descriptors (default 322, as used by MegaLoc evaluation).
- `output`: folder containing one `<viewId>.megaloc.npy` per view (L2-normalized float32 vector of size 8448).
  Cosine similarity between two views is the dot product of their descriptors.

**MegalocImageMatching** (category `MegaLoc`)
- `descriptorsFolderA`, `descriptorsFolderB`: two MegalocImageDescripting output folders
  (use the same folder twice to match a set of views against itself).
- `nbNeighbors`: each view of A is paired with its N most similar views of B (a view is never paired with itself).
- `minSimilarity`: pairs with a lower cosine similarity are dropped. The scale depends on the dataset
  (unrelated views are usually around 0-0.05, hard true pairs can be under 0.1): check the log before raising it.
- `output`: AliceVision image pairs file, to plug into a matching node `imagePairsList`.

## Dependencies

Runtime packages are listed in `MRMEGALOC_SUBREQUIRES` (see `package.py`): aliceVision, pytorch, torchvision and
`megaloc_nnModels`, which provides the model code (`MEGALOC_REPO`) and weights (`MEGALOC_WEIGHTS`) for offline use.
