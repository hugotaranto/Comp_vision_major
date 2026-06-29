# Chessboard Detection, Piece Classification and 2D Board Reconstruction

A modular computer-vision pipeline that takes a single RGB photo of a physical
chessboard and reconstructs the position as a clean digital 2D board. It was
built as a university computer-vision group project.

The pipeline is deliberately classical and mask-first rather than an end-to-end
neural detector: the board is localised with geometric image processing, pieces
are isolated using monocular depth plus prompted segmentation, and piece type is
predicted from the **shape of the segmentation mask only** (no colour, texture,
or raw pixels). Decoupling shape from appearance makes classification largely
invariant to lighting, glare, and piece colour, and keeps the whole system
CPU-friendly and interpretable.

```
photo ─▶ board localisation ─▶ monocular depth ─▶ plane fit + height residuals
      ─▶ SAM segmentation ─▶ homography warp + square assignment + colour
      ─▶ geometric-feature piece classification ─▶ rendered 8×8 board
```

## Results

Measured on an in-house test split (see [Dataset](#dataset) and
[Methodology](#methodology)). These are the figures reported in the project
report.

| Stage | Metric | Result |
|---|---|---|
| Board detection | Mean corner localisation error | 5.37 px |
| Piece detection | False positives | 0 |
| Piece detection | Missed pieces | ~1.2% |
| Piece classification | Overall accuracy (SVM-RBF, C=10) | 91.87% |
| Colour assignment | Accuracy on tested set | 100% |

Per-class piece classification (160-sample held-out test set):

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Bishop | 0.87 | 0.93 | 0.90 | 28 |
| King | 0.81 | 0.96 | 0.88 | 23 |
| Knight | 0.94 | 1.00 | 0.97 | 31 |
| Pawn | 1.00 | 1.00 | 1.00 | 25 |
| Queen | 1.00 | 0.76 | 0.86 | 21 |
| Rook | 0.93 | 0.84 | 0.89 | 32 |

The dominant error is **king ↔ queen** confusion (similar crown silhouettes);
the next is rook ↔ bishop. Pawn and knight are near-perfectly separated by shape.
Running `train_semantic_from_labeled.py` regenerates the confusion matrix as
`Piece_Labeling_and_Classification/confusion_matrix.png` (not committed).

## Tech stack

- **Language:** Python (developed on 3.13)
- **Core CV / numerics:** OpenCV, NumPy, Open3D
- **Deep models (external checkpoints):**
  [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
  `vit_l`, and Apple [Depth Pro](https://github.com/apple/ml-depth-pro)
  monocular depth
- **Classical ML:** scikit-learn (RandomForest, SVM, KNN)
- **Data / plotting:** pandas, matplotlib, seaborn, Pillow

There is no `requirements.txt`; dependencies are listed under
[Installation](#installation).

## Project structure

```
Comp_vision_major/
├── main.py                              # End-to-end pipeline entry point
├── chess_board_reconstructor.py         # Renders the final 2D board from an 8x8 label array
│
├── detection/                           # Board + piece detection / segmentation
│   ├── board_detect.py                  # Board corner localisation (HSV mask + Canny + Hough + line clustering)
│   ├── depth.py                         # Apple Depth Pro monocular-depth wrapper
│   ├── piece_detect.py                  # RANSAC plane fit, height residuals, SAM segmentation, mask filtering
│   ├── pose_detect.py                   # Homography warp to top-down, square assignment, colour detection
│   ├── util.py                          # Image IO (load_rgb), resize/crop/pad, mask saving
│   ├── plots.py                         # Matplotlib/OpenCV visualisation helpers
│   └── detection_output/labels.csv      # Piece labels for generated masks (799 rows)
│
├── Piece_Labeling_and_Classification/   # Piece type classification
│   ├── label_semantics.py               # Interactive mask-labelling tool (visual confirmation, edit/undo/resume)
│   ├── label_detection_output.py        # Labelling/merge tool for detection_output masks
│   ├── train_semantic_from_labeled.py   # Geometric feature extraction + classifier training (80/20 hold-out)
│   ├── train_full_dataset.py            # Train on 100% of data (5-fold CV scored) for a live demo model
│   ├── test_detection_model.py          # Evaluate a saved model on a labelled mask set
│   ├── inference.py                     # predict_pieces_from_semantic(): used by main.py
│   ├── colour_detect.py                 # Luminance-threshold white/black classifier
│   ├── checkpoints/                     # 4 saved model snapshots (.pkl)
│   ├── semantic_chess_classifier.pkl    # Default trained classifier
│   ├── labels_final.csv (final_chess_dataset/)  # 799 labelled pieces
│   └── PROGRESS_SUMMARY.md              # Historical dev notes (partly outdated)
│
├── stream/stream_iphone.py              # DroidCam phone-camera capture utility (demo input)
├── 2d_chess_pieces_images/              # 12 PNG sprites used to render the 2D board
├── smaller_training_set/labels.csv      # Subset label list
└── .gitignore
```

What is **not** in the repository (gitignored): the raw board images, the
generated segmentation masks and display overlays (all `*.png`), the SAM and
Depth Pro model checkpoints (`*.pth`/`*.pt`), and the Depth Pro source tree
(`ml-depth-pro/`). Only the label CSVs, the trained `.pkl` classifiers, the
sprite PNGs, and the source code are committed. You must supply the dataset
images and the two external checkpoints yourself to run the full pipeline.

## Installation

1. **Clone and create an environment** (Python 3.10+; developed on 3.13):

   ```bash
   git clone https://github.com/hugotaranto/Comp_vision_major.git
   cd Comp_vision_major
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install Python dependencies:**

   ```bash
   pip install numpy opencv-python open3d torch scikit-learn pandas matplotlib seaborn pillow
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

3. **Install Apple Depth Pro** and download its checkpoint (follow the
   [ml-depth-pro](https://github.com/apple/ml-depth-pro) instructions). The code
   expects the checkpoint at:

   ```
   ./ml-depth-pro/checkpoints/depth_pro.pt
   ```

4. **Download the SAM `vit_l` checkpoint** (`sam_vit_l_0b3195.pth`) and place it at:

   ```
   ./sam_checkpoints/sam_vit_l_0b3195.pth
   ```

These checkpoint paths are hard-coded near the top of `main.py` and
`detection/piece_detect.py`; edit them there if you store the files elsewhere. A
CUDA GPU is used automatically if available, otherwise the models run on CPU.

## Usage

Run the full pipeline and the detection module from the **repository root**; run
the labelling/training scripts from **inside
`Piece_Labeling_and_Classification/`** (they use relative paths).

### End-to-end reconstruction

Put input photos in the directory named by `IMAGE_DIRECTORY` in `main.py`
(default `./small_test`), then:

```bash
python main.py
```

For each image this localises the board, estimates depth, detects and segments
pieces, classifies each piece, detects its colour, and shows the original photo
next to the rendered 2D board.

> **Note:** `main.py` sets `CLASSIFIER_MODEL_PATH` to
> `./Piece_Labeling_and_Classification/classifier_with_removed.pkl`, which is not
> present in the repo. Point it at a committed model first, e.g.
> `./Piece_Labeling_and_Classification/semantic_chess_classifier.pkl` or one of
> the snapshots in `checkpoints/` (`4_trained_on_new_data.pkl` is the latest).

### Detection / mask generation only

```bash
python -m detection.piece_detect
```

Reads images from `IMAGE_DIRECTORY` in `piece_detect.py` (default `./every_place`)
and writes integer masks, RGB copies, and coloured overlays to
`detection/detection_output/{semantics,images,display}/`.

### Labelling masks

```bash
cd Piece_Labeling_and_Classification
python label_semantics.py
```

Interactive tool: it shows each piece mask beside the original crop and you tag
it with `p` (pawn), `r` (rook), `kn` (knight), `bi` (bishop), `q` (queen),
`ki` (king). Supports `s` (skip), `x` (quit/save), `undo`, and an `edit` mode for
fixing earlier labels. Labels are written to a `labels.csv`. Progress is resumed
automatically (already-labelled pieces are skipped).

### Training the piece classifier

```bash
cd Piece_Labeling_and_Classification
python train_semantic_from_labeled.py     # 80/20 hold-out, picks the best of RF/SVM/KNN
# or
python train_full_dataset.py              # trains on 100% of data, scored with 5-fold CV (demo model)
```

`train_semantic_from_labeled.py` reads `final_chess_dataset/labels_final.csv`,
extracts geometric features, trains Random Forest, SVM (RBF and linear) and KNN,
selects the most accurate on the held-out split, prints a classification report,
saves `confusion_matrix.png`, and serialises the winner to
`semantic_chess_classifier.pkl`.

### Capturing input from a phone

```bash
python stream/stream_iphone.py
```

Streams frames from a phone camera over [DroidCam](https://www.dev47apps.com/)
(set `IPHONE_STREAM_URL` to the IP/port shown in the app); press `s` to save a
snapshot, `q` to quit. Used to build the dataset and for live demo input.

## Dataset

The dataset is self-constructed. About 75 overhead photos of a single
**green-and-white board with black/white pieces** were captured under varied
orientations, heights, tilts, and lighting. The board and piece colours were
chosen to maximise piece-vs-square contrast and avoid confusing dark pieces with
dark squares. Earlier attempts that trained on online photos of multi-coloured
piece sets were unreliable and were dropped (see `PROGRESS_SUMMARY.md` for that
history).

From those photos the detection pipeline produced per-piece segmentation masks,
which were then hand-labelled with `label_semantics.py` into **799 piece
instances across the six classes**:

| Class | Rook | Knight | Bishop | Pawn | King | Queen | Total |
|---|---|---|---|---|---|---|---|
| Count | 158 | 154 | 140 | 126 | 113 | 108 | 799 |

The committed `labels_final.csv` records, per piece: source mask filename,
integer mask value, class, contour area, and bounding box. The raw images and
masks themselves are gitignored and not distributed with the repo. The
classifier therefore trains on **masks generated by this project's own
detection stage**, not on an external dataset.

## Methodology

The system runs as four sequential stages, each a separate module.

### 1. Board localisation — `detection/board_detect.py`

Threshold the board's green squares in HSV, clean the mask with morphology, run
Canny edge detection, and detect line segments with a probabilistic Hough
transform. Line segments are clustered by angle (agglomerative clustering on the
unit circle) into vertical and horizontal groups; the extremal line on each side
is fitted and the four lines are intersected to give the board's outer corners.
Those corners define the homography used to rectify the board to a canonical
8×8 grid.

### 2. Piece detection + segmentation — `detection/depth.py`, `piece_detect.py`

1. **Depth:** Apple Depth Pro estimates a metric depth map from the single RGB
   image (using the image's focal length).
2. **Plane fit:** the board surface is fitted as a plane with Open3D RANSAC on
   the depth point cloud.
3. **Height residuals:** subtracting the plane leaves pieces standing above the
   board. A robust threshold (`median − k·MAD`) plus residual-based refinement,
   morphology, and connected components isolates each piece and its
   distance-transform centroid.
4. **Segmentation:** those centroids are fed as point prompts to SAM (`vit_l`).
   Candidate masks are filtered by SAM score, area, single-connected-component
   checks, and rejection of masks that swallow other pieces' seed points. The
   output is an integer "semantic" mask labelling each piece `1..N`.

### 3. Piece + colour classification — `Piece_Labeling_and_Classification/`, `colour_detect.py`

Classification uses **mask shape only**. `extract_geometric_features` computes
~50 contour/shape descriptors per piece: area, perimeter, bounding-box aspect
ratio and extent, circularity, compactness, solidity, convexity-defect
statistics, the 7 Hu moments and normalised central moments, fitted-ellipse
eccentricity/orientation/axis ratio, polygon-approximation vertex counts at
three tolerances, distance-transform statistics, a skeleton ratio, the
magnitudes of the first five Fourier descriptors, and radial
centroid-to-boundary statistics. Features are standardised (zero mean, unit
variance) before training.

Four scikit-learn classifiers are trained (RandomForest, SVM-RBF, SVM-linear,
KNN) and the best on the held-out split is kept. SVM-RBF (C=10) won at 91.87%.
Colour is a separate, learning-free step (`colour_detect.py`): the mean
luminance inside each mask is thresholded into white/black.

The report describes an 80/20 split with scene-level separation (no board or
viewpoint shared between train and test). The committed
`train_semantic_from_labeled.py` performs a stratified per-piece 80/20 hold-out
on the labelled data (and routes any `detection_output`-sourced rows entirely
into training); reproduce the exact reported split with care if that distinction
matters to you.

### 4. Board reconstruction — `pose_detect.py`, `chess_board_reconstructor.py`

The piece mask is perspective-warped to a top-down 800×800 view using the board
homography. Each piece is assigned to the square containing the majority of its
bottom-half pixels (its base), giving an 8×8 array of piece IDs. Combined with
per-piece type and colour, `chess_board_reconstructor.py` renders a
Chess.com-style board, placing the sprites from `2d_chess_pieces_images/` on the
correct squares. `main.py` ties all four stages together.

## Known limitations and notes

**Trained on one board and one piece set.** The system was trained and evaluated
on a single physical chess set and board, so it does not yet generalise across
piece designs, board colours, or lighting. This is not one weakness but three,
each affecting a different stage of the pipeline:

- **Board colour → localisation.** `detection/board_detect.py` isolates the board
  with an HSV *green* mask, so a wooden, marble, or vinyl board fails before piece
  detection even runs. Detecting the 8×8 alternating-square grid by its geometry
  rather than its colour would remove this dependency.
- **Piece colour + lighting → colour detection.** The white/black split
  (`Piece_Labeling_and_Classification/colour_detect.py`) is a fixed luminance
  threshold tuned to our pieces under
  our lighting. A per-image split (e.g. Otsu or k-means over the detected pieces'
  brightness) would adapt it to whatever two colours a given set uses.
- **Piece design → type classification.** The ~50 geometric/contour features
  describe *our* set's silhouettes, so a different design can make, say, a bishop
  read as a pawn. Broader training data (multiple sets) plus a learned feature
  extractor would raise the ceiling here.

The reported 91.87% accuracy is an 80/20 split *within this one set* — i.e.
in-domain performance, not a measure of generalisation. A leave-one-set-out
evaluation across several physical sets would quantify the real generalisation
gap and is the right next step. A chess-specific avenue for future work: because
the starting position is fully known, a single photo of the initial setup yields
32 labelled pieces, which could calibrate or few-shot-adapt the classifier to an
unseen set on the fly.

The code is research/coursework quality. Worth knowing before you run it:

- **`main.py` default model path is broken** (`classifier_with_removed.pkl` does
  not exist) — repoint it as described under [Usage](#usage).
- **External assets required:** the SAM and Depth Pro checkpoints and the
  `ml-depth-pro` package are not vendored; the full pipeline will not run without
  them.
- **Hard-coded relative paths and input directories** (`./small_test`,
  `./every_place`, `../detection/detection_output`, `final_chess_dataset`, etc.)
  mean scripts must be run from the expected working directory.
- **`train_full_dataset.py` reads `./new_dataset`** and
  **`test_detection_model.py` reads `final_chess_dataset/detection_output/`** —
  neither path is in the repo; treat these as experimental/leftover scripts.
- **`labels_final.csv` and `detection/detection_output/labels.csv` are the same
  799 rows** (line-ending differences only).
- **`PROGRESS_SUMMARY.md` is historical** and predates the final dataset (it
  references "58 labeled pieces" and an older folder layout); the README above is
  the current source of truth.
- **No automated tests or `requirements.txt`.**

Known model-level failure modes (from the report): king↔queen confusion,
occasional misses for occluded or image-border pieces, and corner drift under
strong glare on glossy boards.

## Credits

| Member | Contribution |
|---|---|
| **Hugo Taranto** | Board detection; piece detection and segmentation; 2D board reconstruction (50%) |
| **Rakan Abdulla** | Interactive dataset-labelling tool; piece classification; 2D board reconstruction (50%) |
| **Samir Mishra** | Report; RGB camera integration / experimentation for demo testing (`stream/stream_iphone.py`) |

Implementation of the core CV pipeline was done by Hugo Taranto and Rakan
Abdulla; Samir Mishra authored the report and the phone-camera capture utility
used for demo input. Contributions follow the project report and the git history.
