# FishNet

## Authors

- [Théo Mercé, PhD](https://github.com/TheoMrc)
- [Emilien Reaud, PhD student](https://github.com/EmilienRD)
- [Etienne Windels, Data scientist](https://github.com/ewindels)

This repository contains the FishNet architecture, supervised training code,
preprocessing and evaluation utilities, and the exact weights used by
DanioTracker for the manuscript revision.

FishNet receives a two-channel grayscale image (current frame and
background-subtracted frame),
predicts larval head locations, extracts one 81 x 81 feature patch per detected
head, predicts nine midline points, and classifies rolling posture.

## Released model

- PyTorch state dictionary: models/fishnet.pt
- ONNX inference graph: models/fishnet.onnx
- Architecture configuration: configs/model.json
- Training configuration: configs/training.json
- PyTorch checkpoint SHA-256:
  8c45b9b94396c030fcc4b69a59a6c1fc939a93597d678a239c09bc0ee458edff
- ONNX graph SHA-256:
  5d5596b8c4c8293a327614ae2709bdeb2950ecbd2b6fbf330fde34ae17786421

The checkpoint is the deployed DanioTracker model, copied from DanioTracker
revision 58e45576498e61f1a496b6a1636cae22be8415f3. The training source was prepared
from FishNet revision 0ae8975c78c215cbddad708f307a727f46b94dc3.

## Repository contents

```text
FishNet/
├── fish_net/                Core model, data loading, inference, metrics, and training code
├── cutils/                  Python fallback and Cython acceleration for zone extraction
├── annotation_app/          Flask application for midline and rolling-posture annotation
├── configs/                 Released model and training configurations
├── models/                  Released PyTorch and ONNX weights
├── examples/                Small real annotated example and background images
├── scripts/                 Standalone ONNX inference utility
├── tests/                   Smoke, annotation-app, and real-example tests
├── pyproject.toml           Project metadata, dependencies, and Ruff/Ty configuration
├── setup.py                 Setuptools/Cython build entry point and console scripts
└── tox.ini                  `tox -e format` automation
```

The `fish_net/` package implements the three-stage FishNet pipeline: head
localization, extraction of one feature patch per detected fish, and prediction
of nine ordered midline points plus rolling posture. `inference.py` loads the
released checkpoint for PyTorch inference, while `training.py`, `load_data.py`,
`models.py`, and `metrics.py` contain the supervised training and evaluation
implementation.

`cutils/zones.py` is a source-compatible Python fallback. The accompanying
`cutils/zones.pyx` can be compiled with Cython to accelerate zone extraction;
`setup.py` declares that extension and the required NumPy/Cython build
dependencies.

`annotation_app/` is a self-contained browser application for placing heads,
tracing nine-point body midlines, suggesting midlines with the released model,
and labelling rolling posture. `models/` and `configs/` contain the exact
released artifacts used by inference. The `examples/` directory is a small
demonstration subset and is not the full article dataset.

## Installation and dependencies

Python 3.10 or newer is required. FishNet also contains a Cython extension.
On Windows, install the *Desktop development with C++* workload from Visual
Studio Build Tools; on Linux, install a compiler and Python development
headers (for example `build-essential` and `python3-dev`) before installing
the package.

### Windows PowerShell

    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[annotation,training,onnx,test,format]"

### Linux or macOS

    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[annotation,training,onnx,test,format]"

The dependency groups are:

- Core inference and training: NumPy, pandas, SciPy, Matplotlib, PyTorch, and tqdm.
- `annotation`: Flask, Pillow, and SciPy for the browser annotation app.
- `training`: Weights & Biases plus the data/preprocessing compatibility dependencies h5py, imutils, OpenCV, and OpenPyXL.
- `onnx`: ONNX Runtime for the standalone ONNX inference script.
- `test`: pytest and the real-example/smoke test utilities.
- `format`: tox, Ruff, and Ty for formatting, lint fixes, and type checking.

The setup metadata installs all runtime, annotation, training, and ONNX
dependencies by default; the extras make each workflow explicit. If only the
Python runtime is needed, use:

    python -m pip install -e .

For development checks, run:

    python -m tox -e format
    python -m pytest -q

`tox -e format` runs `ruff format`, applies safe Ruff lint fixes, and then
runs `ty check`. Both tools use the project configuration in
`pyproject.toml`.

## Inference

Prepare a float32 NumPy array with shape (batch, 2, height, width). Channel zero
is the grayscale frame and channel one is the frame minus its background; both
use the training normalization (pixel values divided by 255 before
subtraction). Optional head
coordinates have shape (batch, fish, 2), use (x, y) order, and use (0, 0) for
padding.

    python -m fish_net.inference frames.npy --head-positions heads.npy --output predictions.npz

The output contains head_logits and, if heads were supplied, midline_logits and
roll_logits. The command loads `models/fishnet.pt` by default. For the complete
iterative tracking pipeline that generates the head coordinates, use
DanioTracker; this repository isolates and documents the network itself.

## Training

The full database is intentionally excluded; only the compact example described
below is included. Arrange the images and annotations as described in this
README, then run:

    python -m fish_net.training --annotations-folder path/to/midline-annotations --output-dir outputs

The source reproduces the hyperparameters in configs/training.json, including
the standardized random seed 42. Training writes the selected checkpoint and last-epoch checkpoint under
models/supervised and can log the run to Weights & Biases.

### Acquiring FishNet-compatible images as in the article

The following summarizes the acquisition conditions reported in the article's
Methods. Groups of 12 zebrafish eleutheroembryos or larvae were recorded in a
5-cm-diameter Petri dish containing 4 mL of filtered E3 medium, giving an
approximately 2-mm liquid depth. A Photron FASTCAM Mini WX100 camera viewed the
dish from above. The dish was illuminated from below with a PHLOX LLUB White
LED 50 x 50 backlight at 4% intensity, controlled by a Gardasoft RT 220-20. The
camera height was adjusted so that the dish filled the field of view, and room
temperature was maintained at 28 degrees C.

Recordings were acquired with Photron FASTCAM Viewer 4 at 1,000 frames/s,
512 x 512 pixels and approximately 0.105 mm/pixel. For the ESLT protocol, the
dish contained two circular stainless-steel electrodes delivering a 10-ms,
20-V pulse. An LED visible within the camera field indicated pulse generation.
Each group was recorded five times, approximately 30 s apart. Videos were
exported as sequential JPG or PNG images in one subfolder per recording.

To build the article's FishNet dataset, full-frame images were randomly sampled
from a diverse collection of ESLT recordings encompassing wild-type and Casper
zebrafish at 4, 5 and 7 dpf after 1-4 h exposures to vehicle controls and
multiple neuroactive compounds. The reported dataset comprised 1,057 images
sampled from 278 independent videos and 11,789 annotated fish instances. For
each fish, the annotator marked the head, traced the body midline, smoothed it
with a parametric cubic spline, resampled it to nine equidistant ordered points
from head to tail, and classified posture as normal or balance loss/rolling.

These settings describe how the article data were acquired; substantially
different magnification, contrast, orientation or spatial scale should be
validated manually and may require new annotations and retraining.

### Training-data layout

The loader and annotation app expect the following structure:

    annotation_data/
      experiment_name/
        video_name/
          background.jpg
          frame_000001.jpg
          frame_000002.jpg
          annotations.json

Frames and `background.jpg` must be 512 x 512 grayscale images. Use a matching
static background from the same recording geometry; FishNet forms its second
input channel by subtracting this background from the normalized frame. Each
frame key in `annotations.json` contains a list of fish objects with exactly
nine ordered `midline_points` and a `rolling_proba`, as illustrated by the real
example under `examples/annotation_data/`. Coordinates use image row as `x` and
image column as `y` in the annotation app. A `rolling_proba` above 0.5 is a
positive rolling label.

## Annotation app

The browser app used for FishNet midlines is included in `annotation_app/`.
It was recovered from the historical `midline_annotation_app`, then made
self-contained and connected to the released FishNet checkpoint. The similarly
named current `annotation_app` in Fish-Annotation-Apps only records head
coordinates and is therefore not the FishNet midline tool.

Install the optional dependencies and point the app at a local data directory:

    python -m pip install -e ".[annotation]"
    fishnet-annotate --data-dir path/to/midline-annotations

Open http://127.0.0.1:5000. The app supports head placement, nine ordered
midline points, spline resampling, model-assisted midline suggestions, frame
review status, and rolling-posture labels. `annotations.json` contains only
training records; the app keeps its progress flags in `review_state.json` so
the file can be read directly by `fish_net.load_data.SupervisedDataset`.

Each video folder needs `background.jpg` and one or more frame images. An
existing `annotations.json` is optional: the app discovers images and creates
the file on the first save. Data remain outside the source package and are
ignored if the default `annotation_data/` directory is used.

### Included real-data example

Three annotated 512 x 512 frames and their background image are included under
`examples/annotation_data/`. They are a small, unmodified subset of the real
FishNet midline-annotation database. After installing the annotation extras,
launch the app directly on this example with:

    fishnet-annotate --data-dir examples/annotation_data

Then open http://127.0.0.1:5000 and select the displayed experiment and video.
The example is intended only to exercise and demonstrate the annotation
interface; it is not an additional training dataset.

## Verification

    python -m pytest -q

The formatting and static-analysis check is:

    python -m tox -e format

The tests load the released state dictionary, run all three output heads on
synthetic input, compare the released ONNX graph with PyTorch when ONNX Runtime
is installed, verify an annotation-app save can be consumed by the FishNet
dataset loader, and exercise the included real-data example.

## Data availability

Apart from the compact annotation-app example documented above, no images,
videos, annotations, tabular observations, or derived databases are included.

## License

The source repositories did not contain a license file. Add the authors'
selected license before public release.


