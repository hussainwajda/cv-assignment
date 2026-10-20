# CCA-3 Computer Vision Assignment (Units 3, 4, 5)

Image restoration, segmentation, feature extraction, object recognition and
motion estimation in Python with OpenCV, NumPy, Matplotlib, scikit-image and
Ultralytics YOLO.

| Task | Unit | Topic | Script |
| ---- | ---- | ----- | ------ |
| 1 | 3 | Image restoration, morphology and segmentation | `src/task1_restoration_segmentation.py` |
| 2 | 4 | Feature extraction and image matching (HOG, SIFT) | pending |
| 3 | 5 | Object recognition (Haar cascade, YOLO) | pending |
| 4 | 5 | Video motion estimation | pending |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9 or newer. On a machine without a display, install
`opencv-contrib-python-headless` instead of `opencv-contrib-python`; no script
opens a window, every result is written to disk.

## Input dataset

The scripts read from `data/`. See `data/README.md` for details.

| File | Description | Used by |
| ---- | ----------- | ------- |
| `data/photo1.jpg` | Original photograph | Tasks 1, 2, 3 |
| `data/photo2.jpg` | Second photograph of the same scene from a different angle | Task 2 |
| `data/video.mp4` | Self-recorded 15-30 s video | Tasks 3, 4 |

The `--image` and `--video` arguments override these defaults. When a file is
missing the script falls back to a synthetic stand-in image so the pipeline can
still be executed, and prints a warning; `--no-sample` turns that fallback into
an error. The submitted results use the real media.

## Running the tasks

```bash
python src/task1_restoration_segmentation.py
```

Each script writes to `outputs/<task>/`: numbered PNGs for every step, labelled
comparison figures, and a `results.md` with that task's measurement tables.
Those `results.md` files are the source material for the report in `report/`.

### Task 1 - Image restoration and segmentation (Unit 3)

```bash
python src/task1_restoration_segmentation.py
python src/task1_restoration_segmentation.py --image data/photo1.jpg
python src/task1_restoration_segmentation.py --watershed-image data/coins.jpg
python src/task1_restoration_segmentation.py --gauss-sigma 35 --sp-amount 0.08
python src/task1_restoration_segmentation.py --global-markers
```

Part A adds Gaussian noise (`--gauss-sigma`) and salt-and-pepper noise
(`--sp-amount`) to the photograph, restores each with a median filter and a
Gaussian filter (`--kernel`, `--blur-sigma`), and scores every version against
the clean original with PSNR and SSIM.

Part B covers Otsu thresholding with the polarity normalised from the image
border, erosion, dilation, opening and closing with a shared elliptical
structuring element (`--morph-kernel`), Canny edge detection at three threshold
settings, and marker-controlled watershed segmentation driven by a distance
transform.

| Option | Meaning |
| ------ | ------- |
| `--max-width 1000` | Downscale wide photos before processing (`0` disables) |
| `--kernel 5` | Kernel size of both restoration filters (odd) |
| `--morph-kernel 5` | Size of the morphological structuring element (odd) |
| `--fg-threshold 0.5` | Distance-transform cut-off that creates the watershed markers |
| `--global-markers` | One global cut-off instead of scaling it per connected component |
| `--watershed-image` | Use a different image for the watershed stage |
| `--seed 42` | RNG seed, so the added noise is reproducible |
| `--no-sample` | Fail instead of falling back to synthetic media |

Watershed only demonstrates anything when objects touch, so a close-up of
coins, beans or bottle caps passed with `--watershed-image` works better than a
landscape.

Outputs in `outputs/task1/`:

| Prefix | Content |
| ------ | ------- |
| `a01`-`a08` | Clean image, both noisy versions, four restorations, comparison figure |
| `b01`-`b08` | Grayscale, Otsu binary, four morphological results, Canny maps, comparison figures |
| `c01`-`c07` | Watershed input, distance transform, markers, boundaries, segments, pipeline figure |
| `results.md` | PSNR/SSIM table, morphology table, Canny table, watershed explanation |

## Repository layout

```
cv-assignment/
├── README.md
├── requirements.txt
├── data/                 # photographs and video (see data/README.md)
├── docs/                 # assignment brief
├── src/
│   ├── cv_utils.py       # shared I/O, figures, tables, synthetic fallbacks
│   └── task1_restoration_segmentation.py
├── outputs/              # generated results
└── report/               # technical report
```
