# CCA-3 Computer Vision Assignment (Units 3, 4, 5)

Image restoration, segmentation, feature extraction, object recognition and
motion estimation in Python with OpenCV, NumPy, Matplotlib, scikit-image and
Ultralytics YOLO.

| Task | Unit | Topic | Script |
| ---- | ---- | ----- | ------ |
| 1 | 3 | Image restoration, morphology and segmentation | `src/task1_restoration_segmentation.py` |
| 2 | 4 | Feature extraction and image matching (HOG, SIFT) | `src/task2_features_matching.py` |
| 3 | 5 | Object recognition (Haar cascade, YOLO) | `src/task3_object_recognition.py` |
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

### Task 2 - Feature extraction and image matching (Unit 4)

```bash
python src/task2_features_matching.py
python src/task2_features_matching.py --image data/photo1.jpg --image2 data/photo2.jpg
python src/task2_features_matching.py --angles 30 90 180 --scales 0.5 0.25
python src/task2_features_matching.py --ratio 0.7 --pixels-per-cell 16
```

Part A computes HOG features on the first photograph and saves the cell
histogram visualisation (`--orientations`, `--pixels-per-cell`,
`--cells-per-block`, `--hog-width`), then detects SIFT keypoints in both
photographs and draws each one at its own scale and orientation.

Part B matches the SIFT descriptors between the two photographs with a
brute-force L2 matcher and Lowe's ratio test (`--ratio`), then verifies the
matches by fitting a homography with RANSAC (`--ransac-threshold`). The
matching is repeated with the second photograph rotated (`--angles`) and
resized (`--scales`), and again against transformed copies of the first
photograph as a control. The verified inlier count is the quality measure the
comparison table and chart are built on.

| Option | Meaning |
| ------ | ------- |
| `--image2` | Second photograph of the same scene (default: `data/photo2.jpg`) |
| `--hog-width 400` | Width the image is reduced to before HOG |
| `--pixels-per-cell 8` | Side of one HOG cell in pixels |
| `--orientations 9` | Number of HOG gradient orientation bins |
| `--ratio 0.75` | Lowe ratio-test threshold |
| `--ransac-threshold 5` | Reprojection error in px allowed by RANSAC |
| `--angles 45 90` | Rotation angles applied to the second photograph |
| `--scales 0.5` | Scale factors applied to the second photograph |
| `--max-keypoints 0` | Cap on SIFT keypoints per image (`0` keeps all) |
| `--draw-matches 60` | Number of matches drawn in each figure |

Outputs in `outputs/task2/`:

| Prefix | Content |
| ------ | ------- |
| `a01`-`a04` | Both inputs, HOG visualisation, HOG comparison figure |
| `b01`-`b03` | SIFT keypoints for each photograph and a comparison figure |
| `c01`-`c06` | One match figure per experiment, verified matches drawn in green |
| `d01` | Matching quality chart across all experiments |
| `results.md` | HOG parameters, keypoint counts, matching comparison table |

### Task 3 - Object recognition (Unit 5)

```bash
python src/task3_object_recognition.py
python src/task3_object_recognition.py --conf 0.1
python src/task3_object_recognition.py --scale-factor 1.05 --min-neighbors 3
```

Part A runs four pretrained Haar cascades (frontal face, profile face, upper
body, full body) over each photograph, on a histogram-equalised grayscale copy,
and repeats the frontal-face cascade at six combinations of `--scale-factor`
and `--min-neighbors`, since those two parameters are the only control a
cascade offers over how much it reports.

Part B runs a pretrained YOLO model and draws boxes with class labels and
confidence scores, then repeats the detection at the thresholds in
`--conf-sweep`.

Part C compares the two and identifies the misses and the detections that
nothing supports. A cascade detection counts as supported when at least half of
its box falls inside a person YOLO found; a face detection is held to the top
`--head-fraction` of that person, because a box on someone's shirt is still not
a face.

| Option | Meaning |
| ------ | ------- |
| `--conf 0.25` | YOLO confidence threshold |
| `--conf-sweep 0.1 0.25 0.5` | Thresholds compared in one figure |
| `--scale-factor 1.1` | Image pyramid step of the Haar detector |
| `--min-neighbors 5` | Neighbouring detections required to keep a Haar box |
| `--min-size 30` | Smallest Haar detection in px |
| `--head-fraction 0.2` | Top share of a person box that counts as the head |
| `--cascade-dir` | Directory holding the cascade XML files |
| `--weights` | YOLO weights file |

Outputs in `outputs/task3/`:

| Prefix | Content |
| ------ | ------- |
| `a01`-`a02` | Both input photographs |
| `b01`-`b04` | Cascade detections per photograph, per-cascade comparison, parameter sweep |
| `c01`-`c03` | YOLO detections per photograph, confidence threshold comparison |
| `d01`-`d02` | The two detectors side by side on each photograph |
| `results.md` | Detection tables, parameter sweep, detector comparison, misses |

## Pretrained models

Task 3 needs two pretrained detectors, kept in `models/`:

- `models/cascades/*.xml` are the OpenCV Haar cascades, committed to the
  repository because some OpenCV wheels ship without them. They came from
  `https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/`.
- `models/yolov8n.pt` is downloaded by Ultralytics on the first run and is not
  committed, since `.gitignore` excludes `*.pt`.

## Repository layout

```
cv-assignment/
├── README.md
├── requirements.txt
├── data/                 # photographs and video (see data/README.md)
├── docs/                 # assignment brief
├── models/               # Haar cascades, and YOLO weights fetched at runtime
├── src/
│   ├── cv_utils.py       # shared I/O, figures, tables, synthetic fallbacks
│   ├── task1_restoration_segmentation.py
│   ├── task2_features_matching.py
│   └── task3_object_recognition.py
├── outputs/              # generated results
└── report/               # technical report
```
