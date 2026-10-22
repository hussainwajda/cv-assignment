# Task 2 - Feature Extraction and Image Matching

## Input

- Photograph 1: `data/photo1.jpg` (900 x 1600 px)
- Photograph 2: `data/photo2.jpg` (900 x 1600 px)
- Both photographs show the same scene from different positions.

## A1. HOG Feature Extraction

HOG was computed on a 400 x 711 px grayscale copy of the first photograph. The working resolution is reduced first so that one cell covers a meaningful part of the subject.

| Parameter           | Value                         |
| ------------------- | ----------------------------- |
| Orientation bins    | 9                             |
| Cell size           | 8 x 8 px                      |
| Block size          | 2 x 2 cells                   |
| Block normalisation | L2-Hys                        |
| Cell grid           | 50 columns x 88 rows of cells |
| Descriptor length   | 153468 values                 |

Each cell in the visualisation is drawn as a star of lines, one per orientation bin, with the length of each line set by the weight of that bin. The lines run along the dominant edge direction inside the cell, so the outline of the subject and the long slope of the pyramid stand out as continuous runs of aligned cells. The sand gives a dense field of short, weakly oriented marks from the scattered stones, and the open sky produces nothing at all, since a region of constant brightness has no gradient to bin.

The descriptor is a fixed-length vector for the whole image. That makes HOG a description of global shape, suited to a sliding-window detector, and unlike SIFT it offers no way to say that one part of this photograph corresponds to one part of another.

Figures: `a03_hog_visualisation.png`, `a04_hog_comparison.png`

## A2. SIFT Keypoints and Descriptors

SIFT found **2202** keypoints in the first photograph and **1561** in the second. Each keypoint carries a 128-element descriptor, so the two descriptor sets have shape `(2202, 128)` and `(1561, 128)`.

| Image             | Keypoints | Descriptor array |
| ----------------- | --------- | ---------------- |
| `data/photo1.jpg` | 2202      | 2202 x 128       |
| `data/photo2.jpg` | 1561      | 1561 x 128       |

In `b03_sift_keypoints_comparison.png` each keypoint is drawn as a circle whose radius is the scale it was detected at and whose radial line is its dominant gradient orientation. The descriptor is sampled in that frame, which is what makes it invariant to rotation and to a change of scale.

The keypoints cluster on the textured masonry of the pyramid, on the clothing and on scattered stones, and the empty sky contributes none, since a blob detector needs local structure to respond to.

Figures: `b01_sift_keypoints_photo1.png`, `b02_sift_keypoints_photo2.png`, `b03_sift_keypoints_comparison.png`

## B. Feature Matching and Comparison

Descriptors from the first photograph were matched against each second image with a brute-force L2 matcher, Lowe's ratio test at `0.75`, and a RANSAC homography at a `5` px reprojection threshold. The inlier count is the quality measure: it reports how many matches are geometrically consistent with a single transform between the two views, rather than how many were proposed.

| Experiment                        | Keypoints (image B) | Ratio-test matches | RANSAC inliers | Inlier rate (%) | Mean descriptor distance |
| --------------------------------- | ------------------- | ------------------ | -------------- | --------------- | ------------------------ |
| Photo 2, unchanged                | 1561                | 111                | 37             | 33.3            | 137.2                    |
| Photo 2, rotated 45 deg           | 1744                | 110                | 39             | 35.5            | 140.5                    |
| Photo 2, rotated 90 deg           | 1564                | 111                | 36             | 32.4            | 135.3                    |
| Photo 2, scaled 0.5x              | 316                 | 107                | 27             | 25.2            | 199.3                    |
| Photo 1 vs itself, rotated 45 deg | 3066                | 1294               | 1274           | 98.5            | 74.5                     |
| Photo 1 vs itself, scaled 0.5x    | 571                 | 424                | 385            | 90.8            | 47.0                     |

Matching the two photographs unchanged gives **37 verified matches** from 111 that passed the ratio test, an inlier rate of 33.3%.

Rotating the second photograph barely moves those figures. The rotated rows report 39 and 36 inliers against 37 for the unchanged pair, a spread of a few matches that comes from the random sampling inside RANSAC rather than from the rotation. SIFT assigns every keypoint a dominant gradient orientation and measures its descriptor in that frame, so turning the whole image leaves the descriptor it computes almost unchanged.

The self-matching controls isolate that property. There the two images differ by the transform alone, with no change of viewpoint, lighting or parallax, and the inlier rate rises to 98% and 91% against 33% for the real pair. That gap places the cost on the change of viewpoint. A rotation or a resize maps the same pixels onto new coordinates, so nearly every keypoint survives it, while the second photograph was taken from a different position: the subject has moved relative to the background, surfaces are lit and foreshortened differently, and parts of each frame are absent from the other. Those differences delete keypoints instead of transforming them, which is why two thirds of the ratio-test matches fail the geometric check.

Resizing costs more than rotating. Reducing the second photograph to 0.5x cuts its keypoint count from 1561 to 316, its verified matches from 37 to 27, and raises the mean descriptor distance from 137 to 199. The scale pyramid lets SIFT redetect a feature at a new size, but it cannot recover detail that the resize threw away, so the finest keypoints are never detected in the first place.

Figures: one per experiment (`c01` onward, verified matches drawn in green), and `d01_matching_quality.png`, which plots ratio-test matches against verified inliers for every row of the table.
