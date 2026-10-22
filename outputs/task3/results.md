# Task 3 - Object Recognition with Viola-Jones and YOLO

## Input

- Photograph 1: `data/photo1.jpg` (900 x 1600 px)
- Photograph 2: `data/photo2.jpg` (900 x 1600 px)
- Haar cascades: `models/cascades` (frontal face, profile face, upper body, full body)
- YOLO weights: `yolov8n.pt`, 80 COCO classes

## A. Viola-Jones (Haar Cascade) Detection

Four pretrained cascades were run over each photograph at `scaleFactor = 1.1`, `minNeighbors = 5` and a minimum size of 30 px, on a histogram-equalised grayscale copy.

| Photograph        | Frontal face | Profile face | Upper body | Full body | Runtime (ms) |
| ----------------- | ------------ | ------------ | ---------- | --------- | ------------ |
| `data/photo1.jpg` | 3            | 1            | 0          | 0         | 178          |
| `data/photo2.jpg` | 0            | 0            | 0          | 0         | 117          |

The cascades that responded were frontal face and profile face. The upper body and full body cascades returned nothing on either photograph.

None of those detections is on a face. On the first photograph the cascades reported 4 boxs, and section C shows that every one of them sits somewhere other than a head, while the face that is in the frame goes undetected.

A cascade is a chain of stages, each one a vote over simple intensity-difference features, and a window has to clear every stage to be reported. The frontal-face cascade was trained on upright, forward-facing, unobstructed faces, so a head turned to the side, shaded by a cap and carrying sunglasses breaks the pattern of light and dark regions around the eyes that the early stages test for.

The detector has no confidence to filter on afterwards, so its two parameters are the only control over how much it reports. The frontal face cascade was run at six settings on the first photograph. The last column counts the detections that land on a head, judged against the people YOLO found:

| scaleFactor | minNeighbors | Detections | On a head |
| ----------- | ------------ | ---------- | --------- |
| 1.05        | 3            | 2          | 0         |
| 1.05        | 6            | 2          | 0         |
| 1.1         | 3            | 5          | 0         |
| 1.1         | 6            | 3          | 0         |
| 1.3         | 3            | 0          | 0         |
| 1.3         | 6            | 0          | 0         |

`scaleFactor` sets how fast the detector steps through the image pyramid and `minNeighbors` how many overlapping windows must agree before a box is kept. The count swings from 0 to 5 across these six settings, and not in one direction: a finer pyramid step at `1.05` reports fewer boxes here than `1.1`, because the extra scales it tests change which windows end up adjacent to each other and therefore which ones survive the neighbour count. At `1.3` the steps are coarse enough that nothing passes at all.

Across all six settings, 0 of the detections land on a head. The parameters change how much the cascade reports, not whether it finds this face.

Figures: `b01_haar_photo1.png`, `b03_haar_cascade_comparison.png`, `b04_haar_parameter_sweep.png`

## B. YOLO Detection

`yolov8n.pt` was run at a confidence threshold of `0.25`. The model is pretrained on COCO, so it reports any of 80 classes with a box, a class label and a confidence.

| Photograph        | Detections | Classes | Confidence range | Runtime (ms) |
| ----------------- | ---------- | ------- | ---------------- | ------------ |
| `data/photo1.jpg` | 1          | person  | 0.95             | 19           |
| `data/photo2.jpg` | 2          | person  | 0.63 - 0.90      | 19           |

YOLO divides the image once into a grid and predicts boxes, class scores and a confidence for every cell in a single pass of one network. It is trained on whole objects in natural photographs rather than on a single rigid view, which is why the person is found here in a pose the frontal-face cascade cannot handle.

Because every detection carries a confidence, the output can be thresholded after the fact. On `data/photo2.jpg`:

| Confidence threshold | Detections | Classes |
| -------------------- | ---------- | ------- |
| 0.1                  | 2          | person  |
| 0.25                 | 2          | person  |
| 0.5                  | 2          | person  |

Lowering the threshold to 0.1 changes nothing here. The group of people standing near the pyramid on the right of that photograph is visible to the eye and is not reported at any threshold, so those are not low-confidence detections being filtered out: the model does not produce them at all. They are a few dozen pixels tall, and the detector resizes the image to 640 px before it runs, which leaves almost nothing of them.

Figures: `c01_yolo_photo1.png`, `c03_yolo_confidence_comparison.png`

## C. Comparison of the Two Detectors

Across both photographs the cascades reported 4 detections and YOLO reported 3.

| Property        | Viola-Jones (Haar cascade)                                                          | YOLO                                                                          |
| --------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| What it detects | One trained object class per cascade file                                           | 80 COCO classes from one model                                                |
| Output          | Box only                                                                            | Box, class label and confidence                                               |
| How it works    | Cascade of stages over Haar features, slid over an image pyramid                    | One pass of a convolutional network over the whole image                      |
| Pose tolerance  | Trained on one rigid view, so an off-angle or occluded subject is missed            | Trained on varied natural images, so pose and partial occlusion are tolerated |
| Tuning          | scaleFactor and minNeighbors, set before the run                                    | One confidence threshold, applied after the run                               |
| Cost            | 178 ms on this image for all 4 cascades, CPU only, no model file beyond a small XML | 19 ms on this image, with a yolov8n.pt weights file                           |

### Misses and unsupported detections

The brief asks for at least one case where a detector misses an object or reports one that is not there. Deciding it by overlap keeps the judgement measured. A body detection counts as supported when at least half of its box lies inside a person YOLO found. A face detection is held to the top 20% of that person instead, since a box on someone's shirt is still not a face, and a standing figure is roughly seven heads tall.

- `data/photo1.jpg`: YOLO found 1 person detection at confidence 0.95, the cascades reported 4 detections in total.
- `data/photo2.jpg`: YOLO found 2 person detections at confidence 0.90 and 0.63, the cascades reported 0 detections in total.
- `data/photo1.jpg`: 4 cascade detections failed that test (frontal face at (629, 470), frontal face at (324, 836), frontal face at (18, 778), profile face at (296, 1014)). Some sit on open background and some on the person but well below the head, on the collar and the forearm. Either way no face is there, and the face that is in the frame was not reported by any cascade.

In total 4 cascade detections failed that test, out of 4 reported.

Figures: `d01_detector_comparison_photo1.png`, `d02_detector_comparison_photo2.png`
