# Task 1 - Image Restoration and Segmentation

## Input

- Photograph: `data/photo1.jpg`
- Working resolution: 900 x 1600 px
- Watershed input: `data/photo3_objects.jpg`

## A. Image Restoration (noise + filtering + PSNR/SSIM)

Noise parameters: Gaussian `sigma = 25`, salt-and-pepper `amount = 5.0%` (salt/pepper split 50%/50%). Both filters use a 5x5 kernel; the Gaussian filter uses `sigma = 1.5`.

All values are measured against the original clean photograph (higher PSNR and higher SSIM are better).

| Noise added    | Processing   | PSNR (dB) | SSIM   |
| -------------- | ------------ | --------- | ------ |
| Gaussian noise | degraded     | 20.32     | 0.2638 |
| Gaussian noise | Median 5x5   | 25.84     | 0.5484 |
| Gaussian noise | Gaussian 5x5 | 26.56     | 0.6245 |
| Salt & pepper  | degraded     | 18.43     | 0.3074 |
| Salt & pepper  | Median 5x5   | 27.02     | 0.7281 |
| Salt & pepper  | Gaussian 5x5 | 25.95     | 0.5807 |

Better filter per noise type, on this image:

- Gaussian noise: best PSNR = `Gaussian 5x5` (26.56 dB, up from 20.32 dB), best SSIM = `Gaussian 5x5` (0.6245, up from 0.2638).
- Salt & pepper: best PSNR = `Median 5x5` (27.02 dB, up from 18.43 dB), best SSIM = `Median 5x5` (0.7281, up from 0.3074).

The median filter replaces a pixel with the median of its neighbourhood, so the extreme values of salt-and-pepper noise are discarded and the edges survive. The Gaussian filter takes a weighted average, which smears each impulse across its neighbourhood and blurs edges. Averaging is the correct model for zero-mean Gaussian noise, so the two filters differ far less on that case.

Figure: `a08_restoration_comparison.png`

## B1. Thresholding, Morphology and Canny Edges

Otsu selected a global threshold of **100** automatically. The polarity is normalised so that white = object. All four morphological operators use the same 5x5 elliptical structuring element, so the foreground percentages are directly comparable.

| Operation     | Foreground (%) | Change vs binary | Effect on the mask                                            |
| ------------- | -------------- | ---------------- | ------------------------------------------------------------- |
| Binary (Otsu) | 11.79          | -                | Reference mask                                                |
| Erosion       | 9.28           | -2.51            | Shrinks objects, removes thin protrusions and small specks    |
| Dilation      | 17.36          | +5.57            | Grows objects, fills small gaps and joins nearby blobs        |
| Opening       | 10.61          | -1.18            | Erosion then dilation: removes noise, keeps the original size |
| Closing       | 13.32          | +1.53            | Dilation then erosion: fills holes, keeps the original size   |

Canny was run on a Gaussian-blurred grayscale image with hysteresis thresholds **(99, 196)**, derived from the image median so the edge map needs no re-tuning for a different exposure. The operator is sensitive to that pair, so two fixed settings are included for comparison:

| Canny thresholds | Edge pixels (%) |
| ---------------- | --------------- |
| Auto (99, 196)   | 1.37            |
| Loose (50, 150)  | 3.30            |
| Tight (150, 250) | 0.72            |

Lower thresholds recover faint boundaries and also promote texture and noise to edges. Higher thresholds give a clean map that drops real low-contrast boundaries. Canny works on a grayscale gradient, so two regions that differ strongly in colour and barely in luminance produce no edge, which shows up as gaps where those objects meet.

In the morphology table, erosion and dilation change the object area in opposite directions, as the +/- column shows. Opening and closing are the two composed, so each roughly restores the original area while cleaning the mask: opening deletes specks that erosion removed and dilation could not bring back, closing seals holes that dilation filled and erosion could not reopen.

Figures: `b05_morphology_comparison.png`, `b08_canny_threshold_comparison.png`

## B2. Watershed Segmentation

Input: data/photo3_objects.jpg. Using the **per-component** marker threshold at `0.5 x max(distance)`, the watershed stage reported **21 object region(s)**; the global threshold on the same image reports **1**.

Simple thresholding labels two objects that touch as a single blob. The distance transform resolves that: it peaks at the centre of every object, so thresholding it at 50% of the maximum yields one confident marker per object. `cv2.watershed` then floods outwards from those markers and places a ridge wherever two floods meet, which is on the contact line.

| Stage                                   | Purpose                                                      |
| --------------------------------------- | ------------------------------------------------------------ |
| Otsu binary                             | Separate object pixels from background                       |
| Opening                                 | Delete isolated noise pixels before measuring distances      |
| Dilation -> sure background             | Everything certainly outside an object                       |
| Distance transform                      | Distance of each pixel from the nearest background pixel     |
| Threshold at 50% of max (per-component) | Keep only object cores -> one marker per object              |
| sure_bg - sure_fg                       | The contested band the watershed has to divide               |
| cv2.watershed                           | Flood from the markers; ridges land between touching objects |

A single global cut-off is set by the largest object in the frame, so smaller objects never reach it and lose their marker. Scaling the same fraction inside each connected component gives every blob its own marker, which accounts for the difference between the two counts above. `--global-markers` selects the global variant and `--fg-threshold` tunes the cut-off.

Figure: `c07_watershed_pipeline.png` (boundaries in red, segments colour-coded)
