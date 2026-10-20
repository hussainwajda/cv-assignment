#!/usr/bin/env python3
"""
Task 1 (Unit 3): Image Restoration and Segmentation.

Part A - Image Restoration
    1. Add Gaussian noise and salt-and-pepper noise to the photograph.
    2. Restore each noisy image with a median filter and a Gaussian filter.
    3. Compare every result against the original clean image.
    4. Report PSNR and SSIM for all noisy and restored versions.

Part B - Morphological Processing and Segmentation
    5. Convert the image to binary using Otsu thresholding.
    6. Apply erosion, dilation, opening and closing.
    7. Apply Canny edge detection.
    8. Run marker-controlled watershed segmentation on touching objects.

Usage
    python src/task1_restoration_segmentation.py
    python src/task1_restoration_segmentation.py --image data/photo1.jpg
    python src/task1_restoration_segmentation.py --watershed-image data/coins.jpg

All results are written to ``outputs/task1/`` together with ``results.md``,
which holds the PSNR/SSIM and morphology comparison tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import cv_utils as cu

TASK_NAME = "Task 1 - Image Restoration and Segmentation"


# --------------------------------------------------------------------------- #
# Part A helpers: noise models
# --------------------------------------------------------------------------- #

def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean additive Gaussian noise of standard deviation ``sigma``.

    A model of sensor and thermal noise: every pixel is perturbed by a small
    amount, so the whole image degrades slightly.
    """
    noise = rng.normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float,
    salt_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a fraction ``amount`` of pixels with pure white or pure black.

    A model of transmission and dead-pixel errors: a few pixels are destroyed
    completely (impulse noise) while the rest stay untouched.
    """
    noisy = image.copy()
    height, width = image.shape[:2]
    total = int(round(amount * height * width))
    salt_count = int(round(total * salt_ratio))

    flat_indices = rng.choice(height * width, size=total, replace=False)
    rows, cols = np.unravel_index(flat_indices, (height, width))

    noisy[rows[:salt_count], cols[:salt_count]] = 255   # salt
    noisy[rows[salt_count:], cols[salt_count:]] = 0     # pepper
    return noisy


def quality(reference: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Return ``(PSNR in dB, SSIM)`` of ``test`` measured against ``reference``.

    PSNR is a pixel-wise error measure, where higher is better and above 30 dB
    is good. SSIM compares local luminance, contrast and structure, where 1.0
    is identical, and follows perceived quality more closely.
    """
    psnr = peak_signal_noise_ratio(reference, test, data_range=255)
    ssim = structural_similarity(reference, test, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


# --------------------------------------------------------------------------- #
# Part B helpers: thresholding, morphology, edges, watershed
# --------------------------------------------------------------------------- #

def binarize(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Otsu-threshold ``gray`` so that the objects end up white.

    Otsu picks the threshold that minimises intra-class variance, so it adapts
    to the exposure of the photograph instead of using a fixed value. Every
    morphological operator below assumes white = foreground, so the polarity is
    normalised from the image border: a photograph is framed around its
    subject, so the outer ring of pixels is nearly always background, and the
    result is inverted when that ring comes out white. Counting white pixels
    over the whole image fails once the subject fills more than half the frame.
    """
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate(
        [binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]]
    )
    if np.count_nonzero(border) > 0.5 * border.size:
        binary = cv2.bitwise_not(binary)
    return binary, float(threshold)


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> tuple[np.ndarray, int, int]:
    """Canny edges with thresholds derived from the image median.

    A fixed (100, 200) pair is tuned to one exposure. Deriving the hysteresis
    thresholds from the median keeps the edge map usable on any photograph.
    """
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(gray, lower, upper)
    return edges, lower, upper


def extract_markers(
    opening: np.ndarray,
    distance: np.ndarray,
    fg_threshold: float,
    per_component: bool,
) -> np.ndarray:
    """Threshold the distance transform to get one marker per object.

    The standard recipe thresholds at ``fg_threshold * distance.max()`` over the
    whole image. That single global cut-off is dominated by the largest object:
    a small object whose distance peak never reaches 50% of the global maximum
    loses its marker and is then swallowed by its neighbour. Scaling the
    threshold inside each connected component gives every blob its own marker,
    whatever its size. Both variants are kept, for the comparison in the report.
    """
    if not per_component:
        _, sure_fg = cv2.threshold(
            distance, fg_threshold * distance.max(), 255, cv2.THRESH_BINARY
        )
        return sure_fg.astype(np.uint8)

    component_count, components = cv2.connectedComponents(opening)
    sure_fg = np.zeros(opening.shape, dtype=np.uint8)
    for label in range(1, component_count):
        mask = components == label
        local_max = float(distance[mask].max())
        if local_max <= 0.0:
            continue
        sure_fg[mask & (distance >= fg_threshold * local_max)] = 255
    return sure_fg


def watershed_segment(
    image: np.ndarray,
    fg_threshold: float = 0.5,
    per_component: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Marker-controlled watershed segmentation of touching objects.

    Plain thresholding merges objects that touch into one blob. The distance
    transform peaks at each object centre, so thresholding it yields one
    confident marker per object. Watershed then floods outwards from those
    markers and places a boundary where two floods meet.

    Returns ``(boundary overlay, colour-coded labels, object count, stages)``.
    """
    gray = cu.to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary, _ = binarize(blurred)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)          # certainly background
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)    # distance to nearest edge
    sure_fg = extract_markers(opening, distance, fg_threshold, per_component)
    unknown = cv2.subtract(sure_bg, sure_fg)                     # the contested band

    marker_count, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                 # reserve 0 for "unknown"
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    overlay = image.copy()
    overlay[markers == -1] = (0, 0, 255)  # watershed ridges in red

    # Colour each segment so the separation of touching objects is visible.
    labels = np.zeros_like(image)
    rng = np.random.default_rng(11)
    for label in range(2, markers.max() + 1):
        labels[markers == label] = rng.integers(60, 255, size=3)
    labels[markers == -1] = (255, 255, 255)

    distance_vis = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stages = {
        "binary": binary,
        "opening": opening,
        "sure_bg": sure_bg,
        "distance": distance_vis,
        "sure_fg": sure_fg,
        "unknown": unknown,
    }
    return overlay, labels, marker_count - 1, stages


# --------------------------------------------------------------------------- #
# Part A: restoration pipeline
# --------------------------------------------------------------------------- #

def run_restoration(image: np.ndarray, out_dir: Path, args: argparse.Namespace) -> str:
    """Add noise, restore it two ways and score every result.

    Returns the markdown body for this section of ``results.md``.
    """
    cu.banner("PART A - IMAGE RESTORATION")
    rng = np.random.default_rng(args.seed)

    noisy = {
        "gaussian": add_gaussian_noise(image, args.gauss_sigma, rng),
        "salt_pepper": add_salt_pepper_noise(image, args.sp_amount, args.salt_ratio, rng),
    }

    ksize = args.kernel
    restored: dict[str, np.ndarray] = {}
    for noise_name, noisy_image in noisy.items():
        # Median filter: each pixel becomes the median of its neighbourhood,
        # so a few extreme values cannot drag the result.
        restored[f"{noise_name}_median"] = cv2.medianBlur(noisy_image, ksize)
        # Gaussian filter: weighted average of the neighbourhood. Suited to
        # zero-mean noise, but it averages impulses in instead of rejecting them.
        restored[f"{noise_name}_gaussian"] = cv2.GaussianBlur(
            noisy_image, (ksize, ksize), args.blur_sigma
        )

    cu.save_image(out_dir, "a01_original_clean", image)
    cu.save_image(out_dir, "a02_noisy_gaussian", noisy["gaussian"])
    cu.save_image(out_dir, "a03_noisy_salt_pepper", noisy["salt_pepper"])
    cu.save_image(out_dir, "a04_gaussian_noise_median_filter", restored["gaussian_median"])
    cu.save_image(out_dir, "a05_gaussian_noise_gaussian_filter", restored["gaussian_gaussian"])
    cu.save_image(out_dir, "a06_salt_pepper_median_filter", restored["salt_pepper_median"])
    cu.save_image(out_dir, "a07_salt_pepper_gaussian_filter", restored["salt_pepper_gaussian"])

    cu.save_grid(
        [image, noisy["gaussian"], restored["gaussian_median"], restored["gaussian_gaussian"],
         noisy["salt_pepper"], restored["salt_pepper_median"], restored["salt_pepper_gaussian"]],
        ["Original (clean)",
         f"Gaussian noise (sigma={args.gauss_sigma:g})",
         f"  -> Median {ksize}x{ksize}",
         f"  -> Gaussian {ksize}x{ksize}",
         f"Salt & pepper ({args.sp_amount:.0%})",
         f"  -> Median {ksize}x{ksize}",
         f"  -> Gaussian {ksize}x{ksize}"],
        out_dir / "a08_restoration_comparison.png",
        suptitle="Task 1A - Noise models and restoration filters",
        cols=4,
    )

    rows: list[list] = []
    measurements: dict[str, tuple[float, float]] = {}
    plan = [
        ("Gaussian noise", "degraded", noisy["gaussian"]),
        ("Gaussian noise", f"Median {ksize}x{ksize}", restored["gaussian_median"]),
        ("Gaussian noise", f"Gaussian {ksize}x{ksize}", restored["gaussian_gaussian"]),
        ("Salt & pepper", "degraded", noisy["salt_pepper"]),
        ("Salt & pepper", f"Median {ksize}x{ksize}", restored["salt_pepper_median"]),
        ("Salt & pepper", f"Gaussian {ksize}x{ksize}", restored["salt_pepper_gaussian"]),
    ]
    for noise_label, stage, candidate in plan:
        psnr, ssim = quality(image, candidate)
        measurements[f"{noise_label}|{stage}"] = (psnr, ssim)
        rows.append([noise_label, stage, f"{psnr:.2f}", f"{ssim:.4f}"])

    headers = ["Noise added", "Processing", "PSNR (dB)", "SSIM"]
    cu.print_table(headers, rows)

    # Pick the better filter per noise type from the measured values.
    verdict_lines = []
    for noise_label in ("Gaussian noise", "Salt & pepper"):
        candidates = {
            stage: values for key, values in measurements.items()
            for label, stage in [key.split("|")]
            if label == noise_label and stage != "degraded"
        }
        best_psnr = max(candidates, key=lambda s: candidates[s][0])
        best_ssim = max(candidates, key=lambda s: candidates[s][1])
        degraded = measurements[f"{noise_label}|degraded"]
        verdict_lines.append(
            f"- {noise_label}: best PSNR = `{best_psnr}` "
            f"({candidates[best_psnr][0]:.2f} dB, up from {degraded[0]:.2f} dB), "
            f"best SSIM = `{best_ssim}` "
            f"({candidates[best_ssim][1]:.4f}, up from {degraded[1]:.4f})."
        )

    body = "\n".join([
        f"Noise parameters: Gaussian `sigma = {args.gauss_sigma:g}`, "
        f"salt-and-pepper `amount = {args.sp_amount:.1%}` "
        f"(salt/pepper split {args.salt_ratio:.0%}/{1 - args.salt_ratio:.0%}). "
        f"Both filters use a {ksize}x{ksize} kernel; the Gaussian filter uses "
        f"`sigma = {args.blur_sigma:g}`.",
        "",
        "All values are measured against the original clean photograph "
        "(higher PSNR and higher SSIM are better).",
        "",
        cu.markdown_table(headers, rows),
        "",
        "Better filter per noise type, on this image:",
        "",
        *verdict_lines,
        "",
        "The median filter replaces a pixel with the median of its neighbourhood, "
        "so the extreme values of salt-and-pepper noise are discarded and the edges "
        "survive. The Gaussian filter takes a weighted average, which smears each "
        "impulse across its neighbourhood and blurs edges. Averaging is the correct "
        "model for zero-mean Gaussian noise, so the two filters differ far less on "
        "that case.",
        "",
        "Figure: `a08_restoration_comparison.png`",
    ])
    return body


# --------------------------------------------------------------------------- #
# Part B: morphology, edges and segmentation pipeline
# --------------------------------------------------------------------------- #

def run_morphology(image: np.ndarray, out_dir: Path, args: argparse.Namespace) -> str:
    """Threshold, apply the four morphological operators, run Canny."""
    cu.banner("PART B1 - THRESHOLDING, MORPHOLOGY AND EDGES")
    gray = cu.to_gray(image)
    binary, threshold = binarize(gray)
    print(f"  Otsu threshold = {threshold:.1f}")

    k = args.morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    operations = {
        "erosion": cv2.erode(binary, kernel, iterations=1),
        "dilation": cv2.dilate(binary, kernel, iterations=1),
        "opening": cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
        "closing": cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
    }

    edges, canny_low, canny_high = auto_canny(cv2.GaussianBlur(gray, (5, 5), 0))
    print(f"  Canny hysteresis thresholds = ({canny_low}, {canny_high})")

    cu.save_image(out_dir, "b01_grayscale", gray)
    cu.save_image(out_dir, "b02_binary_otsu", binary)
    for name, result in operations.items():
        cu.save_image(out_dir, f"b03_{name}", result)
    cu.save_image(out_dir, "b04_canny_edges", edges)

    # Canny is sensitive to its hysteresis pair, so record three settings.
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_variants = {
        f"Auto ({canny_low}, {canny_high})": edges,
        "Loose (50, 150)": cv2.Canny(blurred_gray, 50, 150),
        "Tight (150, 250)": cv2.Canny(blurred_gray, 150, 250),
    }
    cu.save_image(out_dir, "b06_canny_loose", canny_variants["Loose (50, 150)"])
    cu.save_image(out_dir, "b07_canny_tight", canny_variants["Tight (150, 250)"])
    cu.save_grid(
        [gray, *canny_variants.values()],
        ["Grayscale", *canny_variants.keys()],
        out_dir / "b08_canny_threshold_comparison.png",
        suptitle="Task 1B - Canny sensitivity to the hysteresis thresholds",
        cols=4,
    )
    canny_rows = [
        [name, f"{100.0 * np.count_nonzero(e) / e.size:.2f}"]
        for name, e in canny_variants.items()
    ]
    cu.print_table(["Canny thresholds", "Edge pixels (%)"], canny_rows)

    cu.save_grid(
        [gray, binary, operations["erosion"], operations["dilation"],
         operations["opening"], operations["closing"], edges],
        ["Grayscale",
         f"Binary (Otsu t={threshold:.0f})",
         f"Erosion {k}x{k}",
         f"Dilation {k}x{k}",
         f"Opening {k}x{k}",
         f"Closing {k}x{k}",
         f"Canny ({canny_low}, {canny_high})"],
        out_dir / "b05_morphology_comparison.png",
        suptitle="Task 1B - Thresholding, morphological operators and edge detection",
        cols=4,
    )

    def foreground_pct(mask: np.ndarray) -> float:
        return 100.0 * np.count_nonzero(mask) / mask.size

    base_pct = foreground_pct(binary)
    headers = ["Operation", "Foreground (%)", "Change vs binary", "Effect on the mask"]
    effects = {
        "erosion": "Shrinks objects, removes thin protrusions and small specks",
        "dilation": "Grows objects, fills small gaps and joins nearby blobs",
        "opening": "Erosion then dilation: removes noise, keeps the original size",
        "closing": "Dilation then erosion: fills holes, keeps the original size",
    }
    rows = [["Binary (Otsu)", f"{base_pct:.2f}", "-", "Reference mask"]]
    for name, result in operations.items():
        pct = foreground_pct(result)
        rows.append([name.capitalize(), f"{pct:.2f}", f"{pct - base_pct:+.2f}", effects[name]])
    cu.print_table(headers, rows)

    body = "\n".join([
        f"Otsu selected a global threshold of **{threshold:.0f}** automatically. The "
        f"polarity is normalised so that white = object. All four morphological "
        f"operators use the same {k}x{k} elliptical structuring element, so the "
        f"foreground percentages are directly comparable.",
        "",
        cu.markdown_table(headers, rows),
        "",
        f"Canny was run on a Gaussian-blurred grayscale image with hysteresis "
        f"thresholds **({canny_low}, {canny_high})**, derived from the image median "
        f"so the edge map needs no re-tuning for a different exposure. The operator "
        f"is sensitive to that pair, so two fixed settings are included for "
        f"comparison:",
        "",
        cu.markdown_table(["Canny thresholds", "Edge pixels (%)"], canny_rows),
        "",
        "Lower thresholds recover faint boundaries and also promote texture and "
        "noise to edges. Higher thresholds give a clean map that drops real "
        "low-contrast boundaries. Canny works on a grayscale gradient, so two "
        "regions that differ strongly in colour and barely in luminance produce no "
        "edge, which shows up as gaps where those objects meet.",
        "",
        "In the morphology table, erosion and dilation change the object area in "
        "opposite directions, as the +/- column shows. Opening and closing are the "
        "two composed, so each roughly restores the original area while cleaning "
        "the mask: opening deletes specks that erosion removed and dilation could "
        "not bring back, closing seals holes that dilation filled and erosion could "
        "not reopen.",
        "",
        "Figures: `b05_morphology_comparison.png`, "
        "`b08_canny_threshold_comparison.png`",
    ])
    return body


def run_watershed(
    image: np.ndarray, out_dir: Path, source_label: str, args: argparse.Namespace
) -> str:
    """Segment touching objects and document every intermediate stage."""
    cu.banner("PART B2 - WATERSHED SEGMENTATION")
    per_component = not args.global_markers
    overlay, labels, count, stages = watershed_segment(
        image, fg_threshold=args.fg_threshold, per_component=per_component
    )
    print(f"  watershed separated {count} object region(s)")

    # Run the other marker strategy as well, for the comparison in the report.
    _, _, other_count, _ = watershed_segment(
        image, fg_threshold=args.fg_threshold, per_component=not per_component
    )
    strategy = "per-component" if per_component else "global"
    other_strategy = "global" if per_component else "per-component"
    print(f"  ({other_strategy} marker threshold would have found {other_count})")

    cu.save_image(out_dir, "c01_watershed_input", image)
    cu.save_image(out_dir, "c02_distance_transform", stages["distance"])
    cu.save_image(out_dir, "c03_sure_foreground", stages["sure_fg"])
    cu.save_image(out_dir, "c04_unknown_region", stages["unknown"])
    cu.save_image(out_dir, "c05_watershed_boundaries", overlay)
    cu.save_image(out_dir, "c06_watershed_segments", labels)

    cu.save_grid(
        [image, stages["binary"], stages["opening"], stages["distance"],
         stages["sure_fg"], stages["unknown"], overlay, labels],
        ["Input", "Binary (Otsu)", "Opening (noise removed)", "Distance transform",
         "Sure foreground (markers)", "Unknown band", "Watershed boundaries",
         f"Segments ({count} found)"],
        out_dir / "c07_watershed_pipeline.png",
        suptitle="Task 1B - Marker-controlled watershed segmentation",
        cols=4,
    )

    headers = ["Stage", "Purpose"]
    rows = [
        ["Otsu binary", "Separate object pixels from background"],
        ["Opening", "Delete isolated noise pixels before measuring distances"],
        ["Dilation -> sure background", "Everything certainly outside an object"],
        ["Distance transform", "Distance of each pixel from the nearest background pixel"],
        [f"Threshold at {args.fg_threshold:.0%} of max ({strategy})",
         "Keep only object cores -> one marker per object"],
        ["sure_bg - sure_fg", "The contested band the watershed has to divide"],
        ["cv2.watershed", "Flood from the markers; ridges land between touching objects"],
    ]
    cu.print_table(headers, rows)

    body = "\n".join([
        f"Input: {source_label}. Using the **{strategy}** marker threshold at "
        f"`{args.fg_threshold:g} x max(distance)`, the watershed stage reported "
        f"**{count} object region(s)**; the {other_strategy} threshold on the same "
        f"image reports **{other_count}**.",
        "",
        "Simple thresholding labels two objects that touch as a single blob. The "
        "distance transform resolves that: it peaks at the centre of every object, "
        "so thresholding it at 50% of the maximum yields one confident marker per "
        "object. `cv2.watershed` then floods outwards from those markers and places "
        "a ridge wherever two floods meet, which is on the contact line.",
        "",
        cu.markdown_table(headers, rows),
        "",
        "A single global cut-off is set by the largest object in the frame, so "
        "smaller objects never reach it and lose their marker. Scaling the same "
        "fraction inside each connected component gives every blob its own marker, "
        "which accounts for the difference between the two counts above. "
        "`--global-markers` selects the global variant and `--fg-threshold` tunes "
        "the cut-off.",
        "",
        "Figure: `c07_watershed_pipeline.png` (boundaries in red, segments colour-coded)",
    ])
    return body


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 1 - image restoration, morphology and segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", help="input photograph (default: data/photo1.jpg)")
    parser.add_argument("--watershed-image",
                        help="separate image with multiple/touching objects "
                             "(default: the same photograph)")
    parser.add_argument("--outdir", default=str(cu.OUTPUT_DIR / "task1"),
                        help="directory for all generated results")
    parser.add_argument("--max-width", type=int, default=1000,
                        help="downscale wider inputs to this width (0 disables)")
    parser.add_argument("--gauss-sigma", type=float, default=25.0,
                        help="standard deviation of the added Gaussian noise")
    parser.add_argument("--sp-amount", type=float, default=0.05,
                        help="fraction of pixels destroyed by salt-and-pepper noise")
    parser.add_argument("--salt-ratio", type=float, default=0.5,
                        help="share of the impulse pixels set to white")
    parser.add_argument("--kernel", type=int, default=5,
                        help="kernel size for the restoration filters (odd)")
    parser.add_argument("--blur-sigma", type=float, default=1.5,
                        help="sigma of the Gaussian restoration filter")
    parser.add_argument("--morph-kernel", type=int, default=5,
                        help="size of the morphological structuring element (odd)")
    parser.add_argument("--fg-threshold", type=float, default=0.5,
                        help="distance-transform cut-off for the watershed markers")
    parser.add_argument("--global-markers", action="store_true",
                        help="use one global distance threshold instead of scaling "
                             "it inside each connected component")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the noise")
    parser.add_argument("--no-sample", action="store_true",
                        help="fail instead of falling back to a synthetic image")
    args = parser.parse_args()

    if args.kernel % 2 == 0 or args.morph_kernel % 2 == 0:
        parser.error("--kernel and --morph-kernel must be odd numbers")
    return args


def main() -> None:
    args = parse_args()
    out_dir = cu.ensure_dir(Path(args.outdir))

    image, label, is_sample = cu.resolve_photo(args.image, allow_sample=not args.no_sample)
    label = cu.display_path(label)  # keep absolute machine paths out of results.md
    image = cu.resize_max_width(image, args.max_width)
    cu.banner(TASK_NAME)
    print(f"  input   : {label}")
    print(f"  size    : {image.shape[1]} x {image.shape[0]} px")
    print(f"  outputs : {out_dir}")

    if args.watershed_image:
        watershed_source = cu.resize_max_width(cu.read_image(args.watershed_image), args.max_width)
        watershed_label = cu.display_path(args.watershed_image)
    else:
        watershed_source, watershed_label = image, label

    restoration_md = run_restoration(image, out_dir, args)
    morphology_md = run_morphology(image, out_dir, args)
    watershed_md = run_watershed(watershed_source, out_dir, watershed_label, args)

    note = ""
    if is_sample:
        note = ("\n> Note: this run used the synthetic stand-in image. Re-run the "
                "script once the photograph is in `data/`.\n")

    cu.write_results(
        out_dir,
        [
            ("Input", f"- Photograph: `{label}`\n"
                      f"- Working resolution: {image.shape[1]} x {image.shape[0]} px\n"
                      f"- Watershed input: `{watershed_label}`{note}"),
            ("A. Image Restoration (noise + filtering + PSNR/SSIM)", restoration_md),
            ("B1. Thresholding, Morphology and Canny Edges", morphology_md),
            ("B2. Watershed Segmentation", watershed_md),
        ],
        title=TASK_NAME,
    )
    cu.banner("TASK 1 COMPLETE")
    print(f"  All images and results.md are in {cu.display_path(out_dir)}\n")


if __name__ == "__main__":
    main()
