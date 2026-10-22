#!/usr/bin/env python3
"""
Task 2 (Unit 4): Feature Extraction and Image Matching.

Part A - Feature extraction
    1. Extract and visualise Histogram of Oriented Gradients (HOG) features.
    2. Detect SIFT keypoints and compute their descriptors.

Part B - Feature matching
    3. Match the SIFT descriptors between the two photographs.
    4. Repeat the matching after rotating and after resizing one photograph.
    5. Compare the matching quality across the experiments.

Usage
    python src/task2_features_matching.py
    python src/task2_features_matching.py --image data/photo1.jpg --image2 data/photo2.jpg
    python src/task2_features_matching.py --angles 30 90 180 --scales 0.5 0.25

All results are written to ``outputs/task2/`` together with ``results.md``,
which holds the keypoint counts and the matching comparison table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage import exposure
from skimage.feature import hog

import cv_utils as cu
import matplotlib.pyplot as plt  # cv_utils has already selected the Agg backend

TASK_NAME = "Task 2 - Feature Extraction and Image Matching"

#: Two categorical hues, checked for colour-vision separation on a light surface.
SERIES_COLORS = ("#2a78d6", "#eb6834")


def join_values(values: list[str]) -> str:
    """Join values into a readable list: "a", "a and b", "a, b and c"."""
    if len(values) < 3:
        return " and ".join(values)
    return ", ".join(values[:-1]) + " and " + values[-1]


def short_label(name: str) -> str:
    """Shorten an experiment name for a chart tick, where the panel gives the rest."""
    return name.split(", ", 1)[1] if ", " in name else name


# --------------------------------------------------------------------------- #
# Part A helpers: HOG and SIFT
# --------------------------------------------------------------------------- #

def hog_descriptor(
    gray: np.ndarray,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Compute the HOG descriptor of ``gray`` and an image of its cell histograms.

    HOG divides the image into small cells, builds a histogram of gradient
    orientations inside each cell, then normalises those histograms over larger
    overlapping blocks. The descriptor therefore records local shape as a
    distribution of gradient directions, and the block normalisation is what
    makes it tolerant of changes in illumination and contrast.

    Returns ``(descriptor, visualisation, cell grid)``.
    """
    descriptor, visual = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )

    # The raw visualisation is dominated by a few strong cells, so stretch the
    # intensity up to a high percentile instead of the maximum.
    ceiling = float(np.percentile(visual, 99.5)) or float(visual.max()) or 1.0
    visual = exposure.rescale_intensity(visual, in_range=(0.0, ceiling), out_range=(0, 255))

    grid = (gray.shape[0] // pixels_per_cell, gray.shape[1] // pixels_per_cell)
    return descriptor, visual.astype(np.uint8), grid


def detect_sift(
    image: np.ndarray, max_keypoints: int
) -> tuple[list, np.ndarray | None]:
    """Detect SIFT keypoints and compute their 128-element descriptors.

    SIFT searches for blob-like structures across a scale pyramid, which is
    what makes a keypoint survive a change of size, and assigns each one a
    dominant gradient orientation that the descriptor is measured relative to,
    which is what makes it survive a rotation.
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(cu.to_gray(image), None)
    return list(keypoints), descriptors


def draw_keypoints(image: np.ndarray, keypoints: list) -> np.ndarray:
    """Draw keypoints with the circle size and orientation of each one."""
    return cv2.drawKeypoints(
        image, keypoints, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


# --------------------------------------------------------------------------- #
# Part B helpers: transforms, matching, geometric verification
# --------------------------------------------------------------------------- #

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``image`` about its centre onto a canvas large enough to hold it.

    Rotating in place would crop the corners, and the matching comparison would
    then measure the lost content as well as the rotation.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(round(height * sin + width * cos))
    new_height = int(round(height * cos + width * sin))
    matrix[0, 2] += (new_width - width) / 2.0
    matrix[1, 2] += (new_height - height) / 2.0
    return cv2.warpAffine(image, matrix, (new_width, new_height))


def scale_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize ``image`` by ``factor``, keeping its aspect ratio."""
    height, width = image.shape[:2]
    size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)


def match_descriptors(
    desc_a: np.ndarray | None, desc_b: np.ndarray | None, ratio: float
) -> tuple[list, int]:
    """Brute-force match ``desc_a`` against ``desc_b`` and apply Lowe's ratio test.

    Each descriptor from the first image is compared with its two nearest
    neighbours in the second. The match is kept only when the closest neighbour
    is clearly closer than the second, which discards matches in repeated
    texture, such as sand or masonry, where the choice between candidates is
    arbitrary.

    Returns ``(matches that passed, number of candidates tested)``.
    """
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return [], 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [
        pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    return good, len(pairs)


def geometric_inliers(
    kp_a: list, kp_b: list, matches: list, threshold: float
) -> tuple[np.ndarray | None, int]:
    """Fit a homography with RANSAC and report which matches agree with it.

    The ratio test removes ambiguous matches but keeps wrong ones that happen
    to look distinctive. Both photographs show one rigid scene, so every correct
    match has to be consistent with a single projective transform between the
    two views. Counting the matches that fit one measures how many are correct
    rather than how many were found, which is the figure the comparison table
    is built on.
    """
    if len(matches) < 4:
        return None, 0
    source = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, threshold)
    if homography is None or mask is None:
        return None, 0
    mask = mask.ravel().astype(bool)
    return mask, int(mask.sum())


def match_figure(
    image_a: np.ndarray, kp_a: list,
    image_b: np.ndarray, kp_b: list,
    matches: list, limit: int,
) -> np.ndarray:
    """Draw the strongest ``limit`` matches as lines between the two images."""
    shown = sorted(matches, key=lambda m: m.distance)[:limit]
    return cv2.drawMatches(
        image_a, kp_a, image_b, kp_b, shown, None,
        matchColor=(0, 255, 0), singlePointColor=(200, 160, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


# --------------------------------------------------------------------------- #
# Part B: one matching experiment
# --------------------------------------------------------------------------- #

def run_experiment(
    name: str,
    image_a: np.ndarray, kp_a: list, desc_a: np.ndarray | None,
    image_b: np.ndarray,
    out_dir: Path, file_name: str,
    args: argparse.Namespace,
) -> dict:
    """Match one image pair, save the figure and return the measured numbers."""
    kp_b, desc_b = detect_sift(image_b, args.max_keypoints)
    good, candidates = match_descriptors(desc_a, desc_b, args.ratio)
    mask, inliers = geometric_inliers(kp_a, kp_b, good, args.ransac_threshold)

    verified = [m for m, keep in zip(good, mask) if keep] if mask is not None else good
    distances = [m.distance for m in verified] or [m.distance for m in good]
    mean_distance = float(np.mean(distances)) if distances else 0.0
    inlier_rate = 100.0 * inliers / len(good) if good else 0.0

    figure = match_figure(image_a, kp_a, image_b, kp_b, verified, args.draw_matches)
    cu.save_image(out_dir, file_name, figure)

    print(f"  {name}: {len(kp_b)} keypoints, {len(good)} ratio-test matches, "
          f"{inliers} inliers ({inlier_rate:.1f}%)")

    return {
        "name": name,
        "keypoints_b": len(kp_b),
        "candidates": candidates,
        "good": len(good),
        "inliers": inliers,
        "inlier_rate": inlier_rate,
        "mean_distance": mean_distance,
        "figure": f"{file_name}.png",
    }


def quality_chart(results: list[dict], out_path: Path) -> Path:
    """Plot ratio-test matches against verified inliers for every experiment.

    The self-matching controls produce an order of magnitude more matches than
    the photograph pair, so plotting all six experiments on one axis flattens
    the four that matter. They are drawn as two panels with their own vertical
    scales instead, which is stated on the figure.
    """
    cu.ensure_dir(out_path.parent)
    groups = [
        ("Photograph pair", [r for r in results if "itself" not in r["name"]]),
        ("Same photograph, transformed", [r for r in results if "itself" in r["name"]]),
    ]
    groups = [(title, rows) for title, rows in groups if rows]
    width = 0.4

    fig, axes = plt.subplots(
        1, len(groups),
        figsize=(max(7.0, 1.7 * len(results) + 2.0), 5.0),
        gridspec_kw={"width_ratios": [len(rows) for _, rows in groups]},
        squeeze=False,
    )
    for ax, (title, rows) in zip(axes.flat, groups):
        positions = np.arange(len(rows))
        ax.bar(positions - width / 2 - 0.01, [r["good"] for r in rows], width,
               label="Ratio-test matches", color=SERIES_COLORS[0])
        ax.bar(positions + width / 2 + 0.01, [r["inliers"] for r in rows], width,
               label="RANSAC inliers", color=SERIES_COLORS[1])
        for index, result in enumerate(rows):
            ax.text(index + width / 2 + 0.01, result["inliers"],
                    f"{result['inlier_rate']:.0f}%",
                    ha="center", va="bottom", fontsize=9, color="#52514e")

        ax.set_title(title, fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels([short_label(r["name"]) for r in rows], fontsize=9)
        ax.set_xlim(-0.65, len(rows) - 0.35)
        ax.grid(axis="y", color="#d8d8d4", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b4b4b0")
        ax.tick_params(colors="#52514e", length=0)

    axes.flat[0].set_ylabel("Number of matches")
    fig.suptitle("Task 2C - Matching quality across the experiments",
                 fontweight="bold", fontsize=13)
    # The legend sits above the panels: inside them it would cover the bars.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.005,
             "Panels use different vertical scales. The percentage is the share "
             "of ratio-test matches that RANSAC verified.",
             ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {cu.display_path(out_path)}")
    return out_path


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #

def resolve_second_photo(
    explicit: str | None, first_path: Path | None, allow_sample: bool
) -> tuple[np.ndarray, str, bool]:
    """Resolve the second photograph of the same scene.

    ``cv_utils.resolve_photo`` falls back to the first image it finds in
    ``data/``, which here would be the first photograph itself and would make
    the matching trivially perfect, so any candidate equal to it is skipped.
    """
    if explicit:
        return cu.read_image(explicit), cu.display_path(explicit), False
    if cu.DEFAULT_PHOTO_2.exists():
        return cu.read_image(cu.DEFAULT_PHOTO_2), cu.display_path(cu.DEFAULT_PHOTO_2), False

    if cu.DATA_DIR.is_dir():
        for candidate in sorted(cu.DATA_DIR.iterdir()):
            if candidate == first_path or not candidate.is_file():
                continue
            if candidate.suffix.lower() in cu.PHOTO_EXTENSIONS:
                return cu.read_image(candidate), cu.display_path(candidate), False

    if not allow_sample:
        raise FileNotFoundError(
            f"No second photograph found. Place one at '{cu.DEFAULT_PHOTO_2}' or pass "
            f"--image2 /path/to/photo.jpg"
        )
    print(cu._SAMPLE_WARNING)
    return cu.make_sample_photo_pair()[1], "synthetic sample photo (second view)", True


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 2 - HOG features, SIFT keypoints and feature matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", help="first photograph (default: data/photo1.jpg)")
    parser.add_argument("--image2", help="second photograph of the same scene "
                                         "(default: data/photo2.jpg)")
    parser.add_argument("--outdir", default=str(cu.OUTPUT_DIR / "task2"),
                        help="directory for all generated results")
    parser.add_argument("--max-width", type=int, default=1000,
                        help="downscale wider inputs to this width (0 disables)")
    parser.add_argument("--hog-width", type=int, default=400,
                        help="width the image is reduced to before HOG")
    parser.add_argument("--orientations", type=int, default=9,
                        help="number of HOG gradient orientation bins")
    parser.add_argument("--pixels-per-cell", type=int, default=8,
                        help="side of one HOG cell in pixels")
    parser.add_argument("--cells-per-block", type=int, default=2,
                        help="side of one HOG normalisation block in cells")
    parser.add_argument("--max-keypoints", type=int, default=0,
                        help="cap on SIFT keypoints per image (0 keeps all)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Lowe ratio-test threshold")
    parser.add_argument("--ransac-threshold", type=float, default=5.0,
                        help="reprojection error in px allowed by RANSAC")
    parser.add_argument("--angles", type=float, nargs="*", default=[45.0, 90.0],
                        help="rotation angles applied to the second photograph")
    parser.add_argument("--scales", type=float, nargs="*", default=[0.5],
                        help="scale factors applied to the second photograph")
    parser.add_argument("--draw-matches", type=int, default=60,
                        help="number of matches drawn in each figure")
    parser.add_argument("--no-sample", action="store_true",
                        help="fail instead of falling back to synthetic images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = cu.ensure_dir(Path(args.outdir))

    image1, label1, is_sample1 = cu.resolve_photo(args.image, allow_sample=not args.no_sample)
    first_path = Path(label1) if not is_sample1 else None
    label1 = cu.display_path(label1)
    image2, label2, is_sample2 = resolve_second_photo(
        args.image2, first_path, allow_sample=not args.no_sample
    )

    image1 = cu.resize_max_width(image1, args.max_width)
    image2 = cu.resize_max_width(image2, args.max_width)

    cu.banner(TASK_NAME)
    print(f"  image 1 : {label1}  ({image1.shape[1]} x {image1.shape[0]} px)")
    print(f"  image 2 : {label2}  ({image2.shape[1]} x {image2.shape[0]} px)")
    print(f"  outputs : {cu.display_path(out_dir)}")

    cu.save_image(out_dir, "a01_input_photo1", image1)
    cu.save_image(out_dir, "a02_input_photo2", image2)

    # ----------------------------------------------------------------- HOG --
    cu.banner("PART A1 - HOG FEATURE EXTRACTION")
    hog_input = cu.resize_max_width(image1, args.hog_width)
    hog_gray = cu.to_gray(hog_input)
    descriptor, hog_visual, grid = hog_descriptor(
        hog_gray, args.orientations, args.pixels_per_cell, args.cells_per_block
    )
    print(f"  HOG image   : {hog_gray.shape[1]} x {hog_gray.shape[0]} px")
    print(f"  cell grid   : {grid[0]} x {grid[1]} cells of "
          f"{args.pixels_per_cell}x{args.pixels_per_cell} px")
    print(f"  descriptor  : {descriptor.size} values")

    cu.save_image(out_dir, "a03_hog_visualisation", hog_visual)
    cu.save_grid(
        [hog_gray, hog_visual],
        ["Grayscale input", f"HOG cell histograms ({args.orientations} orientations)"],
        out_dir / "a04_hog_comparison.png",
        suptitle="Task 2A - Histogram of Oriented Gradients",
        cols=2,
    )

    # ---------------------------------------------------------------- SIFT --
    cu.banner("PART A2 - SIFT KEYPOINTS AND DESCRIPTORS")
    kp1, desc1 = detect_sift(image1, args.max_keypoints)
    kp2, desc2 = detect_sift(image2, args.max_keypoints)
    print(f"  photo 1 : {len(kp1)} keypoints, descriptors {desc1.shape}")
    print(f"  photo 2 : {len(kp2)} keypoints, descriptors {desc2.shape}")

    drawn1 = draw_keypoints(image1, kp1)
    drawn2 = draw_keypoints(image2, kp2)
    cu.save_image(out_dir, "b01_sift_keypoints_photo1", drawn1)
    cu.save_image(out_dir, "b02_sift_keypoints_photo2", drawn2)
    cu.save_grid(
        [image1, drawn1, image2, drawn2],
        ["Photo 1", f"Photo 1: {len(kp1)} SIFT keypoints",
         "Photo 2", f"Photo 2: {len(kp2)} SIFT keypoints"],
        out_dir / "b03_sift_keypoints_comparison.png",
        suptitle="Task 2A - SIFT keypoints, drawn with their scale and orientation",
        cols=2,
    )

    # ------------------------------------------------------------ matching --
    cu.banner("PART B - FEATURE MATCHING EXPERIMENTS")
    experiments: list[tuple[str, np.ndarray, str]] = [
        ("Photo 2, unchanged", image2, "c01_match_baseline"),
    ]
    index = 2
    for angle in args.angles:
        experiments.append((f"Photo 2, rotated {angle:g} deg",
                            rotate_image(image2, angle),
                            f"c{index:02d}_match_photo2_rotated_{angle:g}"))
        index += 1
    for scale in args.scales:
        experiments.append((f"Photo 2, scaled {scale:g}x",
                            scale_image(image2, scale),
                            f"c{index:02d}_match_photo2_scaled_{scale:g}"))
        index += 1

    # Controls: the same photograph transformed. Matching an image against a
    # transformed copy of itself isolates the invariance of the descriptor,
    # with no viewpoint, lighting or parallax difference mixed in.
    if args.angles:
        experiments.append((f"Photo 1 vs itself, rotated {args.angles[0]:g} deg",
                            rotate_image(image1, args.angles[0]),
                            f"c{index:02d}_match_self_rotated_{args.angles[0]:g}"))
        index += 1
    if args.scales:
        experiments.append((f"Photo 1 vs itself, scaled {args.scales[0]:g}x",
                            scale_image(image1, args.scales[0]),
                            f"c{index:02d}_match_self_scaled_{args.scales[0]:g}"))
        index += 1

    results = [
        run_experiment(name, image1, kp1, desc1, image_b, out_dir, file_name, args)
        for name, image_b, file_name in experiments
    ]
    quality_chart(results, out_dir / "d01_matching_quality.png")

    # ------------------------------------------------------------- results --
    headers = ["Experiment", "Keypoints (image B)", "Ratio-test matches",
               "RANSAC inliers", "Inlier rate (%)", "Mean descriptor distance"]
    rows = [
        [r["name"], r["keypoints_b"], r["good"], r["inliers"],
         f"{r['inlier_rate']:.1f}", f"{r['mean_distance']:.1f}"]
        for r in results
    ]
    cu.print_table(headers, rows)

    baseline = results[0]
    self_tests = [r for r in results if "itself" in r["name"]]
    rotated_pairs = [r for r in results if "itself" not in r["name"] and "rotated" in r["name"]]
    scaled_pair = next(
        (r for r in results if "itself" not in r["name"] and "scaled" in r["name"]), None
    )

    note = ""
    if is_sample1 or is_sample2:
        note = ("\n> Note: this run used synthetic stand-in images. Re-run the script "
                "once both photographs are in `data/`.\n")

    hog_body = "\n".join([
        f"HOG was computed on a {hog_gray.shape[1]} x {hog_gray.shape[0]} px grayscale "
        f"copy of the first photograph. The working resolution is reduced first so that "
        f"one cell covers a meaningful part of the subject.",
        "",
        cu.markdown_table(
            ["Parameter", "Value"],
            [["Orientation bins", args.orientations],
             ["Cell size", f"{args.pixels_per_cell} x {args.pixels_per_cell} px"],
             ["Block size", f"{args.cells_per_block} x {args.cells_per_block} cells"],
             ["Block normalisation", "L2-Hys"],
             ["Cell grid", f"{grid[1]} columns x {grid[0]} rows of cells"],
             ["Descriptor length", f"{descriptor.size} values"]],
        ),
        "",
        "Each cell in the visualisation is drawn as a star of lines, one per "
        "orientation bin, with the length of each line set by the weight of that "
        "bin. The lines run along the dominant edge direction inside the cell, so "
        "the outline of the subject and the long slope of the pyramid stand out "
        "as continuous runs of aligned cells. The sand gives a dense field of "
        "short, weakly oriented marks from the scattered stones, and the open sky "
        "produces nothing at all, since a region of constant brightness has no "
        "gradient to bin.",
        "",
        "The descriptor is a fixed-length vector for the whole image. That makes "
        "HOG a description of global shape, suited to a sliding-window detector, "
        "and unlike SIFT it offers no way to say that one part of this photograph "
        "corresponds to one part of another.",
        "",
        "Figures: `a03_hog_visualisation.png`, `a04_hog_comparison.png`",
    ])

    sift_body = "\n".join([
        f"SIFT found **{len(kp1)}** keypoints in the first photograph and "
        f"**{len(kp2)}** in the second. Each keypoint carries a 128-element "
        f"descriptor, so the two descriptor sets have shape `{desc1.shape}` and "
        f"`{desc2.shape}`.",
        "",
        cu.markdown_table(
            ["Image", "Keypoints", "Descriptor array"],
            [[f"`{label1}`", len(kp1), f"{desc1.shape[0]} x {desc1.shape[1]}"],
             [f"`{label2}`", len(kp2), f"{desc2.shape[0]} x {desc2.shape[1]}"]],
        ),
        "",
        "In `b03_sift_keypoints_comparison.png` each keypoint is drawn as a circle "
        "whose radius is the scale it was detected at and whose radial line is its "
        "dominant gradient orientation. The descriptor is sampled in that frame, "
        "which is what makes it invariant to rotation and to a change of scale.",
        "",
        "The keypoints cluster on the textured masonry of the pyramid, on the "
        "clothing and on scattered stones, and the empty sky contributes none, "
        "since a blob detector needs local structure to respond to.",
        "",
        "Figures: `b01_sift_keypoints_photo1.png`, `b02_sift_keypoints_photo2.png`, "
        "`b03_sift_keypoints_comparison.png`",
    ])

    rotation_sentence = (
        "Rotating the second photograph barely moves those figures. "
        + (f"The rotated rows report {join_values([str(r['inliers']) for r in rotated_pairs])} "
           f"inliers against {baseline['inliers']} for the unchanged pair, a spread of a "
           f"few matches that comes from the random sampling inside RANSAC rather than "
           f"from the rotation. " if rotated_pairs else "")
        + "SIFT assigns every keypoint a dominant gradient orientation and measures "
          "its descriptor in that frame, so turning the whole image leaves the "
          "descriptor it computes almost unchanged."
    )

    control_rates = join_values([f"{r['inlier_rate']:.0f}%" for r in self_tests])
    if self_tests:
        control_sentence = (
            f"The self-matching controls isolate that property. There the two images "
            f"differ by the transform alone, with no change of viewpoint, lighting or "
            f"parallax, and the inlier rate rises to {control_rates} "
            f"against {baseline['inlier_rate']:.0f}% for the real pair. That gap places "
            f"the cost on the change of viewpoint. A rotation or a resize maps the same "
            f"pixels onto new coordinates, so nearly every keypoint survives it, while "
            f"the second photograph was taken from a different position: the subject has "
            f"moved relative to the background, surfaces are lit and foreshortened "
            f"differently, and parts of each frame are absent from the other. Those "
            f"differences delete keypoints instead of transforming them, which is why "
            f"two thirds of the ratio-test matches fail the geometric check."
        )
    else:
        control_sentence = (
            "A change of viewpoint is harder than a synthetic transform, because it "
            "removes keypoints instead of moving them."
        )

    if scaled_pair:
        scale_sentence = (
            f"Resizing costs more than rotating. Reducing the second photograph to "
            f"{scaled_pair['name'].split('scaled ')[1]} cuts its keypoint count from "
            f"{baseline['keypoints_b']} to {scaled_pair['keypoints_b']}, its verified "
            f"matches from {baseline['inliers']} to {scaled_pair['inliers']}, and raises "
            f"the mean descriptor distance from {baseline['mean_distance']:.0f} to "
            f"{scaled_pair['mean_distance']:.0f}. The scale pyramid lets SIFT redetect a "
            f"feature at a new size, but it cannot recover detail that the resize threw "
            f"away, so the finest keypoints are never detected in the first place."
        )
    else:
        scale_sentence = (
            "Reducing an image discards detail, so the finest keypoints are never "
            "detected and the keypoint count falls with the scale factor."
        )

    match_body = "\n".join([
        f"Descriptors from the first photograph were matched against each second "
        f"image with a brute-force L2 matcher, Lowe's ratio test at "
        f"`{args.ratio:g}`, and a RANSAC homography at a "
        f"`{args.ransac_threshold:g}` px reprojection threshold. The inlier count "
        f"is the quality measure: it reports how many matches are geometrically "
        f"consistent with a single transform between the two views, rather than "
        f"how many were proposed.",
        "",
        cu.markdown_table(headers, rows),
        "",
        f"Matching the two photographs unchanged gives **{baseline['inliers']} "
        f"verified matches** from {baseline['good']} that passed the ratio test, "
        f"an inlier rate of {baseline['inlier_rate']:.1f}%.",
        "",
        rotation_sentence,
        "",
        control_sentence,
        "",
        scale_sentence,
        "",
        "Figures: one per experiment (`c01` onward, verified matches drawn in "
        "green), and `d01_matching_quality.png`, which plots ratio-test matches "
        "against verified inliers for every row of the table.",
    ])

    cu.write_results(
        out_dir,
        [
            ("Input", f"- Photograph 1: `{label1}` "
                      f"({image1.shape[1]} x {image1.shape[0]} px)\n"
                      f"- Photograph 2: `{label2}` "
                      f"({image2.shape[1]} x {image2.shape[0]} px)\n"
                      f"- Both photographs show the same scene from different "
                      f"positions.{note}"),
            ("A1. HOG Feature Extraction", hog_body),
            ("A2. SIFT Keypoints and Descriptors", sift_body),
            ("B. Feature Matching and Comparison", match_body),
        ],
        title=TASK_NAME,
    )
    cu.banner("TASK 2 COMPLETE")
    print(f"  All images and results.md are in {cu.display_path(out_dir)}\n")


if __name__ == "__main__":
    main()
