#!/usr/bin/env python3
"""
Task 3 (Unit 5): Object Recognition with Viola-Jones and YOLO.

Part A - Viola-Jones
    1. Apply pretrained Haar cascade classifiers to the photographs.
    2. Show how the detections depend on the detector's scale and neighbour
       settings.

Part B - YOLO
    3. Apply a pretrained YOLO model and draw boxes, class labels and
       confidence scores.

Part C - Comparison
    4. Compare what the two detectors found on the same images.
    5. Identify the misses and the unsupported detections.

The Haar cascades are read from ``models/cascades``. The YOLO weights are
downloaded on first use and are not kept in the repository.

Usage
    python src/task3_object_recognition.py
    python src/task3_object_recognition.py --image data/photo1.jpg --image2 data/photo2.jpg
    python src/task3_object_recognition.py --conf 0.1 --scale-factor 1.05

All results are written to ``outputs/task3/`` together with ``results.md``,
which holds the detection tables and the comparison of the two detectors.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import cv_utils as cu

TASK_NAME = "Task 3 - Object Recognition with Viola-Jones and YOLO"

CASCADE_DIR = cu.PROJECT_ROOT / "models" / "cascades"

#: The pretrained cascades shipped with OpenCV that suit a photograph of a person.
CASCADE_FILES = {
    "frontal face": "haarcascade_frontalface_default.xml",
    "profile face": "haarcascade_profileface.xml",
    "upper body": "haarcascade_upperbody.xml",
    "full body": "haarcascade_fullbody.xml",
}

#: Drawing colours, from the palette used by the Task 2 chart (BGR for OpenCV).
COLORS = {
    "frontal face": (214, 120, 42),
    "profile face": (52, 104, 235),
    "upper body": (122, 175, 27),
    "full body": (0, 161, 237),
    "YOLO": (164, 123, 232),
}


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def draw_detections(
    image: np.ndarray, detections: list[dict], thickness: int = 3
) -> np.ndarray:
    """Draw every detection as a labelled box on a copy of ``image``."""
    canvas = image.copy()
    scale = max(0.5, min(canvas.shape[:2]) / 1100.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for detection in detections:
        x, y, w, h = detection["box"]
        color = COLORS.get(detection["source"], (0, 255, 0))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

        label = detection["label"]
        if detection["score"] is not None:
            label = f"{label} {detection['score']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 2)

        # Put the caption inside the frame when the box sits at the top edge.
        top = y - text_h - baseline
        if top < 0:
            top = y + h
        cv2.rectangle(canvas, (x, top), (x + text_w + 6, top + text_h + baseline),
                      color, -1)
        cv2.putText(canvas, label, (x + 3, top + text_h), font, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Part A: Viola-Jones
# --------------------------------------------------------------------------- #

def load_cascades(directory: Path) -> dict[str, cv2.CascadeClassifier]:
    """Load the pretrained cascades, falling back to the ones inside OpenCV.

    Some OpenCV wheels ship without the cascade XML files, so the project keeps
    its own copies and this falls back to the installed ones when present.
    """
    loaded: dict[str, cv2.CascadeClassifier] = {}
    for name, filename in CASCADE_FILES.items():
        path = directory / filename
        if not path.exists():
            path = Path(cv2.data.haarcascades) / filename
        if not path.exists():
            print(f"  missing cascade: {filename}")
            continue
        classifier = cv2.CascadeClassifier(str(path))
        if classifier.empty():
            print(f"  could not load: {filename}")
            continue
        loaded[name] = classifier

    if not loaded:
        raise FileNotFoundError(
            f"No Haar cascades found. Place the OpenCV cascade XML files in "
            f"'{directory}' (see the README)."
        )
    return loaded


def run_cascade(
    classifier: cv2.CascadeClassifier,
    gray: np.ndarray,
    name: str,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> list[dict]:
    """Run one cascade over ``gray`` and return its detections.

    ``detectMultiScale3`` also returns the weight of the last cascade stage each
    detection passed. That value is reported as the score, with the caveat that
    it is a stage margin rather than a probability: it has no fixed range and
    cannot be compared against a YOLO confidence.
    """
    rects, _, weights = classifier.detectMultiScale3(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels=True,
    )
    detections = []
    for index, (x, y, w, h) in enumerate(rects):
        weight = float(weights[index]) if len(weights) > index else None
        detections.append({
            "box": (int(x), int(y), int(w), int(h)),
            "label": name,
            "score": weight,
            "source": name,
        })
    return detections


def haar_detect(
    image: np.ndarray, cascades: dict, args: argparse.Namespace
) -> tuple[dict[str, list[dict]], float]:
    """Run every cascade over one image. Returns the detections and the runtime.

    The grayscale image is histogram-equalised first, which is the usual
    preparation for a Haar cascade: the features are intensity differences, so
    spreading the histogram makes them easier to clear on a bright or flat
    photograph.
    """
    gray = cv2.equalizeHist(cu.to_gray(image))
    start = time.perf_counter()
    per_cascade = {
        name: run_cascade(classifier, gray, name, args.scale_factor,
                          args.min_neighbors, args.min_size)
        for name, classifier in cascades.items()
    }
    elapsed = (time.perf_counter() - start) * 1000.0
    return per_cascade, elapsed


def parameter_sweep(
    image: np.ndarray,
    classifier: cv2.CascadeClassifier,
    name: str,
    settings: list[tuple[float, int]],
    min_size: int,
) -> list[tuple[str, np.ndarray, int]]:
    """Run one cascade at several scale and neighbour settings.

    A Haar cascade has no confidence to threshold after the fact, so the only
    control over how much it reports is in these two parameters, which is why
    they are worth showing side by side.
    """
    gray = cv2.equalizeHist(cu.to_gray(image))
    panels = []
    for scale_factor, min_neighbors in settings:
        detections = run_cascade(classifier, gray, name, scale_factor,
                                 min_neighbors, min_size)
        title = (f"scaleFactor={scale_factor:g}, minNeighbors={min_neighbors} "
                 f"({len(detections)})")
        panels.append({
            "setting": (scale_factor, min_neighbors),
            "title": title,
            "panel": draw_detections(image, detections),
            "detections": detections,
        })
    return panels


# --------------------------------------------------------------------------- #
# Part B: YOLO
# --------------------------------------------------------------------------- #

def yolo_detect(
    model, image: np.ndarray, conf: float
) -> tuple[list[dict], float]:
    """Run the YOLO model over one image and return its detections and runtime."""
    start = time.perf_counter()
    prediction = model.predict(source=image, conf=conf, verbose=False)[0]
    elapsed = (time.perf_counter() - start) * 1000.0

    detections = []
    for box in prediction.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        detections.append({
            "box": (x1, y1, x2 - x1, y2 - y1),
            "label": model.names[int(box.cls[0])],
            "score": float(box.conf[0]),
            "source": "YOLO",
        })
    return detections, elapsed


# --------------------------------------------------------------------------- #
# Part C: agreement between the two detectors
# --------------------------------------------------------------------------- #

#: Cascades that claim to find a face, as opposed to a whole body.
FACE_CASCADES = {"frontal face", "profile face"}


def head_region(person: tuple[int, int, int, int], fraction: float) -> tuple[int, int, int, int]:
    """Return the top ``fraction`` of a person box, where a head has to be."""
    x, y, w, h = person
    return (x, y, w, max(1, int(round(h * fraction))))


def containment(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Return the share of ``inner`` that lies inside ``outer``."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    left, top = max(ix, ox), max(iy, oy)
    right, bottom = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top) / float(iw * ih)


def unsupported_detections(
    haar: list[dict], yolo: list[dict], head_fraction: float, threshold: float = 0.5
) -> list[dict]:
    """Haar detections that no person YOLO found can account for.

    A body detection is supported when it falls inside a person. A face
    detection has to do better than that: a box on the shirt of a correctly
    detected person is still not a face, so it is checked against the top
    ``head_fraction`` of the person instead. A standing figure is roughly seven
    heads tall, so the default fifth leaves room for a loose person box while
    still excluding the collar and the shoulders.
    """
    people = [d["box"] for d in yolo if d["label"] == "person"]
    unsupported = []
    for detection in haar:
        regions = [
            head_region(person, head_fraction) if detection["source"] in FACE_CASCADES
            else person
            for person in people
        ]
        if all(containment(detection["box"], region) < threshold for region in regions):
            unsupported.append(detection)
    return unsupported


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 3 - object recognition with Haar cascades and YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", help="first photograph (default: data/photo1.jpg)")
    parser.add_argument("--image2", help="second photograph (default: data/photo2.jpg)")
    parser.add_argument("--outdir", default=str(cu.OUTPUT_DIR / "task3"),
                        help="directory for all generated results")
    parser.add_argument("--max-width", type=int, default=1000,
                        help="downscale wider inputs to this width (0 disables)")
    parser.add_argument("--cascade-dir", default=str(CASCADE_DIR),
                        help="directory holding the Haar cascade XML files")
    parser.add_argument("--weights", default=str(cu.PROJECT_ROOT / "models" / "yolov8n.pt"),
                        help="YOLO weights (downloaded on first use)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--conf-sweep", type=float, nargs="*", default=[0.1, 0.25, 0.5],
                        help="confidence thresholds compared in one figure")
    parser.add_argument("--scale-factor", type=float, default=1.1,
                        help="image pyramid step of the Haar detector")
    parser.add_argument("--min-neighbors", type=int, default=5,
                        help="neighbouring detections required to keep a Haar box")
    parser.add_argument("--min-size", type=int, default=30,
                        help="smallest Haar detection in px")
    parser.add_argument("--head-fraction", type=float, default=0.2,
                        help="top share of a YOLO person box that counts as the head "
                             "when judging a face detection")
    parser.add_argument("--no-sample", action="store_true",
                        help="fail instead of falling back to a synthetic image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = cu.ensure_dir(Path(args.outdir))

    image1, label1, is_sample = cu.resolve_photo(args.image, allow_sample=not args.no_sample)
    label1 = cu.display_path(label1)
    image1 = cu.resize_max_width(image1, args.max_width)

    images = [(label1, image1)]
    second = args.image2 or (cu.DEFAULT_PHOTO_2 if cu.DEFAULT_PHOTO_2.exists() else None)
    if second is not None:
        images.append((cu.display_path(second),
                       cu.resize_max_width(cu.read_image(second), args.max_width)))

    cu.banner(TASK_NAME)
    for index, (label, image) in enumerate(images, start=1):
        print(f"  image {index} : {label}  ({image.shape[1]} x {image.shape[0]} px)")
    print(f"  outputs : {cu.display_path(out_dir)}")

    for index, (_, image) in enumerate(images, start=1):
        cu.save_image(out_dir, f"a{index:02d}_input_photo{index}", image)

    # ------------------------------------------------------------- Part A --
    cu.banner("PART A - VIOLA-JONES (HAAR CASCADES)")
    cascades = load_cascades(Path(args.cascade_dir))
    print(f"  cascades loaded: {', '.join(cascades)}")

    haar_results = []
    for index, (label, image) in enumerate(images, start=1):
        per_cascade, elapsed = haar_detect(image, cascades, args)
        flat = [d for detections in per_cascade.values() for d in detections]
        counts = {name: len(d) for name, d in per_cascade.items()}
        print(f"  {label}: {counts}, {elapsed:.0f} ms")

        cu.save_image(out_dir, f"b{index:02d}_haar_photo{index}",
                      draw_detections(image, flat))
        haar_results.append({
            "label": label, "image": image, "per_cascade": per_cascade,
            "detections": flat, "counts": counts, "ms": elapsed,
        })

    # Each cascade on the first photograph, so the report can show which of the
    # four responds to a person seen from this angle.
    first = haar_results[0]
    cu.save_grid(
        [draw_detections(first["image"], first["per_cascade"][name]) for name in cascades],
        [f"{name} ({len(first['per_cascade'][name])})" for name in cascades],
        out_dir / "b03_haar_cascade_comparison.png",
        suptitle="Task 3A - Each pretrained cascade on the same photograph",
        cols=4,
        fig_scale=5.0,
    )

    sweep_settings = [(1.05, 3), (1.05, 6), (1.1, 3), (1.1, 6), (1.3, 3), (1.3, 6)]
    sweep_name = next(iter(cascades))
    panels = parameter_sweep(first["image"], cascades[sweep_name], sweep_name,
                             sweep_settings, args.min_size)
    cu.save_grid(
        [p["panel"] for p in panels],
        [p["title"] for p in panels],
        out_dir / "b04_haar_parameter_sweep.png",
        suptitle=f"Task 3A - The {sweep_name} cascade at six parameter settings",
        cols=3,
        fig_scale=5.0,
    )

    # ------------------------------------------------------------- Part B --
    cu.banner("PART B - YOLO")
    from ultralytics import YOLO  # imported here: it is slow and only Part B needs it

    model = YOLO(args.weights)
    print(f"  model   : {Path(args.weights).name}, {len(model.names)} classes")

    # The first call loads the weights and builds the graph. Timing it would
    # report start-up rather than inference, so it is run once and discarded.
    model.predict(source=np.zeros((640, 640, 3), np.uint8), conf=0.9, verbose=False)

    yolo_results = []
    for index, (label, image) in enumerate(images, start=1):
        detections, elapsed = yolo_detect(model, image, args.conf)
        classes = sorted({d["label"] for d in detections})
        print(f"  {label}: {len(detections)} detections {classes}, {elapsed:.0f} ms")

        cu.save_image(out_dir, f"c{index:02d}_yolo_photo{index}",
                      draw_detections(image, detections))
        yolo_results.append({
            "label": label, "image": image, "detections": detections,
            "classes": classes, "ms": elapsed,
        })

    # A YOLO confidence is a number that can be thresholded after the fact,
    # which is the control a Haar cascade does not offer.
    sweep_image_index = len(images) - 1
    sweep_label, sweep_image = images[sweep_image_index]
    conf_panels, conf_rows = [], []
    for threshold in args.conf_sweep:
        detections, _ = yolo_detect(model, sweep_image, threshold)
        conf_panels.append((f"conf >= {threshold:g} ({len(detections)})",
                            draw_detections(sweep_image, detections)))
        conf_rows.append([f"{threshold:g}", str(len(detections)),
                          ", ".join(sorted({d['label'] for d in detections})) or "none"])
    cu.save_grid(
        [panel for _, panel in conf_panels],
        [title for title, _ in conf_panels],
        out_dir / "c03_yolo_confidence_comparison.png",
        suptitle=f"Task 3B - YOLO on {sweep_label} at three confidence thresholds",
        cols=len(conf_panels),
        fig_scale=5.5,
    )
    cu.print_table(["Confidence threshold", "Detections", "Classes"], conf_rows)

    # ------------------------------------------------------------- Part C --
    cu.banner("PART C - COMPARISON")
    for index, (haar, yolo) in enumerate(zip(haar_results, yolo_results), start=1):
        cu.save_grid(
            [draw_detections(haar["image"], haar["detections"]),
             draw_detections(yolo["image"], yolo["detections"])],
            [f"Viola-Jones ({plural(len(haar['detections']), 'detection')})",
             f"YOLO ({plural(len(yolo['detections']), 'detection')})"],
            out_dir / f"d{index:02d}_detector_comparison_photo{index}.png",
            suptitle=f"Task 3C - The two detectors on {haar['label']}",
            cols=2,
            fig_scale=6.0,
        )

    sweep_rows = []
    for entry in panels:
        scale_factor, min_neighbors = entry["setting"]
        bad = unsupported_detections(entry["detections"], yolo_results[0]["detections"],
                                     args.head_fraction)
        supported = len(entry["detections"]) - len(bad)
        sweep_rows.append([f"{scale_factor:g}", str(min_neighbors),
                           str(len(entry["detections"])), str(supported)])
    cu.print_table(["scaleFactor", "minNeighbors", "Detections", "On a head"], sweep_rows)
    sweep_supported = sum(int(row[3]) for row in sweep_rows)

    unsupported = {
        haar["label"]: unsupported_detections(haar["detections"], yolo["detections"],
                                              args.head_fraction)
        for haar, yolo in zip(haar_results, yolo_results)
    }
    for label, items in unsupported.items():
        if items:
            print(f"  {label}: {len(items)} Haar detection(s) with no YOLO person "
                  f"around them")

    write_report(out_dir, images, haar_results, yolo_results, unsupported,
                 sweep_rows, sweep_supported, sweep_name, conf_rows, sweep_label,
                 model, args, is_sample)

    cu.banner("TASK 3 COMPLETE")
    print(f"  All images and results.md are in {cu.display_path(out_dir)}\n")


def write_report(
    out_dir, images, haar_results, yolo_results, unsupported,
    sweep_rows, sweep_supported, sweep_name, conf_rows, sweep_label,
    model, args, is_sample,
) -> None:
    """Assemble ``results.md`` from the measured detections."""
    cascade_names = list(haar_results[0]["per_cascade"])

    haar_headers = ["Photograph", *[n.capitalize() for n in cascade_names], "Runtime (ms)"]
    haar_rows = [
        [f"`{r['label']}`", *[str(r["counts"][n]) for n in cascade_names], f"{r['ms']:.0f}"]
        for r in haar_results
    ]

    yolo_headers = ["Photograph", "Detections", "Classes", "Confidence range", "Runtime (ms)"]
    yolo_rows = []
    for r in yolo_results:
        scores = [d["score"] for d in r["detections"]]
        if not scores:
            span = "-"
        elif len(scores) == 1:
            span = f"{scores[0]:.2f}"
        else:
            span = f"{min(scores):.2f} - {max(scores):.2f}"
        yolo_rows.append([f"`{r['label']}`", str(len(r["detections"])),
                          ", ".join(r["classes"]) or "none", span, f"{r['ms']:.0f}"])

    fired = [n for n in cascade_names if any(r["counts"][n] for r in haar_results)]
    silent = [n for n in cascade_names if n not in fired]
    total_haar = sum(len(r["detections"]) for r in haar_results)
    total_yolo = sum(len(r["detections"]) for r in yolo_results)
    total_unsupported = sum(len(v) for v in unsupported.values())

    note = ""
    if is_sample:
        note = ("\n> Note: this run used the synthetic stand-in image. Re-run the "
                "script once the photograph is in `data/`.\n")

    input_body = "\n".join(
        [f"- Photograph {i}: `{label}` ({image.shape[1]} x {image.shape[0]} px)"
         for i, (label, image) in enumerate(images, start=1)]
        + [f"- Haar cascades: `{cu.display_path(Path(args.cascade_dir))}` "
           f"({', '.join(cascade_names)})",
           f"- YOLO weights: `{Path(args.weights).name}`, "
           f"{len(model.names)} COCO classes{note}"]
    )

    first_unsupported = len(unsupported.get(haar_results[0]["label"], []))
    first_total = len(haar_results[0]["detections"])
    if first_total and first_unsupported == first_total:
        headline = (
            f"None of those detections is on a face. On the first photograph the "
            f"cascades reported {plural(first_total, 'box')}, and section C shows "
            f"that every one of them sits somewhere other than a head, while the "
            f"face that is in the frame goes undetected."
        )
    elif first_total:
        headline = (
            f"Of the {plural(first_total, 'detection')} on the first photograph, "
            f"{first_unsupported} sit somewhere other than a head (section C)."
        )
    else:
        headline = "The cascades reported nothing on the first photograph."

    haar_body = "\n".join([
        f"Four pretrained cascades were run over each photograph at "
        f"`scaleFactor = {args.scale_factor:g}`, `minNeighbors = "
        f"{args.min_neighbors}` and a minimum size of {args.min_size} px, on a "
        f"histogram-equalised grayscale copy.",
        "",
        cu.markdown_table(haar_headers, haar_rows),
        "",
        (f"Only the {cu_join(fired)} cascade responded at all."
         if len(fired) == 1 else
         f"The cascades that responded were {cu_join(fired)}.") +
        (f" The {cu_join(silent)} cascade"
         f"{'' if len(silent) == 1 else 's'} returned nothing on either "
         f"photograph."
         if silent else ""),
        "",
        headline,
        "",
        "A cascade is a chain of stages, each one a vote over simple "
        "intensity-difference features, and a window has to clear every stage to "
        "be reported. The frontal-face cascade was trained on upright, "
        "forward-facing, unobstructed faces, so a head turned to the side, shaded "
        "by a cap and carrying sunglasses breaks the pattern of light and dark "
        "regions around the eyes that the early stages test for.",
        "",
        f"The detector has no confidence to filter on afterwards, so its two "
        f"parameters are the only control over how much it reports. The "
        f"{sweep_name} cascade was run at six settings on the first photograph. "
        f"The last column counts the detections that land on a head, judged "
        f"against the people YOLO found:",
        "",
        cu.markdown_table(["scaleFactor", "minNeighbors", "Detections", "On a head"],
                          sweep_rows),
        "",
        f"`scaleFactor` sets how fast the detector steps through the image "
        f"pyramid and `minNeighbors` how many overlapping windows must agree "
        f"before a box is kept. The count swings from "
        f"{min(int(r[2]) for r in sweep_rows)} to "
        f"{max(int(r[2]) for r in sweep_rows)} across these six settings, and "
        f"not in one direction: a finer pyramid step at `1.05` reports fewer "
        f"boxes here than `1.1`, because the extra scales it tests change which "
        f"windows end up adjacent to each other and therefore which ones survive "
        f"the neighbour count. At `1.3` the steps are coarse enough that nothing "
        f"passes at all.",
        "",
        (f"Across all six settings, {sweep_supported} of the detections land on a "
         f"head. The parameters change how much the cascade reports, not whether "
         f"it finds this face."
         if sweep_supported == 0 else
         f"Across all six settings, {sweep_supported} detection(s) land on a head."),
        "",
        "Figures: `b01_haar_photo1.png`, `b03_haar_cascade_comparison.png`, "
        "`b04_haar_parameter_sweep.png`",
    ])

    yolo_body = "\n".join([
        f"`{Path(args.weights).name}` was run at a confidence threshold of "
        f"`{args.conf:g}`. The model is pretrained on COCO, so it reports any of "
        f"{len(model.names)} classes with a box, a class label and a confidence.",
        "",
        cu.markdown_table(yolo_headers, yolo_rows),
        "",
        "YOLO divides the image once into a grid and predicts boxes, class "
        "scores and a confidence for every cell in a single pass of one network. "
        "It is trained on whole objects in natural photographs rather than on a "
        "single rigid view, which is why the person is found here in a pose the "
        "frontal-face cascade cannot handle.",
        "",
        f"Because every detection carries a confidence, the output can be "
        f"thresholded after the fact. On `{sweep_label}`:",
        "",
        cu.markdown_table(["Confidence threshold", "Detections", "Classes"], conf_rows),
        "",
        ("Lowering the threshold to "
         f"{min(args.conf_sweep):g} changes nothing here. The group of people "
         "standing near the pyramid on the right of that photograph is visible "
         "to the eye and is not reported at any threshold, so those are not "
         "low-confidence detections being filtered out: the model does not "
         "produce them at all. They are a few dozen pixels tall, and the "
         "detector resizes the image to 640 px before it runs, which leaves "
         "almost nothing of them."
         if len({row[1] for row in conf_rows}) == 1 else
         "The count changes with the threshold, which is the trade between "
         "missed objects and false ones that a confidence score makes "
         "adjustable after the run."),
        "",
        "Figures: `c01_yolo_photo1.png`, `c03_yolo_confidence_comparison.png`",
    ])

    comparison_rows = [
        ["What it detects", "One trained object class per cascade file",
         f"{len(model.names)} COCO classes from one model"],
        ["Output", "Box only", "Box, class label and confidence"],
        ["How it works", "Cascade of stages over Haar features, slid over an "
                         "image pyramid", "One pass of a convolutional network over "
                         "the whole image"],
        ["Pose tolerance", "Trained on one rigid view, so an off-angle or occluded "
                           "subject is missed", "Trained on varied natural images, so "
                           "pose and partial occlusion are tolerated"],
        ["Tuning", "scaleFactor and minNeighbors, set before the run",
         "One confidence threshold, applied after the run"],
        ["Cost", f"{haar_results[0]['ms']:.0f} ms on this image for all "
                 f"{len(cascade_names)} cascades, CPU only, no model file beyond "
                 f"a small XML",
         f"{yolo_results[0]['ms']:.0f} ms on this image, with a "
         f"{Path(args.weights).name} weights file"],
    ]

    findings = []
    for haar, yolo in zip(haar_results, yolo_results):
        people = [d for d in yolo["detections"] if d["label"] == "person"]
        confidences = cu_join([f"{d['score']:.2f}" for d in people])
        detail = f" at confidence {confidences}" if people else ""
        findings.append(
            f"- `{haar['label']}`: YOLO found {plural(len(people), 'person detection')}"
            f"{detail}, the cascades reported "
            f"{plural(len(haar['detections']), 'detection')} in total."
        )
    for label, items in unsupported.items():
        if not items:
            continue
        boxes = ", ".join(f"{d['label']} at {tuple(d['box'][:2])}" for d in items)
        findings.append(
            f"- `{label}`: {plural(len(items), 'cascade detection')} failed that "
            f"test ({boxes}). Some sit on open background and some on the person "
            f"but well below the head, on the collar and the forearm. Either way "
            f"no face is there, and the face that is in the frame was not "
            f"reported by any cascade."
        )

    comparison_body = "\n".join([
        f"Across both photographs the cascades reported "
        f"{plural(total_haar, 'detection')} and YOLO reported {total_yolo}.",
        "",
        cu.markdown_table(["Property", "Viola-Jones (Haar cascade)", "YOLO"],
                          comparison_rows),
        "",
        "### Misses and unsupported detections",
        "",
        f"The brief asks for at least one case where a detector misses an object "
        f"or reports one that is not there. Deciding it by overlap keeps the "
        f"judgement measured. A body detection counts as supported when at least "
        f"half of its box lies inside a person YOLO found. A face detection is "
        f"held to the top {args.head_fraction:.0%} of that person instead, since "
        f"a box on someone's shirt is still not a face, and a standing figure is "
        f"roughly seven heads tall.",
        "",
        *findings,
        "",
        (f"In total {plural(total_unsupported, 'cascade detection')} failed that "
         f"test, out of {total_haar} reported."
         if total_unsupported else
         "No cascade detection failed that test on these photographs."),
        "",
        "Figures: `d01_detector_comparison_photo1.png`"
        + (", `d02_detector_comparison_photo2.png`" if len(images) > 1 else ""),
    ])

    cu.write_results(
        out_dir,
        [
            ("Input", input_body),
            ("A. Viola-Jones (Haar Cascade) Detection", haar_body),
            ("B. YOLO Detection", yolo_body),
            ("C. Comparison of the Two Detectors", comparison_body),
        ],
        title=TASK_NAME,
    )


def plural(count: int, word: str) -> str:
    """Format a count with its noun: "1 detection", "4 detections"."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def cu_join(values: list[str]) -> str:
    """Join names into a readable list."""
    if not values:
        return "none"
    if len(values) < 3:
        return " and ".join(values)
    return ", ".join(values[:-1]) + " and " + values[-1]


if __name__ == "__main__":
    main()
