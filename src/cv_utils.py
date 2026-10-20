"""
Shared helpers for the CCA-3 Computer Vision assignment.

Every task script imports this module, so image loading, output saving, figure
generation and the markdown result tables stay consistent across the four
tasks.

The synthetic stand-in generators at the bottom exist only so the pipeline can
be run before the real photograph and video are placed in ``data/``. The
submitted results use the real media.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe: write figures to disk, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

#: Default names the task scripts look for inside ``cv-assignment/data``.
DEFAULT_PHOTO = DATA_DIR / "photo1.jpg"
DEFAULT_PHOTO_2 = DATA_DIR / "photo2.jpg"
DEFAULT_VIDEO = DATA_DIR / "video.mp4"

#: Extensions tried when the default file name is not present.
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: Path) -> str:
    """Format ``path`` for the console, relative to the project when possible.

    ``relative_to`` raises for a path outside the project, which an ``--outdir``
    elsewhere legitimately is, so fall back to the absolute path.
    """
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def banner(title: str) -> None:
    """Print a section header, matching the sections of the report."""
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR (or grayscale) image to RGB for matplotlib."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 version of ``image``."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale ``image`` so its width is at most ``max_width`` (aspect kept).

    A phone photograph is often 4000 px wide. Working at around 1000 px keeps
    the filters, SIFT and watershed fast at the same conclusions.
    """
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = max_width / float(image.shape[1])
    new_size = (max_width, int(round(image.shape[0] * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Image / video I/O
# --------------------------------------------------------------------------- #

def read_image(path: Path | str) -> np.ndarray:
    """Read an image as BGR uint8."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image '{path}'. Check that the file exists and is a "
            f"supported format ({', '.join(PHOTO_EXTENSIONS)})."
        )
    return image


def save_image(out_dir: Path, name: str, image: np.ndarray) -> Path:
    """Write ``image`` into ``out_dir`` as a PNG and return the path."""
    out_dir = ensure_dir(out_dir)
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        name = f"{name}.png"
    path = out_dir / name
    cv2.imwrite(str(path), image)
    print(f"  saved  {display_path(path)}")
    return path


def save_grid(
    images: list[np.ndarray],
    titles: list[str],
    out_path: Path,
    suptitle: str = "",
    cols: int = 3,
    fig_scale: float = 4.2,
) -> Path:
    """Save a labelled grid of images as a single comparison figure.

    One PNG per task section, for the comparison deliverables in the report.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    count = len(images)
    cols = max(1, min(cols, count))
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * fig_scale, rows * fig_scale * 0.85), squeeze=False
    )
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index >= count:
            continue
        image = images[index]
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(to_rgb(image))
        ax.set_title(titles[index], fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {display_path(out_path)}")
    return out_path


def open_video(path: Path | str) -> cv2.VideoCapture:
    """Open a video file for reading."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video '{path}'. Check the path and that the codec is "
            f"supported ({', '.join(VIDEO_EXTENSIONS)}); re-encoding to .mp4 (H.264) "
            f"usually fixes it."
        )
    return cap


def video_info(cap: cv2.VideoCapture) -> dict:
    """Return basic properties of an opened video."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 2),
        "frame_count": frames,
        "duration_s": round(frames / fps, 2) if fps > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Markdown result tables
# --------------------------------------------------------------------------- #

def markdown_table(headers: list[str], rows: list[list]) -> str:
    """Render a GitHub-flavoured markdown table from headers and rows."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in cells)) if cells else len(headers[i])
        for i in range(len(headers))
    ]
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in cells
    ]
    return "\n".join([head, sep, *body])


def print_table(headers: list[str], rows: list[list]) -> None:
    """Print a markdown table to the console."""
    print(markdown_table(headers, rows))


def write_results(out_dir: Path, sections: list[tuple[str, str]], title: str) -> Path:
    """Write ``outputs/<task>/results.md`` from ``(heading, body)`` sections."""
    out_dir = ensure_dir(out_dir)
    path = out_dir / "results.md"
    parts = [f"# {title}", ""]
    for heading, body in sections:
        parts.extend([f"## {heading}", "", body.strip(), ""])
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  saved  {display_path(path)}")
    return path


# --------------------------------------------------------------------------- #
# Synthetic stand-in media (development only)
# --------------------------------------------------------------------------- #

_SAMPLE_WARNING = (
    "!! No input file found, falling back to a synthetic sample image/video.\n"
    "!! For testing the pipeline only. Place the photograph and video in\n"
    "!! data/ and re-run before using the results."
)


def make_sample_photo(width: int = 960, height: int = 640, seed: int = 7) -> np.ndarray:
    """Build a textured synthetic scene with multiple touching objects.

    The scene holds overlapping circles for watershed to split, straight edges
    for Canny, and text for SIFT and HOG to find keypoints and gradients in.
    """
    rng = np.random.default_rng(seed)

    # Soft diagonal gradient background, like a lit tabletop. A shallow
    # gradient: a strong one dominates Otsu, which then splits the background
    # instead of separating the objects from it.
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    base = 0.74 + 0.10 * yy + 0.06 * xx
    image = np.dstack([base * 205, base * 212, base * 220]).astype(np.float32)
    image += rng.normal(0.0, 5.0, (height, width, 1)).astype(np.float32)  # paper grain
    image = np.clip(image, 0, 255).astype(np.uint8)

    sx, sy = width / 960.0, height / 640.0

    def pt(x: int, y: int) -> tuple[int, int]:
        return int(x * sx), int(y * sy)

    # Three touching discs, the watershed test case. Fill colours well below
    # the background brightness give Otsu an unambiguous split.
    cv2.circle(image, pt(280, 415), int(88 * sx), (170, 70, 40), -1)
    cv2.circle(image, pt(442, 418), int(84 * sx), (60, 140, 60), -1)
    cv2.circle(image, pt(362, 280), int(78 * sx), (30, 90, 190), -1)

    # Rectangular objects: straight edges for Canny, gradients for HOG.
    cv2.rectangle(image, pt(640, 236), pt(858, 424), (150, 50, 140), -1)
    cv2.rectangle(image, pt(676, 272), pt(822, 388), (238, 236, 232), 3)  # hole for closing
    cv2.rectangle(image, pt(120, 118), pt(300, 200), (46, 48, 52), -1)

    # A triangle and some text for additional corner structure.
    tri = np.array([pt(700, 520), pt(820, 520), pt(760, 430)], dtype=np.int32)
    cv2.fillPoly(image, [tri], (170, 140, 40))
    cv2.putText(image, "CCA-3 SAMPLE", pt(120, 560), cv2.FONT_HERSHEY_SIMPLEX,
                1.1 * sx, (40, 40, 45), 3, cv2.LINE_AA)

    return cv2.GaussianBlur(image, (3, 3), 0)


def make_sample_photo_pair(width: int = 960, height: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Return two synthetic views of the same scene, for the SIFT matching task.

    The second view is the first one rotated, slightly scaled, shifted and
    brightened, which approximates stepping sideways and re-shooting.
    """
    first = make_sample_photo(width, height, seed=7)
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle=11.0, scale=0.92)
    matrix[0, 2] += 18.0
    matrix[1, 2] -= 12.0
    second = cv2.warpAffine(first, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    second = cv2.convertScaleAbs(second, alpha=1.06, beta=8)  # different exposure
    return first, second


def make_sample_video(
    path: Path,
    seconds: int = 8,
    fps: int = 20,
    size: tuple[int, int] = (640, 360),
) -> Path:
    """Write a synthetic video: a static scene with two objects moving across it.

    One object translates left to right at constant speed and one bounces
    vertically, giving frame differencing, background subtraction and optical
    flow consistent motion to track.
    """
    path = Path(path)
    ensure_dir(path.parent)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for '{path}'.")

    background = cv2.resize(make_sample_photo(width * 2, height * 2, seed=3), size,
                            interpolation=cv2.INTER_AREA)
    total = seconds * fps
    for index in range(total):
        frame = background.copy()
        progress = index / max(1, total - 1)

        # Walking "person": an upright box plus a head.
        x = int(40 + progress * (width - 160))
        y = height - 150
        cv2.rectangle(frame, (x, y), (x + 60, y + 110), (38, 42, 180), -1)
        cv2.circle(frame, (x + 30, y - 18), 22, (60, 70, 200), -1)

        # Bouncing ball.
        bx = int(width * 0.25 + progress * width * 0.5)
        by = int(90 + 70 * abs(np.sin(progress * np.pi * 3)))
        cv2.circle(frame, (bx, by), 26, (40, 200, 230), -1)

        # Mild handheld shake for the last third, leaving the rest stable.
        if progress > 0.66:
            shift = np.float32([[1, 0, np.sin(index * 0.7) * 2.5],
                                [0, 1, np.cos(index * 0.5) * 2.0]])
            frame = cv2.warpAffine(frame, shift, size, borderMode=cv2.BORDER_REPLICATE)

        writer.write(frame)
    writer.release()
    return path


# --------------------------------------------------------------------------- #
# Input resolution (real media first, synthetic fallback second)
# --------------------------------------------------------------------------- #

def _first_existing(directory: Path, extensions: tuple[str, ...]) -> Path | None:
    """Return the first file in ``directory`` with one of ``extensions``."""
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in extensions:
            return candidate
    return None


def resolve_photo(
    explicit: str | None,
    default: Path = DEFAULT_PHOTO,
    allow_sample: bool = True,
    sample_seed: int = 7,
) -> tuple[np.ndarray, str, bool]:
    """Resolve the photograph to work on.

    Order of preference: the ``--image`` argument, the default name in
    ``data/``, any image in ``data/``, then optionally a synthetic stand-in.
    Returns ``(image, label, is_sample)``.
    """
    if explicit:
        return read_image(explicit), str(explicit), False
    if default.exists():
        return read_image(default), str(default), False

    found = _first_existing(DATA_DIR, PHOTO_EXTENSIONS)
    if found is not None:
        return read_image(found), str(found), False

    if not allow_sample:
        raise FileNotFoundError(
            f"No photograph found. Place a photo at '{default}' or pass "
            f"--image /path/to/photo.jpg"
        )
    print(_SAMPLE_WARNING)
    return make_sample_photo(seed=sample_seed), "synthetic sample photo", True


def resolve_video(
    explicit: str | None,
    default: Path = DEFAULT_VIDEO,
    allow_sample: bool = True,
) -> tuple[Path, bool]:
    """Resolve the video to work on. Returns ``(path, is_sample)``."""
    if explicit:
        return Path(explicit), False
    if default.exists():
        return default, False

    found = _first_existing(DATA_DIR, VIDEO_EXTENSIONS)
    if found is not None:
        return found, False

    if not allow_sample:
        raise FileNotFoundError(
            f"No video found. Place a recording at '{default}' or pass "
            f"--video /path/to/video.mp4"
        )
    print(_SAMPLE_WARNING)
    sample = ensure_dir(OUTPUT_DIR / "_sample") / "sample_video.mp4"
    if not sample.exists():
        make_sample_video(sample)
    return sample, True
