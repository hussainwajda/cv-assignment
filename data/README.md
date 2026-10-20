# Input dataset

Media captured for this assignment. The scripts read from this folder.

| File | Description |
| ---- | ----------- |
| `photo1.jpg` | Original unedited photograph. A scene with visible detail and several touching objects suits the watershed stage of Task 1. |
| `photo2.jpg` | Second photograph of the same object or scene from a slightly different angle or distance, used for SIFT matching in Task 2. |
| `video.mp4` | Own recording, roughly 15-30 s, with a moving person or object against a textured background. The camera is held still for part of the clip. |

Other extensions also work (`.png`, `.jpeg`, `.mov`, `.avi`, ...): each script
searches this folder for any supported image or video when the default name is
missing. A file elsewhere can be passed directly:

```bash
python src/task1_restoration_segmentation.py --image /path/to/photo.jpg
```

## Capture notes

Points that affect the results:

- Even lighting, no heavy shadows across the subject, and no filters or crops.
- The subject fills the frame with background at the border, since Otsu
  thresholding reads the border to decide which side is the object.
- The second photograph keeps the same lighting, so matching reflects the
  viewpoint change rather than an exposure change.
- Part of the video is recorded from a stable surface, because background
  subtraction assumes a mostly static camera, and the background has texture so
  optical flow has features to track.

## Video and Git

A 15-30 second phone video runs to tens of megabytes, and the brief asks for
large raw video on Google Drive rather than in Git, so video files in this
folder are ignored by `.gitignore`. The photographs are small and are
committed. A small video can still be committed explicitly:

```bash
git add -f data/video.mp4
```
