# Face Detector with NMS

Files added to provide face detection with Non-Maximum Suppression (NMS) for your project.

## Files Added

1. **fix_images.py** - Configuration to tolerate truncated/corrupted images
   - Imported automatically by the other scripts

2. **detector.py** - Implements:
   - `FaceDetector` class: sliding-window detector at multiple scales
   - `non_max_suppression` function: removes redundant detections using IoU

3. **detect_faces.py** - Main script to:
   - Process images from a test directory
   - Apply detection + NMS
   - Save images with drawn bounding boxes
   - Supports formats: .png, .jpg, .jpeg, .bmp, .gif, .pgm

4. **_fixed.py files** - Fixed versions of some original scripts:
   - `load_data_fixed.py` - copy of `load_data.py` that ensures images are handled robustly
   - `test_fixed.py` - copy of `test.py` that includes the fix for truncated images

## IMPORTANT: Fix for Truncated Images

If you see the error "OSError: image file is truncated":

**Quick fix:** Use the `_fixed.py` scripts which already include the correction::
```bash
python load_data_fixed.py --epochs 10
python test_fixed.py
```

**OR** add this import at the top of your original scripts (`load_data.py`, `test.py`):
```python
import fix_images  # place after other imports
```

## How to Use

### 1. Make sure you have a trained model

**Using the fixed loader (recommended):**
```bash
python load_data_fixed.py --epochs 10
```

**OR** use the original loader (if you added `import fix_images`):
```bash
python load_data.py --epochs 10
```

This will produce the file `best_model.pth`.

### 2. Run detection

**For 36x36 images (project .pgm files):**
```bash
python detect_faces.py --window-size 36 --stride 4 --confidence 0.5
```

**For larger images (default settings):**
```bash
python detect_faces.py
```

### 3. Optional parameters
```bash
python detect_faces.py \
    --test-dir ./test_images \
    --output-dir ./detection_results \
    --model-path best_model.pth \
    --confidence 0.5 \
    --iou 0.3 \
    --window-size 36 \
    --stride 4
```

## Parameters Explained

- `--test-dir`: Directory with images to run detection on (default: `./test_images`)
- `--output-dir`: Where to save images with detections (default: `./detection_results`)
- `--model-path`: Path to the trained model (default: `best_model.pth`)
- `--confidence`: Confidence threshold (0-1). Higher values → fewer, more precise detections
- `--iou`: IoU threshold for NMS (0-1). Lower values → more aggressive duplicate removal
- `--window-size`: Detection window size in pixels (IMPORTANT: should match approximate face size in your images)
- `--stride`: Sliding window step (smaller → more detections, slower)

## WARNING: Window Size

The `--window-size` parameter is CRITICAL and should match the approximate face size in your images:

- **36x36 images** (project .pgm files): use `--window-size 36`
- **Larger images with small faces**: use `--window-size 48` or `--window-size 64`
- **Large images with large faces**: use `--window-size 128` or larger

If `--window-size` is larger than the image, nothing will be detected.

## How It Works

### Sliding Window
The detector scans each image at multiple scales (0.5x, 0.75x, 1.0x, 1.25x, 1.5x) with a fixed window size:
- This allows detection of faces at different sizes
- Each window is classified by the neural network
- Windows with confidence > threshold are saved as detections

### Non-Maximum Suppression (NMS)
Removes redundant bounding boxes:
1. Sort all detections by confidence (highest first)
2. For each detection:
   - Keep the highest-confidence one
   - Remove all others that have IoU > threshold with it
3. Result: only the best non-overlapping detections remain

### IoU (Intersection over Union)
Metric that measures overlap between two bounding boxes:
- IoU = Area of Intersection / Area of Union
- IoU = 0: boxes do not overlap
- IoU = 1: boxes are identical

## Expected Output

The script will:
1. Process each image in `test_dir`
2. Detect faces using sliding window
3. Apply NMS to remove duplicates
4. Save images with green boxes and confidence labels
5. Print a summary

**Example output:**
```
Loading model from best_model.pth...
Model loaded successfully!

Processing: face_00001.pgm
  Found 3 raw detections
  After NMS: 1 detections
  Processed in 0.12s
Saved detection result to ./detection_results/detected_face_00001.pgm

...

Summary:
  Total images processed: 7628
  Total faces detected: 2740
  Average detections per image: 0.36
  Results saved to: ./detection_results
```

## Fine Tuning

### If you have **many false positives**:
```bash
# Increase minimum confidence
python detect_faces.py --window-size 36 --confidence 0.7

# More aggressive NMS
python detect_faces.py --window-size 36 --iou 0.2

# Both
python detect_faces.py --window-size 36 --confidence 0.7 --iou 0.2
```

### If **few or no faces are detected**:
```bash
# Lower confidence
python detect_faces.py --window-size 36 --confidence 0.3

# Smaller stride (more detections)
python detect_faces.py --window-size 36 --stride 2 --confidence 0.4

# Check window size
file test_images/1/* | head -1  # See the actual image size
python detect_faces.py --window-size [CORRECT_SIZE]
```

### If detection is **very slow**:
```bash
# Increase stride
python detect_faces.py --window-size 36 --stride 8

# Reduce scales (edit detector.py, line: scales=[0.5, 0.75, 1.0, 1.25, 1.5])
# Change to: scales=[1.0]
```

## Inspecting Images

To see the size of your images:
```bash
file test_images/0/* | head -3
file test_images/1/* | head -3
```

Adjust `--window-size` accordingly.

## Directory Structure

```
project/
├── fix_images.py              ⬅️ REQUIRED
├── net.py                     (your original file)
├── load_data.py               (your original file)
├── test.py                    (your original file)
├── detector.py                ⬅️ NEW
├── detect_faces.py            ⬅️ NEW
├── load_data_fixed.py         ⬅️ NEW (optional)
├── test_fixed.py              ⬅️ NEW (optional)
├── best_model.pth             (generated by training)
├── train_images/
│   ├── 0/                     (or noface/)
│   └── 1/                     (or face/)
├── test_images/
│   ├── 0/
│   └── 1/
└── detection_results/         (generated automatically)
    ├── detected_image1.pgm
    ├── detected_image2.pgm
    └── ...
```

## Integration with Existing Code

The new files work with:
- `net.py`: your network architecture
- `load_data.py`: used to train the model
- `test.py`: can still be used for evaluation

You don't need to modify the original files (unless you want to add `import fix_images`).

## Quick Commands

```bash
# Train model
python load_data_fixed.py --epochs 10

# Test accuracy
python test_fixed.py

# Detect faces (36x36 images)
python detect_faces.py --window-size 36 --stride 4 --confidence 0.5

# Detect with higher sensitivity
python detect_faces.py --window-size 36 --stride 4 --confidence 0.3

# Detect with higher precision
python detect_faces.py --window-size 36 --stride 4 --confidence 0.7

# Show full help
python detect_faces.py --help
```

## Troubleshooting

**Error: "No module named 'fix_images'"**
- Make sure `fix_images.py` is in the same folder

**Error: "OSError: image file is truncated"**
- Use `load_data_fixed.py` or add `import fix_images`

**Error: "Total faces detected: 0"**
- Check image size with `file test_images/1/* | head -1`
- Adjust `--window-size` to the correct size
- Try `--confidence 0.3` for a lower threshold

**Detection is too slow**
- Increase `--stride` (e.g., 8 or 16)
- Consider using fewer scales (edit `detector.py`)

**No output images**
- Check the `detection_results/` folder
- Only images WITH detections are saved

### Participants:
- Ana Luisa Girio
- Julia Guimarães Simão
- Leonardo
- Maria Isabel
- Nicolas
