# SmartDoc — Old Document Scanner & Digital Image Processing Educational System

SmartDoc is an interactive, presentation-ready Digital Image Processing (DIP) application that converts old, yellowed, stained, shadowed, and tilted document photographs into clean, high-contrast scanned documents and extracts their text using traditional Tesseract OCR.

---

## 1. Key Highlights & Methodological Scope

- **Traditional DIP Only**: Strictly built on classical digital image processing and computer vision techniques using **Python**, **OpenCV**, **NumPy**, **Tkinter**, **Pillow**, **Matplotlib**, and **Tesseract OCR**.
- **Strictly No Deep Learning / AI**: No LLMs, no VLMs, no neural networks, no PyTorch, no TensorFlow, and no cloud-based recognition APIs.
- **Educational Explanation Panel**: Automatically displays detailed mathematical formulations, variable definitions, OpenCV functions, before-and-after empirical pixel measurements, histogram interpretations, and faculty demonstration scripts.
- **Side-by-Side Histogram Analysis**: Embedded Matplotlib visualization using `cv2.calcHist()` showing real-time intensity evolution from yellowed/shadowed distributions to high-contrast bimodal distributions.

---

## 2. Four Core Features & Processing Workflow

The entire application is streamlined around **ONLY FOUR CORE FEATURES**:

```
[ Upload Document Photo ]
           │
           ▼
[ Feature 1: Document Border Detection ]  ◄── (Automatic Canny + approxPolyDP or Manual Selection)
           │
           ▼
[ Feature 2: Perspective Transformation ] ◄── (3x3 Homography Matrix Warping to Flat Rectangle)
           │
           ▼
[ Feature 3: Old Document Enhancement ]   ◄── (6-Step Pipeline: Norm + Denoise + CLAHE + Adaptive Threshold)
           │
           ▼
[ Feature 4: Text Extraction (OCR) ]      ◄── (Traditional Tesseract OCR --oem 3 --psm 6)
           │
           ▼
[ Output: output/scanned_doc.png & output/output.txt ]
```

### Feature 1: Document Border Detection
- **Objective**: Identify the 4 document corners against the surrounding background.
- **Robust Multi-Strategy Pipeline**:
  1. *Normalized Downscaling*: Analyzes at 800px height to eliminate high-frequency internal text clutter.
  2. *Multi-Candidate Detection*: Generates 4 edge/binary maps (Adaptive Canny with median thresholds, Morphological Gradients, Otsu Page Masking, and Dilation Linking).
  3. *Convex Hull & RDP Simplification*: Bridges small tears or ragged edges and evaluates polygon simplification ($45^\circ \le \text{angle} \le 135^\circ$).
  4. *Fallbacks*: Minimum-area rectangle and safe boundary inset if edges are faint.
- **Interactive 4-Corner Editor**: Clicking **"Adjust / Manual Corners"** opens an interactive dialog where you can:
  - Drag any of the 4 colored corner handles (1:TL Yellow, 2:TR Cyan, 3:BR Magenta, 4:BL Lime) with the mouse for pixel-perfect alignment.
  - Or click "Click 4 Points Fresh" to place corners sequentially.
  - Quick clicking directly on the main window's Left panel also works with exact coordinate translation.

### Feature 2: Perspective Transformation
- **Objective**: Rectify off-axis camera angles and tilt into a flat, top-down rectangular scan.
- **Mathematics**: Solves for the 3×3 perspective transformation (homography) matrix $H$ with 8 degrees of freedom:
  $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- **Dimensions**: Computes true destination width and height using Euclidean distances between ordered corners.
- **OpenCV Functions**: `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`.

### Feature 3: Old Document Enhancement
- **Objective**: Restore aged, yellowed, stained, shadowed, and faded documents in a single button click.
- **Internal 6-Step Pipeline**:
  1. *Grayscale Conversion*: $I_{\text{gray}} = 0.299R + 0.587G + 0.114B$ via `cv2.cvtColor`.
  2. *Background Estimation*: 51×51 Gaussian blur extracts slowly varying paper tone and illumination $B(x, y)$.
  3. *Background Normalization*: Intensity division $I_{\text{norm}}(x, y) = \left[ \frac{I(x, y)}{B(x, y) + 1} \right] \times 255$ cancels uneven lighting and yellow paper cast.
  4. *Noise Reduction*: Gentle 3×3 median filter suppresses salt-and-pepper grain without blurring thin ink strokes.
  5. *Contrast Enhancement*: CLAHE (clip limit 2.0, tile grid 8×8) amplifies faded ink contrast locally.
  6. *Resolution-Adaptive Thresholding*: Gaussian-weighted local thresholding ($T(x, y) = \text{LocalMean}(x, y) - 8$) dynamically scaled to image dimensions to prevent hollowed-out letter strokes.
- **Intermediate Stage Inspection**: Interactive dropdown allows viewing any individual intermediate DIP stage.

### Feature 4: Text Extraction (OCR)
- **Objective**: Extract readable text and save it to `output/output.txt`.
- **High-Accuracy Traditional OCR Pipeline**:
  - *Resolution Normalization*: Bicubic upscaling ensures character x-height $\ge 30$ px (optimal 300 DPI for Tesseract).
  - *DIP Preprocessing*: Illumination normalization + CLAHE + Otsu binarization produces solid, dark character strokes.
  - *Despeckling*: Connected-component noise suppression eliminates isolated 1-2 px paper grain specks, preventing random punctuation hallucinations.
  - *Page Segmentation Modes (PSM)*: Selectable layout modes (PSM 6 for Uniform Paragraphs, PSM 3 for Automatic Multi-column/Header Layout, PSM 4 for Single Column). Intelligent fallback ensures maximum character yield.

---

## 3. Project File Structure

```
SmartDoc/
├── main.py                   # Redesigned GUI, 4-step workflow, Matplotlib histograms, history
├── image_processing.py       # DIP algorithms (normalization, CLAHE, adaptive threshold, calcHist)
├── document_scanner.py       # Canny border detection, manual corners, perspective warping
├── algorithm_explanations.py # Mathematical formulas, measured data analysis, faculty scripts
├── ocr.py                    # Tesseract OCR configuration, text extraction, file persistence
├── requirements.txt          # Python package dependencies (including Matplotlib)
├── README.md                 # Project documentation and presentation guide
├── input_images/             # Folder for test document photos
└── output/                   # Generated scans (scanned_doc.png) and text (output.txt)
```

---

## 4. Setup and Installation

### Step 1: Install Python
Ensure Python 3.10 or newer is installed. Make sure **"Add python.exe to PATH"** is checked during installation.

### Step 2: Install Tesseract OCR (Windows)
1. Download the Windows installer from UB Mannheim:
   https://github.com/UB-Mannheim/tesseract/wiki
2. Install to the default directory:
   `C:\Program Files\Tesseract-OCR\tesseract.exe`
   *(If installed elsewhere, verify or update `TESSERACT_PATH` in `ocr.py`).*

### Step 3: Open in VS Code & Install Dependencies
1. Open VS Code and open the `SmartDoc` directory.
2. In the VS Code terminal, install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

---

## 5. Demonstration Guide for College Faculty

1. **Upload an Image**: Click **"Upload Document Photo"** and select a tilted, shadowed old document photo. Point out the original image dimensions and how the original histogram is bunched in the middle greys.
2. **Feature 1 (Border Detection)**: Click **"Detect Border"**. Show the detected green perimeter and red corner points. Explain image gradients ($G = \sqrt{G_x^2 + G_y^2}$) and RDP polygon simplification in the explanation panel. Demonstrate that "Manual Corners" can also be used.
3. **Feature 2 (Perspective Warp)**: Click **"Correct Perspective"**. Show the straightened rectangular scan. Explain the 3×3 homography matrix $H$ and projective transformation. Show how the histogram now reflects only the document area.
4. **Feature 3 (Enhance Old Document)**: Click **"Enhance Old Document"**. Show the cleaned, high-contrast document. Point out the live histogram transformation into a sharp, bimodal distribution (peaks at 0 and 255). Use the stage dropdown to show intermediate stages (e.g. Estimated Background, Background Normalization, CLAHE).
5. **Feature 4 (Extract Text)**: Click **"Extract Text (OCR)"**. Show the extracted text dialog and verify that it has been saved to `output/output.txt`. Explain why traditional DIP preprocessing is essential for OCR character segmentation.
6. **Save Scanned Document**: Click **"Save Processed Image"** to save `output/scanned_doc.png`.
