"""
algorithm_explanations.py
-------------------------
Educational explanations, mathematical formulations, OpenCV mappings,
pixel-level analyses, and faculty presentation guides for SmartDoc's
FOUR CORE FEATURES:

  1. Document Border Detection
  2. Perspective Transformation
  3. Old Document Enhancement
  4. Text Extraction (OCR)

Every explanation strictly separates actual measured image statistics
from illustrative numerical examples.
"""

import numpy as np


FEATURE_BORDER = "Document Border Detection"
FEATURE_PERSPECTIVE = "Perspective Transformation"
FEATURE_ENHANCE = "Old Document Enhancement"
FEATURE_OCR = "Text Extraction (OCR)"


EXPLANATIONS = {

    # =========================================================================
    # FEATURE 1: DOCUMENT BORDER DETECTION
    # =========================================================================
    FEATURE_BORDER: {
        "name": "Feature 1: Document Border Detection",
        "objective": (
            "Locate the physical boundaries and four corner coordinates of a document "
            "photographed against an arbitrary background surface using classical spatial gradients "
            "and contour approximation."
        ),
        "formula": (
            "1. Horizontal and Vertical Gradients (Sobel kernels):\n"
            "   Gx = dI / dx = I * Kx,    Gy = dI / dy = I * Ky\n\n"
            "2. Overall Edge Gradient Magnitude and Direction:\n"
            "   G = sqrt(Gx^2 + Gy^2)\n"
            "   theta = arctan(Gy / Gx)\n\n"
            "3. Contour Polygonal Approximation (Ramer-Douglas-Peucker):\n"
            "   epsilon = 0.02 * Perimeter\n"
            "   d_max <= epsilon"
        ),
        "variables": [
            ("Gx", "Horizontal intensity gradient; rate of brightness change along the x-axis."),
            ("Gy", "Vertical intensity gradient; rate of brightness change along the y-axis."),
            ("G", "Overall edge strength (gradient magnitude) at pixel (x, y)."),
            ("theta", "Spatial orientation angle of the intensity gradient vector."),
            ("epsilon", "Approximation accuracy tolerance parameter for contour simplification."),
            ("Perimeter", "Closed arc length of the contour curve, calculated via cv2.arcLength."),
            ("d_max", "Maximum perpendicular distance from curve points to the approximating line segment."),
        ],
        "opencv_functions": [
            ("cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
             "Converts 3-channel color image to single-channel intensity."),
            ("cv2.GaussianBlur(gray, (5, 5), 0)",
             "Applies 5x5 Gaussian kernel to suppress high-frequency sensor noise before gradient calculation."),
            ("cv2.Canny(blurred, threshold1=50, threshold2=150)",
             "Performs non-maximum suppression along gradient direction and hysteresis edge tracking."),
            ("cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)",
             "Traces closed boundary curves of connected edge components."),
            ("cv2.approxPolyDP(contour, epsilon, closed=True)",
             "Applies RDP algorithm to approximate a detailed curve into a compact polygon."),
            ("cv2.polylines() & cv2.circle()",
             "Draws the green document perimeter boundary and red corner markers."),
        ],
        "steps": [
            "1. Grayscale Conversion: Transforms BGR color channels into single-channel luminance.",
            "2. Gaussian Smoothing: Applies a 5x5 Gaussian blur to remove surface texture noise.",
            "3. Canny Edge Detection: Computes partial derivatives Gx and Gy, locates gradient peaks via "
            "non-maximum suppression, and links strong edges using hysteresis thresholds (50, 150).",
            "4. Contour Retrieval: Extracts closed vector boundaries from binary edge maps.",
            "5. Polygonal Approximation: Simplifies candidate contours to polygons using cv2.approxPolyDP.",
            "6. Quad Identification: Selects the largest convex polygon having exactly 4 vertices and "
            "covering at least 8% of the total photograph area.",
            "7. Corner Ordering: Sorts corners into [Top-Left, Top-Right, Bottom-Right, Bottom-Left].",
        ],
        "illustrative_example": (
            "Suppose adjacent horizontal pixels have intensities I(x-1, y) = 40 and I(x+1, y) = 160,\n"
            "and adjacent vertical pixels have intensities I(x, y-1) = 60 and I(x, y+1) = 180.\n"
            "  Gx = (160 - 40) / 2 = 60\n"
            "  Gy = (180 - 60) / 2 = 60\n"
            "  Edge Strength G = sqrt(60^2 + 60^2) = sqrt(3600 + 3600) = sqrt(7200) ≈ 84.85\n"
            "Because G = 84.85 exceeds Canny's lower threshold (50), this pixel is classified as a valid edge candidate."
        ),
        "histogram_interpretation": (
            "Border detection itself does not alter the pixel distribution of the original image. "
            "However, the Canny edge detector operates on gradient differences: in regions where the "
            "document paper meets the dark table background, the histogram of gradient magnitudes shows "
            "pronounced high-value spikes corresponding to high-contrast document edges."
        ),
        "why_used": (
            "Accurately locating the page perimeter in an angled photograph is the prerequisite for "
            "perspective rectification. Automating this step eliminates the need for manual cropping."
        ),
        "limitations": [
            "Requires adequate visual contrast between the document paper and the underlying background surface.",
            "Severely torn, heavily folded, or non-quadrilateral paper shapes may fail automatic 4-corner approximation.",
            "Fallback Provided: Manual 4-corner interactive point selection allows users to click the corners directly.",
        ],
        "faculty_demo": (
            "Faculty Explanation:\n"
            "\"In Feature 1, we isolate the document boundary without neural networks using classical edge analysis. "
            "We first compute spatial intensity gradients G = sqrt(Gx^2 + Gy^2) using the Canny operator. "
            "We then trace closed boundaries with cv2.findContours and simplify curves into polygons using the "
            "Ramer-Douglas-Peucker algorithm (cv2.approxPolyDP). By sorting by enclosed area, the largest four-vertex "
            "convex polygon represents the physical page corners. If the boundary is indistinct, the user can manually "
            "click the 4 corners.\""
        ),
    },

    # =========================================================================
    # FEATURE 2: PERSPECTIVE TRANSFORMATION
    # =========================================================================
    FEATURE_PERSPECTIVE: {
        "name": "Feature 2: Perspective Transformation",
        "objective": (
            "Rectify geometric perspective distortion caused by off-axis camera angles, mapping "
            "a tilted quadrilateral document into a true top-down flat rectangle."
        ),
        "formula": (
            "Projective Transformation in Homogeneous Coordinates:\n\n"
            "  [ x' ]       [ h11  h12  h13 ]   [ x ]\n"
            "  [ y' ]   ~   [ h21  h22  h23 ] * [ y ]\n"
            "  [ 1  ]       [ h31  h32  1   ]   [ 1 ]\n\n"
            "Cartesian Destination Coordinates:\n"
            "  x' = (h11*x + h12*y + h13) / (h31*x + h32*y + 1)\n"
            "  y' = (h21*x + h22*y + h23) / (h31*x + h32*y + 1)\n\n"
            "Destination Rectangle Dimensions (Euclidean Distances):\n"
            "  Width  = max( ||TR - TL||, ||BR - BL|| )\n"
            "  Height = max( ||BL - TL||, ||BR - TR|| )"
        ),
        "variables": [
            ("(x, y)", "Source coordinates of a pixel in the angled/tilted photograph."),
            ("(x', y')", "Destination coordinates of the pixel in the flat, rectified document image."),
            ("H", "3x3 Perspective Transformation (Homography) matrix having 8 degrees of freedom."),
            ("h11, h12, h21, h22", "Affine rotation, scaling, and shear components."),
            ("h13, h23", "Spatial translation components along x and y axes."),
            ("h31, h32", "Perspective warp parameters representing camera tilt relative to the page plane."),
            ("TL, TR, BR, BL", "Ordered corner points: Top-Left, Top-Right, Bottom-Right, Bottom-Left."),
        ],
        "opencv_functions": [
            ("cv2.getPerspectiveTransform(src_points, dst_points)",
             "Constructs and solves an 8-equation linear system to determine the 3x3 homography matrix H."),
            ("cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_LINEAR)",
             "Applies backward geometric mapping with bilinear interpolation to synthesize the rectified image."),
        ],
        "steps": [
            "1. Corner Ordering: Ensures the 4 corners follow [TL, TR, BR, BL] sequence.",
            "2. Metric Computation: Uses Euclidean norms to determine target rectangular width and height.",
            "3. Coordinate Mapping: Sets destination target coordinates (0,0), (W-1,0), (W-1,H-1), (0,H-1).",
            "4. Matrix Calculation: cv2.getPerspectiveTransform solves for the unique 3x3 homography matrix H.",
            "5. Image Warping: cv2.warpPerspective resamples image pixels into the flat rectangle.",
        ],
        "illustrative_example": (
            "Assume a tilted document with corners at TL(120, 80), TR(600, 100), BR(580, 720), BL(100, 700).\n"
            "  Top Edge Width    = sqrt((600 - 120)^2 + (100 - 80)^2) = sqrt(480^2 + 20^2) ≈ 480.4 px\n"
            "  Bottom Edge Width = sqrt((580 - 100)^2 + (720 - 700)^2) = sqrt(480^2 + 20^2) ≈ 480.4 px\n"
            "  Destination Width = int(max(480.4, 480.4)) = 480 px\n"
            "Target corner (120, 80) is mathematically transformed to (0, 0), flattening the page."
        ),
        "histogram_interpretation": (
            "Before perspective correction, the image histogram includes the surrounding table, shadows, "
            "and background objects. After perspective transformation, the canvas is cropped tightly to "
            "the document boundaries: background pixels are eliminated, so the histogram reflects solely "
            "the paper fibers, aged yellow background, and ink strokes."
        ),
        "why_used": (
            "Perspective distortion changes text aspect ratio, causes keystoning, and slants horizontal "
            "baselines. Straightening the page into an upright rectangle is crucial for OCR character segmentation."
        ),
        "limitations": [
            "Severe perspective angles (>60 degrees) compress pixels near the far edge, causing minor interpolation blur.",
            "Requires 4 accurately identified corners; inaccurate corners cause skewed or stretched rectangles.",
        ],
        "faculty_demo": (
            "Faculty Explanation:\n"
            "\"Feature 2 implements projective geometry using a 3x3 homography matrix H. "
            "Because parallel lines converge under perspective projection, an affine transform is insufficient. "
            "Using 4 point correspondences between the detected quadrilateral and our calculated destination rectangle, "
            "we solve an 8-parameter linear system using cv2.getPerspectiveTransform. "
            "Finally, cv2.warpPerspective interpolates every pixel to produce a true rectangular bird's-eye view.\""
        ),
    },

    # =========================================================================
    # FEATURE 3: OLD DOCUMENT ENHANCEMENT
    # =========================================================================
    FEATURE_ENHANCE: {
        "name": "Feature 3: Old Document Enhancement",
        "objective": (
            "Restore damaged, yellowed, shadowed, and stained historical documents into clean, "
            "high-contrast scanned documents using a principled sequence of six traditional DIP algorithms."
        ),
        "formula": (
            "Step 1: Grayscale Conversion (Luminance Weights):\n"
            "   I_gray(x, y) = 0.299*R + 0.587*G + 0.114*B\n\n"
            "Step 2: Background Illumination Estimation (Low-Pass Filter):\n"
            "   B(x, y) = GaussianBlur(I_gray, (51, 51), sigma=0)\n\n"
            "Step 3: Background Normalization (Uneven Illumination Compensation):\n"
            "   I_norm(x, y) = [ I_gray(x, y) / (B(x, y) + epsilon) ] * K\n"
            "   (where epsilon = 1.0, K = 255.0)\n\n"
            "Step 4: Non-Linear Noise Filtering (Median Filter):\n"
            "   I_denoised(x, y) = median{ I_norm(x+i, y+j) | (i,j) in 3x3 window }\n\n"
            "Step 5: Local Contrast Enhancement (CLAHE):\n"
            "   P(k) = n_k / N, clipped at clip_limit beta = 2.0 per 8x8 contextual tile\n\n"
            "Step 6: Adaptive Gaussian Thresholding (Local Binarization):\n"
            "   T(x, y) = LocalMean_Gaussian(x, y) - C\n"
            "   I_final(x, y) = 255 if I_denoised(x, y) > T(x, y) else 0  (C = 10, block = 15)"
        ),
        "variables": [
            ("I_gray(x, y)", "Single-channel grayscale intensity derived from weighted RGB channels."),
            ("0.299, 0.587, 0.114", "Perceptual luminance weights matching human photopic spectral sensitivity."),
            ("B(x, y)", "Estimated low-frequency paper background and shadow intensity profile."),
            ("epsilon", "Small positive constant (1.0) preventing division by zero."),
            ("K", "Scaling constant (255) restoring the normalized ratio to standard 8-bit dynamic range."),
            ("beta (Clip Limit)", "CLAHE threshold (2.0) that clips histogram peaks to prevent noise amplification."),
            ("LocalMean(x, y)", "Gaussian-weighted average intensity over a 15x15 local neighborhood."),
            ("C", "Tuning constant (10) subtracted from local mean to ensure dark ink falls below threshold."),
            ("T(x, y)", "Dynamically computed local binarization threshold for pixel (x, y)."),
        ],
        "opencv_functions": [
            ("cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
             "Converts BGR to grayscale using perceptual luminance weights."),
            ("cv2.GaussianBlur(gray, (51, 51), 0)",
             "Applies large spatial kernel to extract slowly-varying illumination and paper tone."),
            ("cv2.divide(gray, background, scale=255)",
             "Performs pixel-wise division to cancel out uneven shadows and yellow cast."),
            ("cv2.medianBlur(normalized, 3)",
             "Suppresses salt-and-pepper noise while preserving thin ink stroke boundaries."),
            ("cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)",
             "Amplifies local text-to-paper contrast across 8x8 grid tiles."),
            ("cv2.adaptiveThreshold(..., 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 10)",
             "Binarizes the document into pure black ink (0) and pure white background (255)."),
        ],
        "steps": [
            "Step 1: Grayscale Conversion: Removes color channels so the pipeline works on luminance.",
            "Step 2: Background Estimation: A large 51x51 Gaussian blur averages away fine text, leaving only the shadow/paper profile.",
            "Step 3: Background Normalization: Dividing image intensity by the background eliminates shadows and yellow cast.",
            "Step 4: Noise Reduction: A 3x3 median filter eliminates sensor speckles without blurring stroke edges.",
            "Step 5: Contrast Enhancement: CLAHE enhances faint, faded characters while preventing noise spikes.",
            "Step 6: Adaptive Thresholding: Local Gaussian-weighted binarization yields clean black text on white paper.",
        ],
        "illustrative_example": (
            "Background Normalization Numerical Example:\n"
            "  Case 1 (Bright Area):  Ink pixel I = 60,  Paper background B = 220\n"
            "    I_norm = (60 / (220 + 1)) * 255 = (60 / 221) * 255 ≈ 69 (Dark ink)\n"
            "  Case 2 (Deep Shadow):  Ink pixel I = 20,  Paper background B = 73\n"
            "    I_norm = (20 / (73 + 1)) * 255  = (20 / 74) * 255  ≈ 69 (Identical ink intensity!)\n"
            "Notice how division completely cancels out the illumination variation across bright and shadowed regions."
        ),
        "histogram_interpretation": (
            "HISTOGRAM EVOLUTION ACROSS ENHANCEMENT:\n"
            "  1. Original/Tilted Image: Narrow, unimodal distribution bunched in middle greys (values 70-170).\n"
            "     Yellowing and shadows prevent clear separation between text and background.\n"
            "  2. After Normalization & CLAHE: Histogram spreads across the entire dynamic range (0 to 255).\n"
            "  3. After Adaptive Thresholding: Transforms into a clean, strictly BIMODAL histogram:\n"
            "     - Peak at 0 (Pure Black): Represents all ink strokes, punctuation, and text glyphs.\n"
            "     - Peak at 255 (Pure White): Represents the pristine, cleaned paper background."
        ),
        "why_used": (
            "Old documents suffer from non-uniform aging. A global threshold fails because shadowed paper "
            "is darker than illuminated text. Combining illumination division with local adaptive thresholding "
            "solves shadows, stains, and faded ink in a single principled pipeline."
        ),
        "limitations": [
            "Realistic Assessment: Grayscale conversion alone does not chemically remove stains; it removes chrominance.",
            "Heavily soaked ink bleeding through thin paper (bleed-through) may partially remain as residual artifact.",
            "Excessive median filtering can clip the dots on 'i' or thin punctuation commas; hence a gentle 3x3 kernel is used.",
        ],
        "faculty_demo": (
            "Faculty Explanation:\n"
            "\"Feature 3 solves the core problem of old documents: non-uniform illumination and yellowing. "
            "A single global threshold cannot separate ink from paper across shadows. "
            "We model the image as I(x, y) = R(x, y) * L(x, y), where L is slow illumination and R is paper reflectance. "
            "By estimating L using a 51x51 Gaussian filter and dividing I by L, we isolate the true reflectance R. "
            "We then apply a gentle 3x3 median filter for speckle removal, CLAHE for local contrast recovery, "
            "and Gaussian adaptive thresholding T(x, y) = LocalMean(x, y) - 10 to obtain crisp, binarized text.\""
        ),
    },

    # =========================================================================
    # FEATURE 4: TEXT EXTRACTION (OCR)
    # =========================================================================
    FEATURE_OCR: {
        "name": "Feature 4: Text Extraction (OCR)",
        "objective": (
            "Extract readable, machine-encoded text from the preprocessed document image and save it "
            "to a formatted text file (output/output.txt) using traditional Tesseract OCR."
        ),
        "formula": (
            "Traditional Tesseract OCR Architecture & Workflow:\n\n"
            "  [Preprocessed Image] \n"
            "         │\n"
            "         ▼\n"
            "  1. Connected Component Analysis (Blob Detection)\n"
            "         │  Groups contiguous black pixels into character candidates\n"
            "         ▼\n"
            "  2. Text Line & Baseline Profile Fitting\n"
            "         │  Detects horizontal lines and word bounding boxes\n"
            "         ▼\n"
            "  3. Character Cell Segmentation & Chopping\n"
            "         │  Separates touching characters and ligatures\n"
            "         ▼\n"
            "  4. Classical Topological Feature Extraction\n"
            "         │  Extracts stroke directions, loops, concavities, aspect ratios\n"
            "         ▼\n"
            "  5. Linguistic & Dictionary Analysis (2-Pass Disambiguation)\n"
            "         │  Resolves ambiguities using word-level language models\n"
            "         ▼\n"
            "  [Extracted Unicode Text String]"
        ),
        "variables": [
            ("--oem 3", "OCR Engine Mode 3: Default mode using Tesseract's built-in offline engine."),
            ("--psm 6", "Page Segmentation Mode 6: 'Assume a single uniform block of text' (ideal for scanned pages)."),
            ("Blob", "A connected component of foreground ink pixels representing a glyph or character part."),
            ("Baseline", "The imaginary horizontal line upon which characters rest, used to track text lines."),
            ("x-height", "Height of lowercase letters (such as 'x') used to normalize character feature scales."),
        ],
        "opencv_functions": [
            ("ocr.prepare_for_ocr(image)",
             "Ensures 8-bit single-channel input and checks polarity (ensuring dark text on light background)."),
            ("pytesseract.image_to_string(image, config='--oem 3 --psm 6')",
             "Passes the preprocessed NumPy array to the local Tesseract binary and retrieves text."),
            ("ocr.save_text(text, path='output/output.txt')",
             "Persists the extracted UTF-8 text string to disk."),
        ],
        "steps": [
            "1. Polarity Verification: Ensures ink pixels are dark (0) and paper background is bright (255).",
            "2. Connected Component Grouping: Tesseract identifies contiguous foreground pixel blobs.",
            "3. Line & Word Construction: Blobs are gathered into parallel lines using baseline fitting.",
            "4. Character Segmentation: Words are chopped into individual character cells.",
            "5. Feature Matching: Character features (strokes, loops, holes) are matched against trained models.",
            "6. 2-Pass Recognition: Ambiguous characters are resolved using dictionary and linguistic context.",
            "7. File Output: Text is saved to output/output.txt and displayed in the results viewer.",
        ],
        "illustrative_example": (
            "How DIP Preprocessing Directly Affects OCR:\n"
            "  Scenario A (Raw Unenhanced Document Photo):\n"
            "    - Stains and shadows are grouped as erroneous ink blobs, causing garbled characters like '##%%@!'.\n"
            "    - Tilted lines prevent baseline fitting, causing word lines to break prematurely.\n"
            "  Scenario B (After Features 1-3 Preprocessing):\n"
            "    - Document is rectangular and baselines are strictly horizontal.\n"
            "    - Stains and shadows are completely eliminated by background normalization.\n"
            "    - Characters form clean, unbroken closed contours, resulting in accurate text recognition."
        ),
        "histogram_interpretation": (
            "OCR engines perform optimally when the input image exhibits a strictly BIMODAL intensity histogram. "
            "A bimodal histogram guarantees zero ambiguity between ink foreground and paper background. "
            "When an unenhanced image with a broad, continuous histogram is sent to OCR, thresholding ambiguity "
            "leads to broken character stems, merged letters, and high character error rates (CER)."
        ),
        "why_used": (
            "Converts static raster images into searchable, editable text files, completing the digitization pipeline."
        ),
        "limitations": [
            "Severely degraded or physically missing paper sections cannot be synthesized.",
            "Cursive ornate calligraphy or overlapping handwritten annotations may have lower accuracy than standard print.",
        ],
        "faculty_demo": (
            "Faculty Explanation:\n"
            "\"Feature 4 performs optical character recognition using traditional Tesseract OCR via pytesseract. "
            "We deliberately avoid opaque cloud AI APIs and run Tesseract locally with fixed parameters: OEM 3 and PSM 6. "
            "Crucially, OCR success is not achieved through neural guesswork, but through our upstream DIP pipeline: "
            "perspective rectification flattens baselines, while background normalization and adaptive thresholding "
            "provide clean, noise-free character outlines for connected-component blob analysis.\""
        ),
    },
}


def get_explanation(feature_name):
    """Retrieve explanation dictionary for a given feature."""
    if feature_name in EXPLANATIONS:
        return EXPLANATIONS[feature_name]
    # Fallback to border
    return EXPLANATIONS[FEATURE_BORDER]


def format_full_explanation(feature_name, measured_data=None):
    """
    Format a complete, structured, highly professional educational explanation
    incorporating theoretical formulas AND actual measured image metrics.

    Parameters:
      feature_name  : Name of the feature applied
      measured_data : Optional dictionary containing actual empirical measurements:
                      - 'image_size': (width, height)
                      - 'stats_before': dict of min, max, mean, std
                      - 'stats_after': dict of min, max, mean, std
                      - 'patch_before': 3x3 array
                      - 'patch_after': 3x3 array
                      - 'corners': 4x2 array of corners
                      - 'homography_matrix': 3x3 array
                      - 'dest_size': (width, height)
                      - 'ocr_char_count': int
                      - 'status_note': string

    Returns:
      A comprehensive formatted plain-text string.
    """
    exp = get_explanation(feature_name)
    lines = []

    separator = "=" * 78
    sub_sep = "-" * 78

    lines.append(separator)
    lines.append(f"  {exp['name'].upper()}")
    lines.append(separator)
    lines.append("")

    # 1. Objective
    lines.append("[1] OBJECTIVE")
    lines.append(f"    {exp['objective']}")
    lines.append("")

    # 2. Mathematical Formula
    lines.append("[2] MATHEMATICAL FORMULATION")
    for formula_line in exp["formula"].split("\n"):
        lines.append(f"    {formula_line}")
    lines.append("")

    # 3. Variable Explanations
    lines.append("[3] EXPLANATION OF VARIABLES & PARAMETERS")
    for var_name, meaning in exp["variables"]:
        lines.append(f"    • {var_name:<18} : {meaning}")
    lines.append("")

    # 4. OpenCV Functions
    lines.append("[4] OPENCV IMPLEMENTATION & FUNCTIONS")
    for func_sig, description in exp["opencv_functions"]:
        lines.append(f"    • Function : {func_sig}")
        lines.append(f"      Role     : {description}")
    lines.append("")

    # 5. Step-by-Step Processing
    lines.append("[5] STEP-BY-STEP PROCESSING WORKFLOW")
    for step in exp["steps"]:
        lines.append(f"    {step}")
    lines.append("")

    # 6. Illustrative Example vs Actual Measured Data
    lines.append("[6] PIXEL ANALYSIS: THEORY VS. MEASURED DATA")
    lines.append("    A. Illustrative Theoretical Example:")
    for ex_line in exp["illustrative_example"].split("\n"):
        lines.append(f"       {ex_line}")
    lines.append("")

    lines.append("    B. Actual Empirical Measurements From Uploaded Image:")
    if measured_data:
        if "status_note" in measured_data:
            lines.append(f"       Status          : {measured_data['status_note']}")
        if "image_size" in measured_data:
            w, h = measured_data["image_size"]
            lines.append(f"       Current Size    : {w} x {h} pixels")
        if "corners" in measured_data and measured_data["corners"] is not None:
            c = measured_data["corners"]
            lines.append("       Identified Corners (TL, TR, BR, BL):")
            labels = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"]
            for idx, pt in enumerate(c):
                lines.append(f"         {labels[idx]:<18} : x = {pt[0]:.1f}, y = {pt[1]:.1f}")
        if "homography_matrix" in measured_data and measured_data["homography_matrix"] is not None:
            H = measured_data["homography_matrix"]
            lines.append("       Computed 3x3 Homography Matrix H:")
            for row in H:
                lines.append(f"         [ {row[0]:11.4e}  {row[1]:11.4e}  {row[2]:11.4e} ]")
        if "dest_size" in measured_data and measured_data["dest_size"] is not None:
            dw, dh = measured_data["dest_size"]
            lines.append(f"       Rectified Size  : Width = {dw} px, Height = {dh} px")

        # Measured pixel patches
        if "patch_before" in measured_data and measured_data["patch_before"] is not None:
            lines.append("       Center 3x3 Pixel Patch (BEFORE):")
            for row in measured_data["patch_before"]:
                lines.append("         " + "  ".join(f"{int(val):4d}" for val in row))
        if "patch_after" in measured_data and measured_data["patch_after"] is not None:
            lines.append("       Center 3x3 Pixel Patch (AFTER):")
            for row in measured_data["patch_after"]:
                lines.append("         " + "  ".join(f"{int(val):4d}" for val in row))

        # Measured statistics
        if "stats_before" in measured_data and measured_data["stats_before"]:
            sb = measured_data["stats_before"]
            lines.append(f"       Statistics BEFORE: Min={sb['min']}, Max={sb['max']}, Mean={sb['mean']:.2f}, Std={sb['std']:.2f}")
        if "stats_after" in measured_data and measured_data["stats_after"]:
            sa = measured_data["stats_after"]
            lines.append(f"       Statistics AFTER : Min={sa['min']}, Max={sa['max']}, Mean={sa['mean']:.2f}, Std={sa['std']:.2f}")
        if "ocr_char_count" in measured_data:
            lines.append(f"       OCR Recognized  : {measured_data['ocr_char_count']} characters extracted")
    else:
        lines.append("       (Awaiting processing step to compute empirical image statistics)")
    lines.append("")

    # 7. Histogram Interpretation
    lines.append("[7] HISTOGRAM INTERPRETATION")
    for hist_line in exp["histogram_interpretation"].split("\n"):
        lines.append(f"    {hist_line}")
    lines.append("")

    # 8. Why the Technique is Used
    lines.append("[8] PURPOSE & BENEFITS")
    lines.append(f"    {exp['why_used']}")
    lines.append("")

    # 9. Limitations
    lines.append("[9] LIMITATIONS & REALISTIC ASSESSMENT")
    for lim in exp["limitations"]:
        lines.append(f"    • {lim}")
    lines.append("")

    # 10. Faculty Demonstration Explanation
    lines.append("[10] FACULTY PRESENTATION SCRIPT")
    lines.append(sub_sep)
    for demo_line in exp["faculty_demo"].split("\n"):
        lines.append(f"    {demo_line}")
    lines.append(sub_sep)

    return "\n".join(lines)
