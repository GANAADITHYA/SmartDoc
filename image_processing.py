"""
image_processing.py
-------------------
Traditional Digital Image Processing (DIP) algorithms for SmartDoc.

All operations strictly use classical computer vision and image processing methods:
  - Grayscale conversion (weighted luminance)
  - Background illumination estimation (large Gaussian filter)
  - Background normalization (intensity division)
  - Noise reduction (median filtering)
  - Local contrast enhancement (CLAHE)
  - Adaptive thresholding (local mean binarization)
  - Intensity histogram calculation (cv2.calcHist)
  - Statistical analysis

STRICTLY TRADITIONAL ALGORITHMS:
No machine learning, no deep learning, and no neural networks are used.
Every function produces a new NumPy array and preserves the original image.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Validation and Helper Functions
# ---------------------------------------------------------------------------

def check_image(image):
    """
    Validate that the input is a non-empty NumPy image array.
    Raises ValueError if invalid.
    """
    if image is None:
        raise ValueError("No image provided. Please upload an image first.")
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a valid NumPy array.")
    if image.size == 0:
        raise ValueError("The provided image is empty.")
    return True


def check_odd_kernel(kernel_size):
    """
    Ensure the kernel size is a positive odd integer (e.g. 3, 5, 7, 51).
    Spatial filters require a well-defined center pixel.
    """
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"Kernel size must be a positive odd integer (e.g. 3, 5, 7, 51). Got {kernel_size}."
        )
    return True


def to_grayscale(image):
    """
    Convert image to single-channel 8-bit grayscale if it has color channels.
    Uses standard ITU-R BT.601 luminance weights:
      Gray = 0.299 * R + 0.587 * G + 0.114 * B
    """
    check_image(image)
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_grayscale(image):
    """User-facing alias for grayscale conversion."""
    return to_grayscale(image)


# ---------------------------------------------------------------------------
# Spatial Filtering & Background Estimation
# ---------------------------------------------------------------------------

def apply_gaussian_blur(image, kernel_size=5):
    """
    Apply a 2D Gaussian filter to smooth an image and suppress high-frequency noise.

    Formula:
      G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

    OpenCV:
      cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    """
    check_image(image)
    check_odd_kernel(kernel_size)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_blur(image, kernel_size=3):
    """
    Apply a non-linear median filter to suppress salt-and-pepper noise and speckles
    while preserving sharp edge transitions.

    Formula:
      I_out(x, y) = median({ I(x + i, y + j) | (i, j) in W })

    OpenCV:
      cv2.medianBlur(image, kernel_size)
    """
    check_image(image)
    check_odd_kernel(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def estimate_background(gray_image, blur_size=51):
    """
    Estimate slowly-varying illumination and paper background intensity.
    A large Gaussian blur filters out fine text details, retaining only
    the low-frequency illumination profile (shadows and paper tone).

    OpenCV:
      cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)
    """
    check_image(gray_image)
    check_odd_kernel(blur_size)
    if len(gray_image.shape) != 2:
        gray_image = to_grayscale(gray_image)
    return cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)


def normalize_background(gray_image, background_image, scaling_factor=255.0):
    """
    Compensate for uneven illumination and yellowed paper via background normalization.

    Formula:
      I_norm(x, y) = [ I(x, y) / (B(x, y) + epsilon) ] * K

    Where:
      - I(x, y) = Grayscale image intensity
      - B(x, y) = Estimated low-frequency background
      - epsilon = Small constant (1.0) to prevent division by zero
      - K = Scaling factor (255) to map the ratio back to standard 8-bit dynamic range.
    """
    check_image(gray_image)
    check_image(background_image)

    # Convert to 32-bit float for accurate division
    img_f = gray_image.astype(np.float32)
    bg_f = background_image.astype(np.float32)

    # Normalize by division with epsilon safeguard
    epsilon = 1.0
    normalized_f = (img_f / (bg_f + epsilon)) * scaling_factor

    # Clip to valid 8-bit range [0, 255] and convert back to uint8
    normalized = np.clip(normalized_f, 0, 255).astype(np.uint8)
    return normalized


def remove_shadows(image, blur_size=51):
    """
    Combined helper: estimates background and performs normalization.
    Returns (normalized_image, background_image).
    """
    gray = to_grayscale(image)
    bg = estimate_background(gray, blur_size=blur_size)
    normalized = normalize_background(gray, bg, scaling_factor=255.0)
    return normalized, bg


# ---------------------------------------------------------------------------
# Contrast Enhancement & Thresholding
# ---------------------------------------------------------------------------

def apply_clahe(image, clip_limit=2.0, tile_grid_size=8):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contrast in small contextual tiles (e.g. 8x8) while
    clipping the histogram to prevent noise over-amplification.

    Parameters:
      clip_limit     : Threshold for contrast limiting (default: 2.0)
      tile_grid_size : Size of grid for histogram equalization (default: 8x8)

    OpenCV:
      clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
      clahe.apply(gray_image)
    """
    gray = to_grayscale(image)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be strictly greater than 0.")
    if not isinstance(tile_grid_size, int) or tile_grid_size < 1:
        raise ValueError("tile_grid_size must be a positive integer.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray)


def apply_adaptive_threshold(image, block_size=None, c_value=8):
    """
    Binarize document image using adaptive Gaussian-weighted thresholding.
    Computes a localized threshold for every individual pixel based on its
    local neighborhood mean minus a fine-tuning constant C.

    If block_size is None, it is dynamically computed based on image resolution
    to prevent hollowing out thick strokes on high-resolution photographs.

    Formula:
      T(x, y) = LocalMean(x, y) - C
      I_bin(x, y) = 255 if I(x, y) > T(x, y) else 0
    """
    gray = to_grayscale(image)
    if block_size is None or block_size <= 0:
        max_dim = max(gray.shape[:2])
        block_size = int(max_dim / 45)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(15, min(block_size, 61))

    check_odd_kernel(block_size)
    if block_size < 3:
        raise ValueError("block_size must be an odd integer >= 3.")

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )
    return thresh


# ---------------------------------------------------------------------------
# FEATURE 3: Complete Old Document Enhancement Pipeline
# ---------------------------------------------------------------------------

def enhance_old_document(image, bg_blur_size=51, noise_kernel=3,
                         clip_limit=2.0, tile_grid=8,
                         block_size=None, c_value=8):
    """
    Complete Old Document Enhancement Pipeline.
    Combines a principled sequence of 6 traditional DIP algorithms to address:
      - Yellowed paper and aged tone
      - Coffee stains and liquid spills
      - Uneven shadows and non-uniform illumination
      - Faded ink strokes
      - Sensor noise and paper grain

    Sequence:
      STEP 1: Grayscale Conversion (cv2.cvtColor)
      STEP 2: Background Estimation (large Gaussian blur 51x51)
      STEP 3: Background Normalization (cv2.divide / formula)
      STEP 4: Noise Reduction (gentle 3x3 median blur)
      STEP 5: Contrast Enhancement (CLAHE: clipLimit=2.0, tileGridSize=8x8)
      STEP 6: Adaptive Thresholding (Resolution-adaptive Gaussian binarization)

    Returns:
      (final_enhanced_image, intermediate_stages)
      where intermediate_stages is a list of (stage_name, image) tuples.
    """
    check_image(image)
    stages = []

    # Step 1: Grayscale Conversion
    gray = to_grayscale(image)
    stages.append(("1. Grayscale Conversion", gray))

    # Step 2: Background Estimation
    background = estimate_background(gray, blur_size=bg_blur_size)
    stages.append(("2. Estimated Background", background))

    # Step 3: Background Normalization
    normalized = normalize_background(gray, background, scaling_factor=255.0)
    stages.append(("3. Background Normalization", normalized))

    # Step 4: Noise Reduction
    denoised = apply_median_blur(normalized, kernel_size=noise_kernel)
    stages.append(("4. Noise Reduction (Median)", denoised))

    # Step 5: Contrast Enhancement (CLAHE)
    contrast = apply_clahe(denoised, clip_limit=clip_limit, tile_grid_size=tile_grid)
    stages.append(("5. Contrast Enhancement (CLAHE)", contrast))

    # Step 6: Adaptive Thresholding
    final_thresh = apply_adaptive_threshold(contrast, block_size=block_size, c_value=c_value)
    stages.append(("6. Adaptive Thresholding", final_thresh))

    return final_thresh, stages


# ---------------------------------------------------------------------------
# Histogram and Statistical Analysis
# ---------------------------------------------------------------------------

def compute_histogram(gray_image):
    """
    Calculate the 256-bin intensity histogram of a grayscale image using cv2.calcHist.

    Parameters:
      gray_image : 2D single-channel NumPy array (or BGR image, auto-converted)

    Returns:
      1D NumPy float32 array of length 256, where histogram[i] represents
      the total count of pixels with intensity value i.
    """
    gray = to_grayscale(gray_image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten()


def compute_statistics(image):
    """
    Compute actual empirical pixel statistics of an image.
    Used by the Educational Explanation Panel to display real measured metrics.

    Returns:
      dict with min, max, mean, std, height, width, channels, and dynamic_range.
    """
    check_image(image)
    if len(image.shape) == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape[:3]

    min_val = int(np.min(image))
    max_val = int(np.max(image))
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "dynamic_range": max_val - min_val,
    }


def center_patch(gray_image, size=3):
    """
    Extract a square patch (e.g. 3x3) from the geometric center of a grayscale image.
    Used for before-and-after empirical pixel intensity comparison.
    """
    gray = to_grayscale(gray_image)
    height, width = gray.shape
    if height < size or width < size:
        return None
    cy, cx = height // 2, width // 2
    radius = size // 2
    return gray[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
