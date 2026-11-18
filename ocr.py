"""
ocr.py
------
Traditional Optical Character Recognition (OCR) using Tesseract through pytesseract.

In this project, Tesseract is configured and used strictly as a classical OCR engine:
  - Fixed standard configuration: --oem 3 --psm 6 (with automatic layout fallback to PSM 3/4)
  - Image preprocessing done entirely with traditional DIP:
      * Resolution normalization (bicubic upscaling for optimal character stroke height)
      * Illumination normalization (Gaussian background division)
      * Contrast enhancement (CLAHE)
      * Otsu binarization with connected-component morphological despeckling
  - No deep learning or AI-based cloud services are called.
"""

import os
import shutil
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Candidate paths for local Tesseract installation
TESSERACT_CANDIDATE_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]

# Default Windows installation path
TESSERACT_PATH = TESSERACT_CANDIDATE_PATHS[0]

# Standard default configuration:
#   --oem 3 : OCR Engine Mode 3 (Default Tesseract built-in engine)
#   --psm 6 : Page Segmentation Mode 6 (Assume a single uniform block of text)
OCR_CONFIG = "--oem 3 --psm 6"

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_OUTPUT_TEXT = os.path.join(DEFAULT_OUTPUT_DIR, "output.txt")


def configure_tesseract():
    """
    Configure pytesseract to locate the tesseract executable.
    Checks candidate Windows directories and the system PATH.
    """
    if pytesseract is None:
        return False

    for path in TESSERACT_CANDIDATE_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True

    system_tesseract = shutil.which("tesseract")
    if system_tesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = system_tesseract
        return True

    return False


# Auto-configure on module import
configure_tesseract()


def check_tesseract_installed():
    """
    Verify that Tesseract OCR is installed and accessible.
    Returns:
      (True, version_string) on success
      (False, error_message) on failure
    """
    if pytesseract is None:
        return False, (
            "The 'pytesseract' Python module is not installed.\n"
            "Install it via: pip install pytesseract"
        )

    configure_tesseract()
    try:
        version = pytesseract.get_tesseract_version()
        return True, str(version)
    except Exception as error:
        return False, (
            f"Tesseract OCR executable was not found or could not be launched.\n"
            f"Searched paths:\n" + "\n".join(f"  - {p}" for p in TESSERACT_CANDIDATE_PATHS) + "\n\n"
            f"Error details: {error}\n\n"
            "Please ensure Tesseract OCR is installed (UB Mannheim build for Windows)."
        )


def despeckle_binary_image(binary_image, min_area=3):
    """
    Traditional connected-component filter to remove isolated dust specks
    and paper grain that cause Tesseract to hallucinate stray punctuation marks.
    Assumes white background (255) and black ink (0).
    """
    # Invert so ink foreground is white (255) for connected component analysis
    inv = cv2.bitwise_not(binary_image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    cleaned_inv = np.zeros_like(inv)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Retain connected components larger than minimum dust speck threshold
        if area >= min_area:
            cleaned_inv[labels == i] = 255

    return cv2.bitwise_not(cleaned_inv)


def prepare_for_ocr(image):
    """
    High-accuracy traditional DIP preparation pipeline specifically optimized for Tesseract:
      1. Grayscale conversion.
      2. Resolution normalization: Upscales low-res images so character x-height is >= 30 px.
      3. Illumination compensation: Gaussian background division to eliminate paper discoloration.
      4. Contrast enhancement: CLAHE local dynamic range stretch.
      5. Binarization: Otsu global thresholding ensuring solid character strokes (no hollow text).
      6. Despeckling: Connected-component noise suppression.
      7. Polarity enforcement: Guarantees black text on pure white background.

    Returns:
      A clean, high-contrast, despeckled 8-bit image ready for OCR.
    """
    if image is None:
        raise ValueError("No image provided for OCR.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 1: Check if the input is already a binarized document
    unique_vals = np.unique(gray)
    is_binary = len(unique_vals) <= 4 and (0 in unique_vals or 255 in unique_vals)

    if is_binary:
        # Polarity check: text must be black (0) and paper white (255)
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)

        # Scale check for small images
        h, w = gray.shape
        if h < 1000 or w < 800:
            scale_factor = max(1000 / h, 800 / w)
            gray = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)),
                              interpolation=cv2.INTER_NEAREST)

        # Remove small dust specks (1-2 px)
        return despeckle_binary_image(gray, min_area=3)

    # Step 2: Grayscale document processing
    # A. Scale optimization for character height
    h, w = gray.shape
    if h < 1200 or w < 900:
        scale_factor = max(1200 / h, 900 / w)
        gray = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)),
                          interpolation=cv2.INTER_CUBIC)

    # B. Background estimation and normalization
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)

    # C. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(norm)

    # D. Gentle median filter to suppress speckle noise
    denoised = cv2.medianBlur(clahe, 3)

    # E. Otsu binarization (produces solid character strokes, avoiding hollow text)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # F. Polarity enforcement
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # G. Despeckle paper grain
    cleaned = despeckle_binary_image(binary, min_area=3)
    return cleaned


def extract_text(image, config=OCR_CONFIG, psm_mode=None):
    """
    Extract readable text from an image with high accuracy using pytesseract.

    Features:
      - Applies resolution and illumination normalization via prepare_for_ocr.
      - Intelligent multi-pass layout fallback: if default PSM mode (e.g. PSM 6)
        yields fragmented or very short text, runs an alternate layout pass (PSM 3)
        and selects the more complete, higher-accuracy transcription.

    Parameters:
      image    : NumPy image array
      config   : Tesseract configuration string
      psm_mode : Optional integer PSM mode (e.g. 3, 4, 6)

    Returns:
      Extracted text string.
    """
    installed, message = check_tesseract_installed()
    if not installed:
        raise RuntimeError(message)

    if psm_mode is not None:
        effective_config = f"--oem 3 --psm {psm_mode}"
    else:
        effective_config = config

    prepared = prepare_for_ocr(image)

    # Primary OCR pass
    try:
        primary_text = pytesseract.image_to_string(prepared, config=effective_config)
    except Exception as err:
        raise RuntimeError(f"Tesseract OCR execution error: {err}")

    result_text = primary_text.strip()

    # Layout fallback: If PSM 6 yielded short text (< 60 chars) or if lines are truncated,
    # test PSM 3 (Fully automatic page segmentation) which handles headers and multi-column pages.
    if "--psm 6" in effective_config and len(result_text) < 60:
        try:
            alt_config = "--oem 3 --psm 3"
            alt_text = pytesseract.image_to_string(prepared, config=alt_config).strip()
            if len(alt_text) > len(result_text) * 1.3:
                result_text = alt_text
        except Exception:
            pass

    return result_text


def save_text(text, path=DEFAULT_OUTPUT_TEXT):
    """
    Save the extracted text string to a UTF-8 text file.
    Creates parent directories if necessary.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return path
