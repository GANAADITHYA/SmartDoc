"""
ocr.py
------
Traditional Optical Character Recognition (OCR) using Tesseract through pytesseract.

In this project, Tesseract is configured and used strictly as a classical OCR engine:
  - Fixed standard configuration: --oem 3 --psm 6
  - Image preprocessing done entirely with traditional DIP (perspective flattening,
    background normalization, CLAHE, and adaptive thresholding).
  - No deep learning or AI-based OCR services are called over the network.
"""

import os
import shutil
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Default Windows installation path for Tesseract OCR (UB Mannheim build)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Standard configuration:
#   --oem 3 : OCR Engine Mode 3 (Default, Tesseract's built-in engine)
#   --psm 6 : Page Segmentation Mode 6 (Assume a single uniform block of text)
OCR_CONFIG = "--oem 3 --psm 6"

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_OUTPUT_TEXT = os.path.join(DEFAULT_OUTPUT_DIR, "output.txt")


def configure_tesseract():
    """
    Configure pytesseract to locate the tesseract executable.
    Checks the default Windows install path and system PATH.
    """
    if pytesseract is None:
        return False

    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        return True

    # Fall back to finding tesseract on the system PATH
    system_tesseract = shutil.which("tesseract")
    if system_tesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = system_tesseract
        return True

    return False


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
            f"Checked path: {TESSERACT_PATH}\n"
            f"Details: {error}\n\n"
            "Please ensure Tesseract OCR is installed (e.g. from UB Mannheim build)."
        )


def prepare_for_ocr(image):
    """
    Prepare an image for OCR by ensuring single-channel 8-bit format
    and dark text on a light background (polarity check).
    """
    if image is None:
        raise ValueError("No image provided for OCR.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # If the image is inverted (light text on dark background), invert it
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    return gray


def extract_text(image, config=OCR_CONFIG):
    """
    Extract text from an image using pytesseract with fixed configuration.

    Parameters:
      image  : NumPy image array (grayscale or BGR)
      config : Tesseract parameter string (default: '--oem 3 --psm 6')

    Returns:
      Extracted text string.
    """
    installed, message = check_tesseract_installed()
    if not installed:
        raise RuntimeError(message)

    prepared = prepare_for_ocr(image)
    extracted = pytesseract.image_to_string(prepared, config=config)
    return extracted.strip()


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


OCR_WORKFLOW_INFO = {
    "what_is_ocr": (
        "OCR (Optical Character Recognition) is the electronic conversion of document images "
        "into editable, searchable machine-encoded text. It operates by identifying character "
        "shapes and pattern features within raster images."
    ),
    "what_is_tesseract": (
        "Tesseract is an open-source, offline OCR engine originally developed by HP and maintained "
        "by Google. In this project, it executes locally via pytesseract without network calls."
    ),
    "oem_explanation": (
        "OEM (OCR Engine Mode): Specifies the underlying recognition algorithm. Mode 3 ('--oem 3') "
        "selects the default engine mode with the best available baseline accuracy."
    ),
    "psm_explanation": (
        "PSM (Page Segmentation Mode): Informs Tesseract how text regions are structured on the page. "
        "Mode 6 ('--psm 6') enforces: 'Assume a single uniform block of text', which perfectly matches "
        "straightened, scanned document pages."
    ),
    "why_dip_helps_ocr": (
        "OCR engines rely on clean character contours. Before image enhancement, old documents suffer "
        "from non-uniform illumination, yellow stains, and tilt, which cause character segmentation "
        "failures and misclassification. Perspective flattening aligns text baselines horizontally, "
        "while background normalization and adaptive thresholding produce crisp black character glyphs "
        "on a pristine white background, dramatically boosting recognition accuracy."
    ),
}
