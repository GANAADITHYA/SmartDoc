"""
document_scanner.py
-------------------
Traditional boundary detection and perspective rectification for SmartDoc.

Pipeline for automatic boundary detection (Classical DIP only - no deep learning):
  1. Grayscale Conversion: ITU-R BT.601 weighted luminance.
  2. Noise Reduction: 5x5 Gaussian blur to suppress camera grain and texture.
  3. Canny Edge Detection: Multi-stage edge detector computing gradient magnitude
     G = sqrt(Gx^2 + Gy^2) and non-maximum suppression with hysteresis thresholding.
  4. Contour Detection: cv2.findContours traces closed boundary curves.
  5. Polygon Approximation: cv2.approxPolyDP (Ramer-Douglas-Peucker algorithm)
     simplifies curved outlines into polygonal vertices based on epsilon tolerance.
  6. Document Quad Selection: Identifies the largest convex 4-corner contour.

Perspective Transformation:
  - Euclidean distance computes output width and height from corner coordinates.
  - cv2.getPerspectiveTransform solves for the 3x3 homography matrix H.
  - cv2.warpPerspective maps every source pixel into a flat rectangular document.
"""

import cv2
import numpy as np

from image_processing import to_grayscale, check_image


def order_points(points):
    """
    Sort four 2D corner coordinates into a consistent geometric order:
      Index 0: Top-Left (TL)
      Index 1: Top-Right (TR)
      Index 2: Bottom-Right (BR)
      Index 3: Bottom-Left (BL)

    Method:
      - Top-Left has the minimum sum (x + y).
      - Bottom-Right has the maximum sum (x + y).
      - Top-Right has the minimum difference (y - x).
      - Bottom-Left has the maximum difference (y - x).

    Parameters:
      points: Array-like of four (x, y) coordinates.

    Returns:
      NumPy float32 array of shape (4, 2) in [TL, TR, BR, BL] order.
    """
    pts = np.array(points, dtype="float32").reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"Exactly 4 corner points are required. Received {pts.shape[0]}.")

    ordered = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).flatten()  # y - x

    ordered[0] = pts[np.argmin(sums)]   # Top-Left (TL)
    ordered[2] = pts[np.argmax(sums)]   # Bottom-Right (BR)
    ordered[1] = pts[np.argmin(diffs)]  # Top-Right (TR)
    ordered[3] = pts[np.argmax(diffs)]  # Bottom-Left (BL)
    return ordered


def find_document_corners(image, min_area_ratio=0.08):
    """
    Automatically detect the four corners of a document page against its background.

    Steps:
      1. Convert to grayscale.
      2. Gaussian Blur (5x5, sigma=0) to suppress high-frequency noise.
      3. Canny Edge Detection (thresholds 50 and 150) to isolate candidate boundaries.
      4. Find contours via cv2.findContours.
      5. Simplify contours using cv2.approxPolyDP with adaptive perimeter ratios.
      6. Select the largest 4-point convex polygon covering >= min_area_ratio of the image.

    Returns:
      ordered_corners (float32, shape (4, 2)), or None if detection failed.
    """
    check_image(image)
    gray = to_grayscale(image)

    # Step 1 & 2: Gaussian smoothing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny Edge Detection (gradient magnitude thresholding)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges slightly to close minor gaps in the page boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Step 4: Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(image.shape[0] * image.shape[1])
    min_area = min_area_ratio * image_area

    # Sort contours by enclosed area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Try standard epsilon tolerances (0.02, 0.03, 0.015) for RDP polygon approximation
    for epsilon_factor in [0.02, 0.03, 0.015]:
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                break  # Remaining contours are too small

            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)

            # Check if polygon has exactly 4 vertices and is convex
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                return order_points(polygon.reshape(4, 2))

    return None


def validate_corners(corners, image_shape):
    """
    Validate that the corner coordinates are geometrically sensible and within image bounds.
    """
    corners = order_points(corners)
    height, width = image_shape[0], image_shape[1]

    # Boundary check
    if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= width) \
            or np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= height):
        raise ValueError("One or more corner points lie outside the image boundaries.")

    # Dimension checks via Euclidean distances
    top_w = np.linalg.norm(corners[1] - corners[0])
    bottom_w = np.linalg.norm(corners[2] - corners[3])
    left_h = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])

    if min(top_w, bottom_w, left_h, right_h) < 20:
        raise ValueError("Selected corners are too close together to form a valid document page.")

    return corners


def draw_boundary(image, corners):
    """
    Draw a clean, presentation-ready visual overlay on the image:
      - Green polygon perimeter (cv2.polylines)
      - Red circular markers on each corner (cv2.circle)
      - Numbered and named corner badges: 1:TL, 2:TR, 3:BR, 4:BL.

    Returns:
      A new BGR image copy with the boundary and corner badges rendered.
    """
    check_image(image)
    preview = image.copy()
    if len(preview.shape) == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    pts = corners.astype(np.int32).reshape((-1, 1, 2))

    # Draw boundary polygon in bright green
    cv2.polylines(preview, [pts], isClosed=True, color=(0, 230, 0), thickness=3, lineType=cv2.LINE_AA)

    labels = ["1: TL", "2: TR", "3: BR", "4: BL"]
    badge_colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

    for idx, (x, y) in enumerate(corners.astype(np.int32)):
        # Draw outer circle and center dot
        cv2.circle(preview, (int(x), int(y)), 10, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(preview, (int(x), int(y)), 7, badge_colors[idx], -1, lineType=cv2.LINE_AA)

        # Draw label background box for readability
        label_text = labels[idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)

        lx = int(x) + 12
        ly = int(y) - 8
        # Adjust position if near right or top boundary
        if lx + tw > preview.shape[1] - 5:
            lx = int(x) - tw - 12
        if ly - th < 5:
            ly = int(y) + th + 12

        cv2.rectangle(preview, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (20, 20, 20), -1)
        cv2.putText(preview, label_text, (lx, ly), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return preview


def correct_perspective(image, corners=None):
    """
    Rectify perspective distortion to produce a top-down, flat rectangular document.

    Implementation:
      1. Order corners: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
      2. Compute destination width and height using Euclidean distances:
           Width  = max( ||TR - TL||, ||BR - BL|| )
           Height = max( ||BL - TL||, ||BR - TR|| )
      3. Construct destination quad: (0,0), (W-1,0), (W-1,H-1), (0,H-1).
      4. Calculate 3x3 transformation matrix H via cv2.getPerspectiveTransform.
      5. Apply transformation via cv2.warpPerspective.

    Returns:
      (corrected_image, boundary_preview, corners_used, matrix_h)
    """
    check_image(image)

    if corners is None:
        corners = find_document_corners(image)
        if corners is None:
            raise ValueError(
                "Automatic boundary detection could not find four page corners.\n"
                "Please use 'Manual Corner Selection' to click the 4 corners on the original image."
            )

    corners = validate_corners(corners, image.shape)
    top_left, top_right, bottom_right, bottom_left = corners

    # Calculate destination width: maximum of top and bottom edge Euclidean lengths
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    dest_width = int(max(width_top, width_bottom))

    # Calculate destination height: maximum of left and right edge Euclidean lengths
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    dest_height = int(max(height_left, height_right))

    if dest_width < 20 or dest_height < 20:
        raise ValueError("Calculated destination document dimensions are too small. Please reselect corners.")

    # Define source coordinates and destination rectangular corners
    source = corners.astype("float32")
    destination = np.array([
        [0, 0],
        [dest_width - 1, 0],
        [dest_width - 1, dest_height - 1],
        [0, dest_height - 1],
    ], dtype="float32")

    # Compute 3x3 perspective transformation matrix H
    matrix_h = cv2.getPerspectiveTransform(source, destination)

    # Warp the image into the rectangular target coordinate frame
    corrected = cv2.warpPerspective(image, matrix_h, (dest_width, dest_height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    boundary_preview = draw_boundary(image, corners)
    return corrected, boundary_preview, corners, matrix_h
