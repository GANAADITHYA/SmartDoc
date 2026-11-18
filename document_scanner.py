"""
document_scanner.py
-------------------
Traditional boundary detection and perspective rectification for SmartDoc.

Robust Pipeline for automatic boundary detection (Classical DIP only - no deep learning):
  1. Multi-Scale Downscaling: Resizes image to standard analysis height (800 px)
     to suppress high-frequency text clutter while highlighting outer page boundaries.
  2. Multi-Candidate Gradient & Morphological Analysis:
     - Candidate A: Bilateral filter + Adaptive Canny (median-derived thresholds) + Morphological Close
     - Candidate B: Morphological Gradient (cv2.MORPH_GRADIENT) + Otsu thresholding
     - Candidate C: Otsu binarization + Large structuring element closing (solid page mask)
     - Candidate D: Multi-threshold Canny edge linking
  3. Convex Hull & Polygon Simplification:
     - Computes convex hull to bridge small paper tears, ragged edges, and staple folds.
     - Multi-epsilon Ramer-Douglas-Peucker (approxPolyDP) simplification (epsilons 0.01 to 0.06).
     - Geometric quadrilateral validation: internal corner angles checked between 45° and 135°.
  4. Robust Fallback Mechanisms:
     - Fallback A: Minimum Area Bounding Box (cv2.minAreaRect) on the largest convex component.
     - Fallback B: Safe 5% boundary inset ensuring valid initial corners for interactive manual adjustment.

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


def is_valid_quadrilateral(polygon):
    """
    Validate that a 4-vertex polygon represents a plausible document quadrilateral:
      - Must have exactly 4 vertices.
      - Must be convex.
      - All four internal angles must fall within reasonable document perspective limits (45° to 135°).
    """
    if len(polygon) != 4 or not cv2.isContourConvex(polygon):
        return False

    pts = polygon.reshape(4, 2)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 10 or norm2 < 10:
            return False
        cos_angle = np.dot(v1, v2) / (norm1 * norm2 + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        if angle < 45.0 or angle > 135.0:
            return False

    return True


def find_document_corners(image, min_area_ratio=0.06):
    """
    Automatically and robustly detect the four corners of a document page against its background.

    Uses a multi-candidate classical DIP strategy:
      - Normalizes resolution to height 800 for noise-free edge analysis.
      - Evaluates Adaptive Canny, Morphological Gradients, and Otsu Page Masking.
      - Simplifies convex hulls via multi-epsilon RDP polygon approximation.
      - Employs minimum-area rectangle and boundary inset fallbacks if boundaries are faint.

    Returns:
      NumPy float32 array of shape (4, 2) ordered as [TL, TR, BR, BL], or None on complete failure.
    """
    check_image(image)
    h_orig, w_orig = image.shape[:2]

    # Step 1: Scale to standard height (800px) for robust, clutter-free analysis
    target_h = 800.0
    scale = target_h / float(h_orig)
    target_w = int(w_orig * scale)
    small = cv2.resize(image, (target_w, int(target_h)), interpolation=cv2.INTER_AREA)

    if len(small.shape) == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = small.copy()

    image_area = float(target_w * target_h)
    min_area = min_area_ratio * image_area

    # Step 2: Generate complementary candidate edge and binary maps
    candidate_maps = []

    # Map 1: Bilateral smoothing + Adaptive Canny (median-based thresholds) + Closing
    blurred_bi = cv2.bilateralFilter(gray, 9, 75, 75)
    med_val = np.median(blurred_bi)
    lower_c = int(max(15, (1.0 - 0.33) * med_val))
    upper_c = int(min(240, (1.0 + 0.33) * med_val))
    canny1 = cv2.Canny(blurred_bi, lower_c, upper_c)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    canny1_closed = cv2.morphologyEx(canny1, cv2.MORPH_CLOSE, k_close)
    candidate_maps.append(canny1_closed)

    # Map 2: Morphological Gradient (highlights contrast transitions between paper and desk)
    k_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k_grad)
    _, grad_thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate_maps.append(grad_thresh)

    # Map 3: Otsu thresholding + Large Structuring Element Closing (treats page as a solid component)
    blurred_g = cv2.GaussianBlur(gray, (7, 7), 0)
    _, otsu1 = cv2.threshold(blurred_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Check polarity: document paper is usually brighter than surrounding desk
    if np.mean(otsu1) > 127:
        k_page = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        otsu_closed = cv2.morphologyEx(otsu1, cv2.MORPH_CLOSE, k_page)
        candidate_maps.append(otsu_closed)
    else:
        otsu_inv = cv2.bitwise_not(otsu1)
        k_page = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        otsu_closed = cv2.morphologyEx(otsu_inv, cv2.MORPH_CLOSE, k_page)
        candidate_maps.append(otsu_closed)

    # Map 4: Standard Canny with dilation
    blurred_std = cv2.GaussianBlur(gray, (5, 5), 0)
    canny2 = cv2.Canny(blurred_std, 30, 110)
    canny2_dil = cv2.dilate(canny2, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    candidate_maps.append(canny2_dil)

    best_corners = None
    best_area = 0

    # Step 3: Search for optimal 4-vertex convex quadrilateral across candidate maps
    for edge_map in candidate_maps:
        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in sorted_contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                break

            hull = cv2.convexHull(cnt)
            perimeter = cv2.arcLength(hull, True)

            # Try multi-epsilon RDP polygon simplification
            for eps_factor in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]:
                poly = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
                if len(poly) == 4 and is_valid_quadrilateral(poly):
                    if area > best_area:
                        best_area = area
                        scaled_pts = poly.reshape(4, 2) / scale
                        best_corners = order_points(scaled_pts)
                    break

            if best_corners is not None and best_area > 0.35 * image_area:
                break
        if best_corners is not None and best_area > 0.45 * image_area:
            break

    # Step 4: Fallback - Fit minimum area rectangle to the largest prominent convex contour
    if best_corners is None:
        for edge_map in candidate_maps:
            contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 0.12 * image_area:
                hull = cv2.convexHull(largest)
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                scaled_box = box / scale
                best_corners = order_points(scaled_box)
                break

    # Step 5: Safe fallback - 5% image boundary inset
    if best_corners is None:
        inset_x = w_orig * 0.05
        inset_y = h_orig * 0.05
        best_corners = np.array([
            [inset_x, inset_y],
            [w_orig - inset_x, inset_y],
            [w_orig - inset_x, h_orig - inset_y],
            [inset_x, h_orig - inset_y]
        ], dtype="float32")

    return best_corners


def validate_corners(corners, image_shape):
    """
    Validate that corner coordinates are geometrically sensible and within image bounds.
    """
    corners = order_points(corners)
    height, width = image_shape[0], image_shape[1]

    # Clip slight out-of-bounds to image perimeter
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)

    top_w = np.linalg.norm(corners[1] - corners[0])
    bottom_w = np.linalg.norm(corners[2] - corners[3])
    left_h = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])

    if min(top_w, bottom_w, left_h, right_h) < 15:
        raise ValueError("Selected corners are too close together to form a valid document page.")

    return corners


def draw_boundary(image, corners):
    """
    Draw a clean, presentation-ready visual overlay:
      - Green polygon perimeter (cv2.polylines)
      - Distinct colored markers for each corner:
        1: Top-Left (Yellow)
        2: Top-Right (Cyan)
        3: Bottom-Right (Magenta)
        4: Bottom-Left (Lime)
    """
    check_image(image)
    preview = image.copy()
    if len(preview.shape) == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(preview, [pts], isClosed=True, color=(0, 230, 0), thickness=3, lineType=cv2.LINE_AA)

    labels = ["1: TL", "2: TR", "3: BR", "4: BL"]
    badge_colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]  # BGR

    for idx, (x, y) in enumerate(corners.astype(np.int32)):
        # Outer black ring, white middle, colored center
        cv2.circle(preview, (int(x), int(y)), 12, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(preview, (int(x), int(y)), 9, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(preview, (int(x), int(y)), 6, badge_colors[idx], -1, lineType=cv2.LINE_AA)

        label_text = labels[idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)

        lx = int(x) + 14
        ly = int(y) - 8
        if lx + tw > preview.shape[1] - 5:
            lx = int(x) - tw - 14
        if ly - th < 5:
            ly = int(y) + th + 14

        cv2.rectangle(preview, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (15, 23, 42), -1)
        cv2.putText(preview, label_text, (lx, ly), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return preview


def correct_perspective(image, corners=None):
    """
    Rectify perspective distortion to produce a top-down, flat rectangular document.
    Returns (corrected_image, boundary_preview, corners_used, matrix_h).
    """
    check_image(image)

    if corners is None:
        corners = find_document_corners(image)
        if corners is None:
            raise ValueError(
                "Automatic boundary detection could not find four page corners.\n"
                "Please use 'Adjust / Manual Corners' to position the 4 corners."
            )

    corners = validate_corners(corners, image.shape)
    top_left, top_right, bottom_right, bottom_left = corners

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    dest_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    dest_height = int(max(height_left, height_right))

    if dest_width < 20 or dest_height < 20:
        raise ValueError("Calculated destination document dimensions are too small. Please reselect corners.")

    source = corners.astype("float32")
    destination = np.array([
        [0, 0],
        [dest_width - 1, 0],
        [dest_width - 1, dest_height - 1],
        [0, dest_height - 1],
    ], dtype="float32")

    matrix_h = cv2.getPerspectiveTransform(source, destination)
    corrected = cv2.warpPerspective(image, matrix_h, (dest_width, dest_height),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    boundary_preview = draw_boundary(image, corners)
    return corrected, boundary_preview, corners, matrix_h
