"""
main.py
-------
SmartDoc - Interactive Old Document Scanner and DIP Educational System.

A clean, presentation-friendly Digital Image Processing desktop application
organized around ONLY FOUR CORE FEATURES:
  1. Document Border Detection
  2. Perspective Transformation
  3. Old Document Enhancement
  4. Text Extraction (OCR)

STRICTLY TRADITIONAL COMPUTER VISION & DIP:
  - OpenCV, NumPy, Tkinter, Pillow, Matplotlib, Pytesseract.
  - Zero machine learning, zero deep learning, zero cloud AI.

Features:
  - Automatic & manual 4-corner document boundary detection.
  - Perspective rectification via 3x3 homography matrix warping.
  - Principled 6-step old document restoration pipeline with stage inspection.
  - Traditional Tesseract OCR (--oem 3 --psm 6) with auto-save to output/output.txt.
  - Embedded side-by-side Matplotlib grayscale intensity histograms.
  - In-depth Educational Explanation Panel with real measured pixel statistics,
    mathematical formulas, and faculty presentation scripts.
"""

import os
import queue
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Set Matplotlib config dir to avoid permission warnings on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_smartdoc"))

import cv2
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import algorithm_explanations as ae
import document_scanner as ds
import image_processing as ip
import ocr

# ---------------------------------------------------------------------------
# Constants and UI Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "input_images"
OUTPUT_DIR = "output"
DEFAULT_SCANNED_DOC = os.path.join(OUTPUT_DIR, "scanned_doc.png")
DEFAULT_TEXT_DOC = os.path.join(OUTPUT_DIR, "output.txt")

COLOR_BG = "#f8fafc"            # Modern crisp light background
COLOR_CARD = "#ffffff"          # Pure white cards
COLOR_NAVY = "#0f172a"          # Deep navy for primary headings
COLOR_SUBNAVY = "#1e293b"       # Secondary navy
COLOR_BLUE = "#1d4ed8"          # Primary action blue
COLOR_BLUE_HOVER = "#1e40af"
COLOR_GREEN = "#059669"         # Success / confirm green
COLOR_BORDER = "#cbd5e1"        # Subtle card border
COLOR_TEXT = "#334155"          # Main text
COLOR_MUTED = "#64748b"         # Muted helper text

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_CARD_TITLE = ("Segoe UI", 11, "bold")
FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_CODE = ("Consolas", 10)

DISPLAY_MAX_WIDTH = 580
DISPLAY_MAX_HEIGHT = 440


# ---------------------------------------------------------------------------
# Helper Display Functions
# ---------------------------------------------------------------------------

def to_photo(image, max_w=DISPLAY_MAX_WIDTH, max_h=DISPLAY_MAX_HEIGHT):
    """
    Scale an OpenCV image to fit within max_w x max_h preserving aspect ratio,
    and convert it to a Tkinter PhotoImage.
    """
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)
    return ImageTk.PhotoImage(Image.fromarray(resized)), scale


def describe_image(image):
    """Format short summary string of image dimensions and channel format."""
    if image is None:
        return "No image"
    if len(image.shape) == 2:
        h, w = image.shape
        return f"{w} x {h} px | Grayscale (8-bit)"
    h, w, c = image.shape
    return f"{w} x {h} px | BGR Color ({c} channels)"


# ---------------------------------------------------------------------------
# Main Application Class
# ---------------------------------------------------------------------------

class SmartDocApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartDoc - Old Document Scanner & DIP Educational System")
        self.root.geometry("1440x960")
        self.root.minsize(1100, 750)
        self.root.configure(bg=COLOR_BG)

        # Image state
        self.original_image = None       # Never modified pristine upload
        self.current_processed = None     # Current processed output
        self.processing_stack = []        # For Undo: list of (processed_img, feature_name, measured_data)
        self.history_records = []         # For History table: list of dicts

        # Feature 1 & 2 state
        self.detected_corners = None     # 4x2 float32 array
        self.manual_points = []          # List of manually clicked points
        self.manual_selection_mode = False
        self.left_display_scale = 1.0
        self.homography_matrix = None

        # Feature 3 state
        self.pipeline_stages = []        # List of (stage_name, stage_image)

        # Feature 4 state
        self.last_extracted_text = ""
        self.ocr_queue = queue.Queue()

        # UI References
        self.left_photo = None
        self.right_photo = None

        # Ensure output directory exists
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self._build_ui()
        self.set_status("Welcome to SmartDoc. Click 'Upload Document Photo' to begin.")
        self._init_explanation()

    # -----------------------------------------------------------------------
    # GUI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # Create a scrollable master frame
        self.canvas_scroll = tk.Canvas(self.root, bg=COLOR_BG, highlightthickness=0)
        self.scrollbar_v = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas_scroll.yview)
        self.main_container = tk.Frame(self.canvas_scroll, bg=COLOR_BG)

        self.main_container.bind(
            "<Configure>",
            lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        )
        self.canvas_window = self.canvas_scroll.create_window((0, 0), window=self.main_container, anchor="nw")
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar_v.set)

        self.canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mousewheel scrolling across entire application
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas_scroll.bind("<Configure>", self._on_canvas_configure)

        # Build individual sections
        self._build_header_section()
        self._build_workflow_cards()
        self._build_image_comparison_section()
        self._build_histogram_section()
        self._build_explanation_section()
        self._build_bottom_controls()
        self._build_status_bar()

    def _on_canvas_configure(self, event):
        self.canvas_scroll.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        # Don't hijack mousewheel if focus is inside text widget
        focused = self.root.focus_get()
        if isinstance(focused, tk.Text):
            return
        self.canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # -----------------------------------------------------------------------
    # Top Section: Header & Upload
    # -----------------------------------------------------------------------

    def _build_header_section(self):
        header = tk.Frame(self.main_container, bg=COLOR_CARD, padx=20, pady=12,
                          highlightbackground=COLOR_BORDER, highlightthickness=1)
        header.pack(fill=tk.X, padx=14, pady=(10, 6))

        # Title & Subtitle
        left_box = tk.Frame(header, bg=COLOR_CARD)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        title_lbl = tk.Label(left_box, text="SmartDoc — Old Document Scanner",
                             font=FONT_TITLE, fg=COLOR_NAVY, bg=COLOR_CARD)
        title_lbl.pack(anchor="w")

        desc_text = (
            "Classical Digital Image Processing (DIP) & Tesseract OCR System | "
            "Border Detection • Perspective Rectification • Old Document Enhancement • Text Extraction"
        )
        sub_lbl = tk.Label(left_box, text=desc_text, font=FONT_SUBTITLE, fg=COLOR_MUTED, bg=COLOR_CARD)
        sub_lbl.pack(anchor="w", pady=(2, 0))

        # Upload Button
        btn_upload = tk.Button(header, text="📁 Upload Document Photo", font=FONT_LABEL,
                               bg=COLOR_BLUE, fg="white", activebackground=COLOR_BLUE_HOVER,
                               activeforeground="white", relief=tk.FLAT, padx=16, pady=8,
                               cursor="hand2", command=self.upload_image)
        btn_upload.pack(side=tk.RIGHT, padx=6)

    # -----------------------------------------------------------------------
    # Workflow Cards: Four Main Features
    # -----------------------------------------------------------------------

    def _build_workflow_cards(self):
        card_container = tk.Frame(self.main_container, bg=COLOR_BG)
        card_container.pack(fill=tk.X, padx=14, pady=6)

        for col in range(4):
            card_container.columnconfigure(col, weight=1)

        # Card 1: Border Detection
        c1 = tk.Frame(card_container, bg=COLOR_CARD, padx=12, pady=10,
                      highlightbackground=COLOR_BORDER, highlightthickness=1)
        c1.grid(row=0, column=0, sticky="nsew", padx=4)
        tk.Label(c1, text="1. BORDER DETECTION", font=FONT_CARD_TITLE, fg=COLOR_NAVY, bg=COLOR_CARD).pack(anchor="w")
        tk.Label(c1, text="Find 4 document corners via Canny gradients & approxPolyDP",
                 font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD, wraplength=260, justify="left").pack(anchor="w", pady=(2, 8))

        btn_box1 = tk.Frame(c1, bg=COLOR_CARD)
        btn_box1.pack(fill=tk.X)
        self.btn_detect = tk.Button(btn_box1, text="Detect Border", font=FONT_BODY, bg="#0284c7", fg="white",
                                    relief=tk.FLAT, padx=8, pady=4, cursor="hand2", command=self.feature1_detect_border)
        self.btn_detect.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_manual_corner = tk.Button(btn_box1, text="Manual Corners", font=FONT_SMALL, bg="#e2e8f0", fg=COLOR_NAVY,
                                           relief=tk.FLAT, padx=6, pady=4, cursor="hand2", command=self.toggle_manual_corners)
        self.btn_manual_corner.pack(side=tk.LEFT)

        # Card 2: Perspective Correction
        c2 = tk.Frame(card_container, bg=COLOR_CARD, padx=12, pady=10,
                      highlightbackground=COLOR_BORDER, highlightthickness=1)
        c2.grid(row=0, column=1, sticky="nsew", padx=4)
        tk.Label(c2, text="2. PERSPECTIVE WARP", font=FONT_CARD_TITLE, fg=COLOR_NAVY, bg=COLOR_CARD).pack(anchor="w")
        tk.Label(c2, text="Straighten tilted document into a flat rectangle using 3x3 matrix H",
                 font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD, wraplength=260, justify="left").pack(anchor="w", pady=(2, 8))

        self.btn_perspective = tk.Button(c2, text="Correct Perspective", font=FONT_BODY, bg="#0284c7", fg="white",
                                         relief=tk.FLAT, padx=10, pady=4, cursor="hand2", command=self.feature2_correct_perspective)
        self.btn_perspective.pack(anchor="w")

        # Card 3: Old Document Enhancement
        c3 = tk.Frame(card_container, bg=COLOR_CARD, padx=12, pady=10,
                      highlightbackground=COLOR_BORDER, highlightthickness=1)
        c3.grid(row=0, column=2, sticky="nsew", padx=4)
        tk.Label(c3, text="3. ENHANCE DOCUMENT", font=FONT_CARD_TITLE, fg=COLOR_NAVY, bg=COLOR_CARD).pack(anchor="w")
        tk.Label(c3, text="6-step pipeline: norm, median, CLAHE, adaptive threshold",
                 font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD, wraplength=260, justify="left").pack(anchor="w", pady=(2, 8))

        btn_box3 = tk.Frame(c3, bg=COLOR_CARD)
        btn_box3.pack(fill=tk.X)
        self.btn_enhance = tk.Button(btn_box3, text="Enhance Old Document", font=FONT_BODY, bg=COLOR_GREEN, fg="white",
                                     relief=tk.FLAT, padx=8, pady=4, cursor="hand2", command=self.feature3_enhance_document)
        self.btn_enhance.pack(side=tk.LEFT, padx=(0, 4))

        # Intermediate stage combobox
        self.stage_var = tk.StringVar(value="View Stage...")
        self.stage_box = ttk.Combobox(c3, textvariable=self.stage_var, state="readonly",
                                      values=["Run Enhance first"], width=24, font=FONT_SMALL)
        self.stage_box.bind("<<ComboboxSelected>>", self.on_stage_selected)
        self.stage_box.pack(anchor="w", pady=(6, 0))

        # Card 4: Text Extraction (OCR)
        c4 = tk.Frame(card_container, bg=COLOR_CARD, padx=12, pady=10,
                      highlightbackground=COLOR_BORDER, highlightthickness=1)
        c4.grid(row=0, column=3, sticky="nsew", padx=4)
        tk.Label(c4, text="4. EXTRACT TEXT (OCR)", font=FONT_CARD_TITLE, fg=COLOR_NAVY, bg=COLOR_CARD).pack(anchor="w")
        tk.Label(c4, text="Traditional Tesseract OCR (--oem 3 --psm 6) to text file",
                 font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD, wraplength=260, justify="left").pack(anchor="w", pady=(2, 8))

        btn_box4 = tk.Frame(c4, bg=COLOR_CARD)
        btn_box4.pack(fill=tk.X)
        self.btn_ocr = tk.Button(btn_box4, text="Extract Text (OCR)", font=FONT_BODY, bg="#7c3aed", fg="white",
                                 relief=tk.FLAT, padx=8, pady=4, cursor="hand2", command=self.feature4_extract_text)
        self.btn_ocr.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_view_text = tk.Button(btn_box4, text="View Text", font=FONT_SMALL, bg="#e2e8f0", fg=COLOR_NAVY,
                                       relief=tk.FLAT, padx=6, pady=4, cursor="hand2", command=self.open_text_dialog)
        self.btn_view_text.pack(side=tk.LEFT)

    # -----------------------------------------------------------------------
    # Image Comparison Section: Left (Original) vs Right (Processed)
    # -----------------------------------------------------------------------

    def _build_image_comparison_section(self):
        comparison_frame = tk.Frame(self.main_container, bg=COLOR_BG)
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
        comparison_frame.columnconfigure(0, weight=1)
        comparison_frame.columnconfigure(1, weight=1)

        # LEFT PANEL: Original Document
        left_card = tk.Frame(comparison_frame, bg=COLOR_CARD, padx=12, pady=10,
                             highlightbackground=COLOR_BORDER, highlightthickness=1)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        header_l = tk.Frame(left_card, bg=COLOR_CARD)
        header_l.pack(fill=tk.X)
        tk.Label(header_l, text="ORIGINAL DOCUMENT", font=FONT_LABEL, fg=COLOR_NAVY, bg=COLOR_CARD).pack(side=tk.LEFT)
        self.lbl_left_info = tk.Label(header_l, text="No image loaded", font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD)
        self.lbl_left_info.pack(side=tk.RIGHT)

        self.lbl_left_image = tk.Label(left_card, text="\n\nClick 'Upload Document Photo' to load an image\n\n",
                                       font=FONT_BODY, bg="#f1f5f9", fg=COLOR_MUTED, relief=tk.FLAT)
        self.lbl_left_image.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.lbl_left_image.bind("<Button-1>", self.on_left_image_click)

        # RIGHT PANEL: Processed Document
        right_card = tk.Frame(comparison_frame, bg=COLOR_CARD, padx=12, pady=10,
                              highlightbackground=COLOR_BORDER, highlightthickness=1)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        header_r = tk.Frame(right_card, bg=COLOR_CARD)
        header_r.pack(fill=tk.X)
        tk.Label(header_r, text="PROCESSED DOCUMENT", font=FONT_LABEL, fg=COLOR_NAVY, bg=COLOR_CARD).pack(side=tk.LEFT)
        self.lbl_right_info = tk.Label(header_r, text="Awaiting processing step", font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD)
        self.lbl_right_info.pack(side=tk.RIGHT)

        self.lbl_right_image = tk.Label(right_card, text="\n\nProcessed output will appear here\n\n",
                                        font=FONT_BODY, bg="#f1f5f9", fg=COLOR_MUTED, relief=tk.FLAT)
        self.lbl_right_image.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    # -----------------------------------------------------------------------
    # Histogram Section: Image Histogram Analysis
    # -----------------------------------------------------------------------

    def _build_histogram_section(self):
        hist_card = tk.Frame(self.main_container, bg=COLOR_CARD, padx=14, pady=10,
                             highlightbackground=COLOR_BORDER, highlightthickness=1)
        hist_card.pack(fill=tk.X, padx=14, pady=6)

        title_box = tk.Frame(hist_card, bg=COLOR_CARD)
        title_box.pack(fill=tk.X)
        tk.Label(title_box, text="IMAGE HISTOGRAM ANALYSIS (cv2.calcHist)",
                 font=FONT_LABEL, fg=COLOR_NAVY, bg=COLOR_CARD).pack(side=tk.LEFT)
        tk.Label(title_box, text="X-axis: Intensity (0 = Black, 255 = White) | Y-axis: Pixel Frequency",
                 font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD).pack(side=tk.RIGHT)

        # Embedded Matplotlib Figure
        self.fig = Figure(figsize=(10.5, 2.5), dpi=100, facecolor="#ffffff")
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_proc = self.fig.add_subplot(122)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.22, wspace=0.22)

        self._style_histogram_axis(self.ax_orig, "Original Document Intensity Histogram")
        self._style_histogram_axis(self.ax_proc, "Processed Document Intensity Histogram")

        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=hist_card)
        self.hist_canvas.get_tk_widget().pack(fill=tk.X, expand=True, pady=(4, 2))
        self.hist_canvas.draw()

        # Brief educational note beneath graphs
        note_text = (
            "Observation Guide: Aged/yellowed paper concentrates pixels in middle greys (values 70-160). "
            "After enhancement and adaptive thresholding, the histogram becomes sharply bimodal with peaks at 0 (ink) and 255 (paper)."
        )
        tk.Label(hist_card, text=note_text, font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD).pack(anchor="w", pady=(2, 0))

    def _style_histogram_axis(self, ax, title):
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=COLOR_NAVY, pad=6)
        ax.set_xlim([0, 255])
        ax.set_xlabel("Pixel Intensity (0-255)", fontsize=8, color=COLOR_TEXT)
        ax.set_ylabel("Pixel Count", fontsize=8, color=COLOR_TEXT)
        ax.tick_params(axis="both", which="major", labelsize=7.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_facecolor("#fafafa")

    def update_histograms(self, original_img, processed_img=None):
        """Redraw both original and processed histograms."""
        self.ax_orig.clear()
        self.ax_proc.clear()

        self._style_histogram_axis(self.ax_orig, "Original Document Intensity Histogram")
        self._style_histogram_axis(self.ax_proc, "Processed Document Intensity Histogram")

        if original_img is not None:
            hist_orig = ip.compute_histogram(original_img)
            self.ax_orig.fill_between(range(256), hist_orig, color="#2563eb", alpha=0.35)
            self.ax_orig.plot(range(256), hist_orig, color="#1d4ed8", linewidth=1.2)
            mean_orig = float(np.mean(ip.to_grayscale(original_img)))
            self.ax_orig.axvline(mean_orig, color="#dc2626", linestyle=":", linewidth=1, label=f"Mean: {mean_orig:.1f}")
            self.ax_orig.legend(loc="upper right", fontsize=7.5)

        if processed_img is not None:
            hist_proc = ip.compute_histogram(processed_img)
            self.ax_proc.fill_between(range(256), hist_proc, color="#059669", alpha=0.35)
            self.ax_proc.plot(range(256), hist_proc, color="#047857", linewidth=1.2)
            mean_proc = float(np.mean(ip.to_grayscale(processed_img)))
            self.ax_proc.axvline(mean_proc, color="#dc2626", linestyle=":", linewidth=1, label=f"Mean: {mean_proc:.1f}")
            self.ax_proc.legend(loc="upper right", fontsize=7.5)
        else:
            self.ax_proc.text(128, 0.5, "Awaiting Processed Image", horizontalalignment="center",
                              verticalalignment="center", transform=self.ax_proc.transAxes,
                              fontsize=9, color=COLOR_MUTED)

        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.22, wspace=0.22)
        self.hist_canvas.draw()

    # -----------------------------------------------------------------------
    # Educational Explanation Section
    # -----------------------------------------------------------------------

    def _build_explanation_section(self):
        exp_card = tk.LabelFrame(self.main_container, text=" HOW DOES THE IMAGE PROCESSING WORK? ",
                                 font=FONT_LABEL, fg=COLOR_NAVY, bg=COLOR_CARD, padx=12, pady=8,
                                 highlightbackground=COLOR_BORDER, highlightthickness=1)
        exp_card.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)

        text_container = tk.Frame(exp_card, bg=COLOR_CARD)
        text_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_explanation = tk.Text(text_container, height=14, wrap=tk.WORD,
                                       font=FONT_CODE, bg="#ffffff", fg=COLOR_TEXT,
                                       yscrollcommand=scrollbar.set, relief=tk.FLAT,
                                       padx=10, pady=8)
        self.txt_explanation.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.txt_explanation.yview)

    def _init_explanation(self):
        welcome_text = (
            "=" * 78 + "\n"
            "  SMARTDOC: DIGITAL IMAGE PROCESSING & TRADITIONAL OCR ARCHITECTURE\n"
            "=" * 78 + "\n\n"
            "This application demonstrates the complete classical pipeline for restoring old documents:\n\n"
            "  1. BORDER DETECTION       : Locate document page corners using Canny edge gradients\n"
            "                              and Ramer-Douglas-Peucker contour polygon approximation.\n"
            "  2. PERSPECTIVE WARP       : Rectify perspective distortion using a 3x3 homography matrix H.\n"
            "  3. OLD DOCUMENT ENHANCE   : 6-stage DIP pipeline (Grayscale -> Background Normalization ->\n"
            "                              Median Noise Reduction -> CLAHE -> Adaptive Thresholding).\n"
            "  4. TEXT EXTRACTION (OCR)  : Traditional offline Tesseract OCR (--oem 3 --psm 6).\n\n"
            "Click 'Upload Document Photo' to load an image. When you execute any of the four features,\n"
            "this panel will automatically display its complete mathematical formulas, OpenCV parameters,\n"
            "actual measured image statistics, and a faculty presentation explanation."
        )
        self._set_explanation_text(welcome_text)

    def _set_explanation_text(self, text):
        self.txt_explanation.config(state=tk.NORMAL)
        self.txt_explanation.delete("1.0", tk.END)
        self.txt_explanation.insert(tk.END, text)
        self.txt_explanation.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # Bottom Action Bar & Status Bar
    # -----------------------------------------------------------------------

    def _build_bottom_controls(self):
        bottom_bar = tk.Frame(self.main_container, bg=COLOR_CARD, padx=14, pady=10,
                              highlightbackground=COLOR_BORDER, highlightthickness=1)
        bottom_bar.pack(fill=tk.X, padx=14, pady=(6, 4))

        # Left controls: History, Undo, Reset
        btn_hist = tk.Button(bottom_bar, text="📜 Processing History", font=FONT_BODY, bg="#e2e8f0", fg=COLOR_NAVY,
                             relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=self.open_history_dialog)
        btn_hist.pack(side=tk.LEFT, padx=(0, 6))

        btn_undo = tk.Button(bottom_bar, text="↩ Undo Step", font=FONT_BODY, bg="#e2e8f0", fg=COLOR_NAVY,
                             relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=self.undo_step)
        btn_undo.pack(side=tk.LEFT, padx=6)

        btn_reset = tk.Button(bottom_bar, text="🔄 Reset to Original", font=FONT_BODY, bg="#fee2e2", fg="#991b1b",
                              relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=self.reset_all)
        btn_reset.pack(side=tk.LEFT, padx=6)

        # Right controls: Save Processed Image, Save Extracted Text
        btn_save_img = tk.Button(bottom_bar, text="💾 Save Processed Image", font=FONT_BODY, bg=COLOR_BLUE, fg="white",
                                 relief=tk.FLAT, padx=12, pady=5, cursor="hand2", command=self.save_processed_image)
        btn_save_img.pack(side=tk.RIGHT, padx=6)

        btn_save_txt = tk.Button(bottom_bar, text="📄 Save Extracted Text", font=FONT_BODY, bg=COLOR_GREEN, fg="white",
                                 relief=tk.FLAT, padx=12, pady=5, cursor="hand2", command=self.save_text_dialog)
        btn_save_txt.pack(side=tk.RIGHT, padx=6)

    def _build_status_bar(self):
        self.status_bar = tk.Label(self.main_container, text="Ready", font=FONT_SMALL,
                                   bg="#0f172a", fg="#ffffff", anchor="w", padx=12, pady=4)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def set_status(self, message):
        self.status_bar.config(text=f"  {message}")

    # -----------------------------------------------------------------------
    # Image Upload & Management
    # -----------------------------------------------------------------------

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            initialdir=INPUT_DIR,
            title="Select Old Document Photo",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Image Load Error", f"Could not decode image at:\n{file_path}")
            return

        self.original_image = image
        self.current_processed = image.copy()
        self.processing_stack = []
        self.history_records = []
        self.detected_corners = None
        self.manual_points = []
        self.manual_selection_mode = False
        self.pipeline_stages = []
        self.last_extracted_text = ""

        self.stage_box.config(values=["Awaiting Enhance Old Document"])
        self.stage_var.set("Awaiting Enhance Old Document")

        self._show_original(self.original_image)
        self._show_processed(self.original_image, "Original Uploaded Image")
        self.update_histograms(self.original_image, None)

        filename = os.path.basename(file_path)
        self.set_status(f"Loaded '{filename}' ({describe_image(image)}). Proceed with Step 1: Detect Document Border.")
        self._init_explanation()

    def _show_original(self, image_to_render):
        if image_to_render is None:
            return
        photo, scale = to_photo(image_to_render)
        self.left_photo = photo
        self.left_display_scale = scale
        self.lbl_left_image.config(image=photo, text="")
        self.lbl_left_info.config(text=describe_image(self.original_image))

    def _show_processed(self, image_to_render, label_desc=""):
        if image_to_render is None:
            return
        photo, _scale = to_photo(image_to_render)
        self.right_photo = photo
        self.lbl_right_image.config(image=photo, text="")
        self.lbl_right_info.config(text=f"{label_desc} ({describe_image(image_to_render)})")

    # -----------------------------------------------------------------------
    # FEATURE 1: Document Border Detection
    # -----------------------------------------------------------------------

    def feature1_detect_border(self):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please upload a document image first.")
            return

        start_time = time.perf_counter()
        corners = ds.find_document_corners(self.original_image)
        elapsed = time.perf_counter() - start_time

        if corners is not None:
            self.detected_corners = corners
            boundary_preview = ds.draw_boundary(self.original_image, corners)
            self._show_original(boundary_preview)

            measured_data = self._gather_measured_data(
                feature_name=ae.FEATURE_BORDER,
                before_img=self.original_image,
                after_img=self.original_image,
                corners=corners,
                status_note="Four corners detected automatically via Canny edge detection & approxPolyDP"
            )

            exp_text = ae.format_full_explanation(ae.FEATURE_BORDER, measured_data)
            self._set_explanation_text(exp_text)

            self.history_records.append({
                "step": len(self.history_records) + 1,
                "feature": "1. Border Detection (Auto)",
                "input_size": describe_image(self.original_image),
                "output_size": describe_image(self.original_image),
                "time": f"{elapsed:.3f} s",
                "notes": "4 corners identified automatically"
            })

            self.set_status("Border detected successfully! Review corners on Left image. Proceed with Step 2: Correct Perspective.")
        else:
            messagebox.showwarning(
                "Automatic Detection Inconclusive",
                "Automatic boundary detection could not find a clear four-corner quadrilateral.\n\n"
                "Switching to MANUAL CORNER SELECTION:\n"
                "Please click the 4 corners directly on the LEFT image in order:\n"
                "1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left"
            )
            self.toggle_manual_corners(force_enable=True)

    def toggle_manual_corners(self, force_enable=None):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please upload an image first.")
            return

        if force_enable is not None:
            self.manual_selection_mode = force_enable
        else:
            self.manual_selection_mode = not self.manual_selection_mode

        self.manual_points = []
        if self.manual_selection_mode:
            self.btn_manual_corner.config(bg="#f59e0b", fg="white", text="Selecting Corners...")
            self._show_original(self.original_image)
            self.set_status("MANUAL MODE: Click the 4 corners on the LEFT picture (Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left).")
        else:
            self.btn_manual_corner.config(bg="#e2e8f0", fg=COLOR_NAVY, text="Manual Corners")
            self.set_status("Manual corner selection cancelled.")

    def on_left_image_click(self, event):
        if not self.manual_selection_mode or self.original_image is None:
            return

        h, w = self.original_image.shape[:2]
        orig_x = min(max(event.x / self.left_display_scale, 0), w - 1)
        orig_y = min(max(event.y / self.left_display_scale, 0), h - 1)

        self.manual_points.append((orig_x, orig_y))
        count = len(self.manual_points)

        # Draw current clicked points
        preview = self.original_image.copy()
        for idx, (px, py) in enumerate(self.manual_points):
            cv2.circle(preview, (int(px), int(py)), 9, (0, 0, 255), -1)
            cv2.putText(preview, f"{idx+1}", (int(px)+12, int(py)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if count > 1:
            pts_arr = np.array(self.manual_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts_arr], count == 4, (0, 255, 0), 2)

        self._show_original(preview)

        if count < 4:
            labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            self.set_status(f"Recorded corner {count} of 4 ({labels[count-1]}). Next: click {labels[count]}.")
        else:
            self.manual_selection_mode = False
            self.btn_manual_corner.config(bg="#e2e8f0", fg=COLOR_NAVY, text="Manual Corners")
            ordered = ds.order_points(self.manual_points)
            self.detected_corners = ordered

            boundary_preview = ds.draw_boundary(self.original_image, ordered)
            self._show_original(boundary_preview)

            measured_data = self._gather_measured_data(
                feature_name=ae.FEATURE_BORDER,
                before_img=self.original_image,
                after_img=self.original_image,
                corners=ordered,
                status_note="Four corners selected manually by user"
            )
            exp_text = ae.format_full_explanation(ae.FEATURE_BORDER, measured_data)
            self._set_explanation_text(exp_text)

            self.history_records.append({
                "step": len(self.history_records) + 1,
                "feature": "1. Border Detection (Manual)",
                "input_size": describe_image(self.original_image),
                "output_size": describe_image(self.original_image),
                "time": "Manual",
                "notes": "4 corners manually chosen"
            })
            self.set_status("Manual corners confirmed! Proceed with Step 2: Correct Perspective.")

    # -----------------------------------------------------------------------
    # FEATURE 2: Perspective Transformation
    # -----------------------------------------------------------------------

    def feature2_correct_perspective(self):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please upload a document image first.")
            return

        corners_to_use = self.detected_corners
        if corners_to_use is None:
            # Try auto detection if user jumped directly to Step 2
            corners_to_use = ds.find_document_corners(self.original_image)
            if corners_to_use is None:
                messagebox.showerror(
                    "Corners Required",
                    "Please detect or manually select the 4 document corners in Step 1 first."
                )
                return
            self.detected_corners = corners_to_use

        start_time = time.perf_counter()
        try:
            corrected, boundary_prev, used_corners, matrix_h = ds.correct_perspective(
                self.original_image, corners_to_use
            )
        except Exception as err:
            messagebox.showerror("Perspective Correction Error", str(err))
            return
        elapsed = time.perf_counter() - start_time

        # Save previous state to stack for Undo
        self.processing_stack.append((self.current_processed.copy(), "Perspective Transformation", None))
        self.current_processed = corrected
        self.homography_matrix = matrix_h

        # Update visuals
        self._show_original(boundary_prev)
        self._show_processed(corrected, "Perspective-Corrected Document")
        self.update_histograms(self.original_image, corrected)

        measured_data = self._gather_measured_data(
            feature_name=ae.FEATURE_PERSPECTIVE,
            before_img=self.original_image,
            after_img=corrected,
            corners=used_corners,
            homography=matrix_h,
            dest_size=(corrected.shape[1], corrected.shape[0]),
            status_note="Rectified perspective distortion via 3x3 homography matrix warping"
        )
        exp_text = ae.format_full_explanation(ae.FEATURE_PERSPECTIVE, measured_data)
        self._set_explanation_text(exp_text)

        self.history_records.append({
            "step": len(self.history_records) + 1,
            "feature": "2. Perspective Rectification",
            "input_size": describe_image(self.original_image),
            "output_size": describe_image(corrected),
            "time": f"{elapsed:.3f} s",
            "notes": f"Warped to {corrected.shape[1]}x{corrected.shape[0]} px rectangle"
        })

        self.set_status("Perspective transformation complete! Document flattened. Proceed with Step 3: Enhance Old Document.")

    # -----------------------------------------------------------------------
    # FEATURE 3: Old Document Enhancement
    # -----------------------------------------------------------------------

    def feature3_enhance_document(self):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please upload a document image first.")
            return

        input_for_enhance = self.current_processed if self.current_processed is not None else self.original_image

        start_time = time.perf_counter()
        try:
            enhanced, stages = ip.enhance_old_document(
                input_for_enhance,
                bg_blur_size=51,
                noise_kernel=3,
                clip_limit=2.0,
                tile_grid=8,
                block_size=15,
                c_value=10
            )
        except Exception as err:
            messagebox.showerror("Enhancement Error", f"Old document enhancement failed:\n{err}")
            return
        elapsed = time.perf_counter() - start_time

        # Save previous state to stack for Undo
        self.processing_stack.append((self.current_processed.copy(), "Old Document Enhancement", None))
        self.current_processed = enhanced
        self.pipeline_stages = stages

        # Populate stage combobox
        stage_names = ["Final Enhanced Document"] + [s_name for s_name, _ in stages]
        self.stage_box.config(values=stage_names)
        self.stage_var.set("Final Enhanced Document")

        # Update visuals
        self._show_processed(enhanced, "Enhanced Document (Clean Scan)")
        self.update_histograms(self.original_image, enhanced)

        measured_data = self._gather_measured_data(
            feature_name=ae.FEATURE_ENHANCE,
            before_img=input_for_enhance,
            after_img=enhanced,
            status_note="Completed 6-step DIP enhancement: Norm + Denoise + CLAHE + Adaptive Threshold"
        )
        exp_text = ae.format_full_explanation(ae.FEATURE_ENHANCE, measured_data)
        self._set_explanation_text(exp_text)

        self.history_records.append({
            "step": len(self.history_records) + 1,
            "feature": "3. Old Document Enhancement",
            "input_size": describe_image(input_for_enhance),
            "output_size": describe_image(enhanced),
            "time": f"{elapsed:.3f} s",
            "notes": "6-stage DIP pipeline executed"
        })

        # Automatically save enhanced image to default output path
        try:
            cv2.imwrite(DEFAULT_SCANNED_DOC, enhanced)
        except Exception:
            pass

        self.set_status("Enhancement complete! Document cleaned. Proceed with Step 4: Extract Text (OCR) or Save Image.")

    def on_stage_selected(self, _event=None):
        selected = self.stage_var.get()
        if not self.pipeline_stages:
            return

        if selected == "Final Enhanced Document":
            self._show_processed(self.current_processed, "Final Enhanced Document")
            self.update_histograms(self.original_image, self.current_processed)
            return

        for name, img in self.pipeline_stages:
            if name == selected:
                self._show_processed(img, f"Stage: {name}")
                self.update_histograms(self.original_image, img)
                self.set_status(f"Displaying intermediate DIP stage: {name}")
                break

    # -----------------------------------------------------------------------
    # FEATURE 4: Text Extraction (OCR)
    # -----------------------------------------------------------------------

    def feature4_extract_text(self):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please upload and process a document image first.")
            return

        image_to_ocr = self.current_processed.copy()
        self.set_status("Running Tesseract OCR engine (--oem 3 --psm 6)... please wait.")
        self.btn_ocr.config(state=tk.DISABLED, text="Extracting...")

        def ocr_worker():
            try:
                text = ocr.extract_text(image_to_ocr)
                self.ocr_queue.put(("SUCCESS", text))
            except Exception as err:
                self.ocr_queue.put(("ERROR", str(err)))

        threading.Thread(target=ocr_worker, daemon=True).start()
        self.root.after(100, self._poll_ocr_thread)

    def _poll_ocr_thread(self):
        try:
            status, payload = self.ocr_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_ocr_thread)
            return

        self.btn_ocr.config(state=tk.NORMAL, text="Extract Text (OCR)")

        if status == "ERROR":
            messagebox.showerror("OCR Extraction Error", payload)
            self.set_status("OCR failed. Please check Tesseract installation.")
            return

        self.last_extracted_text = payload
        char_count = len(payload)

        # Save extracted text to output/output.txt
        try:
            saved_path = ocr.save_text(payload, DEFAULT_TEXT_DOC)
            save_msg = f"Saved to {saved_path}"
        except Exception as save_err:
            save_msg = f"Save error: {save_err}"

        measured_data = self._gather_measured_data(
            feature_name=ae.FEATURE_OCR,
            before_img=self.current_processed,
            after_img=self.current_processed,
            ocr_char_count=char_count,
            status_note=f"Tesseract recognized {char_count} characters. {save_msg}"
        )
        exp_text = ae.format_full_explanation(ae.FEATURE_OCR, measured_data)
        self._set_explanation_text(exp_text)

        self.history_records.append({
            "step": len(self.history_records) + 1,
            "feature": "4. Text Extraction (OCR)",
            "input_size": describe_image(self.current_processed),
            "output_size": f"{char_count} chars",
            "time": "Async",
            "notes": f"Saved to {DEFAULT_TEXT_DOC}"
        })

        self.open_text_dialog()
        self.set_status(f"OCR Complete! {char_count} characters extracted and saved to {DEFAULT_TEXT_DOC}.")

    def open_text_dialog(self):
        if not self.last_extracted_text:
            messagebox.showinfo("No Text", "No text has been extracted yet. Click 'Extract Text (OCR)' first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Extracted Document Text — Tesseract OCR")
        win.geometry("780x540")
        win.configure(bg=COLOR_BG)

        # Header
        top_frame = tk.Frame(win, bg=COLOR_CARD, padx=14, pady=10, highlightbackground=COLOR_BORDER, highlightthickness=1)
        top_frame.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(top_frame, text="EXTRACTED TEXT CONTENT", font=FONT_LABEL, fg=COLOR_NAVY, bg=COLOR_CARD).pack(side=tk.LEFT)
        char_lbl = tk.Label(top_frame, text=f"Total Characters: {len(self.last_extracted_text)}",
                            font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD)
        char_lbl.pack(side=tk.RIGHT)

        # Text box
        text_frame = tk.Frame(win, bg=COLOR_CARD)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        scroller = ttk.Scrollbar(text_frame)
        scroller.pack(side=tk.RIGHT, fill=tk.Y)

        txt_box = tk.Text(text_frame, wrap=tk.WORD, font=FONT_CODE, bg="#ffffff", fg=COLOR_TEXT,
                          yscrollcommand=scroller.set, padx=10, pady=8)
        txt_box.insert(tk.END, self.last_extracted_text)
        txt_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroller.config(command=txt_box.yview)

        # Actions
        btn_frame = tk.Frame(win, bg=COLOR_BG, padx=10, pady=6)
        btn_frame.pack(fill=tk.X)

        tk.Button(btn_frame, text="Save to File...", font=FONT_BODY, bg=COLOR_GREEN, fg="white",
                  relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
                  command=lambda: self.save_text_dialog(self.last_extracted_text)).pack(side=tk.RIGHT, padx=4)
        tk.Button(btn_frame, text="Close", font=FONT_BODY, bg="#e2e8f0", fg=COLOR_NAVY,
                  relief=tk.FLAT, padx=10, pady=4, cursor="hand2",
                  command=win.destroy).pack(side=tk.RIGHT, padx=4)

    # -----------------------------------------------------------------------
    # Empirical Measurements Helper
    # -----------------------------------------------------------------------

    def _gather_measured_data(self, feature_name, before_img, after_img,
                              corners=None, homography=None, dest_size=None,
                              ocr_char_count=None, status_note=""):
        data = {
            "feature": feature_name,
            "status_note": status_note,
            "image_size": (before_img.shape[1], before_img.shape[0]) if before_img is not None else (0, 0),
            "stats_before": ip.compute_statistics(before_img) if before_img is not None else None,
            "stats_after": ip.compute_statistics(after_img) if after_img is not None else None,
            "patch_before": ip.center_patch(before_img, size=3) if before_img is not None else None,
            "patch_after": ip.center_patch(after_img, size=3) if after_img is not None else None,
            "corners": corners,
            "homography_matrix": homography,
            "dest_size": dest_size,
        }
        if ocr_char_count is not None:
            data["ocr_char_count"] = ocr_char_count
        return data

    # -----------------------------------------------------------------------
    # History, Undo, Reset, and Saving
    # -----------------------------------------------------------------------

    def undo_step(self):
        if not self.processing_stack:
            messagebox.showinfo("Nothing to Undo", "No processing steps have been applied yet.")
            return

        prev_img, step_name, _extra = self.processing_stack.pop()
        self.current_processed = prev_img

        self._show_processed(prev_img, f"Restored previous: {step_name}")
        self.update_histograms(self.original_image, prev_img)

        if self.history_records:
            self.history_records.pop()

        self.set_status(f"Undid step: '{step_name}'. Restored previous image state.")

    def reset_all(self):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Upload an image first.")
            return

        if not messagebox.askyesno("Reset Application", "Reset document back to original upload?"):
            return

        self.current_processed = self.original_image.copy()
        self.processing_stack = []
        self.detected_corners = None
        self.manual_points = []
        self.manual_selection_mode = False
        self.pipeline_stages = []
        self.last_extracted_text = ""

        self.btn_manual_corner.config(bg="#e2e8f0", fg=COLOR_NAVY, text="Manual Corners")
        self.stage_box.config(values=["Awaiting Enhance Old Document"])
        self.stage_var.set("Awaiting Enhance Old Document")

        self._show_original(self.original_image)
        self._show_processed(self.original_image, "Original Uploaded Image")
        self.update_histograms(self.original_image, None)

        self._init_explanation()
        self.set_status("Reset complete. Original document image restored.")

    def open_history_dialog(self):
        if not self.history_records:
            messagebox.showinfo("Processing History", "No processing operations have been performed yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Processing History & Execution Log")
        win.geometry("900x380")
        win.configure(bg=COLOR_BG)

        cols = ("step", "feature", "input_size", "output_size", "time", "notes")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=12)

        tree.heading("step", text="Step #")
        tree.column("step", width=60, anchor="center")

        tree.heading("feature", text="Feature / Operation")
        tree.column("feature", width=220, anchor="w")

        tree.heading("input_size", text="Input Dimensions")
        tree.column("input_size", width=140, anchor="w")

        tree.heading("output_size", text="Output Dimensions")
        tree.column("output_size", width=140, anchor="w")

        tree.heading("time", text="Exec Time")
        tree.column("time", width=90, anchor="center")

        tree.heading("notes", text="Operation Notes")
        tree.column("notes", width=230, anchor="w")

        for rec in self.history_records:
            tree.insert("", tk.END, values=(rec["step"], rec["feature"], rec["input_size"],
                                            rec["output_size"], rec["time"], rec["notes"]))

        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def save_processed_image(self):
        if self.current_processed is None:
            messagebox.showinfo("No Image", "There is no processed image to save.")
            return

        save_path = filedialog.asksaveasfilename(
            initialdir=OUTPUT_DIR,
            initialfile="scanned_doc.png",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("TIFF Image", "*.tif")]
        )
        if not save_path:
            return

        success = cv2.imwrite(save_path, self.current_processed)
        if success:
            messagebox.showinfo("Save Successful", f"Processed document image saved to:\n{save_path}")
            self.set_status(f"Saved processed image to {save_path}")
        else:
            messagebox.showerror("Save Failed", "Could not write image to the specified path.")

    def save_text_dialog(self, text=None):
        content = text if text is not None else self.last_extracted_text
        if not content:
            messagebox.showinfo("No Text", "No text has been extracted yet. Run 'Extract Text (OCR)' first.")
            return

        save_path = filedialog.asksaveasfilename(
            initialdir=OUTPUT_DIR,
            initialfile="output.txt",
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            ocr.save_text(content, save_path)
            messagebox.showinfo("Save Successful", f"Extracted text saved to:\n{save_path}")
            self.set_status(f"Saved extracted text to {save_path}")
        except Exception as err:
            messagebox.showerror("Save Failed", f"Could not write text file:\n{err}")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    SmartDocApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
