# 🚗 Egyptian License Plate Recognition (ALPR) System

> **A MATLAB-based Automatic License Plate Recognition system engineered specifically for Egyptian license plates, featuring HSV color segmentation, Sobel edge detection, morphological filtering, HOG-based OCR, and Arabic character recognition.**

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Egyptian License Plate Structure](#2-egyptian-license-plate-structure)
3. [Key Features](#3-key-features)
4. [System Architecture & Pipeline Workflow](#4-system-architecture--pipeline-workflow)
5. [Project Structure](#5-project-structure)
6. [Requirements & Installation](#6-requirements--installation)
7. [How to Run](#7-how-to-run)
8. [Pipeline Explanation — Stage by Stage](#8-pipeline-explanation--stage-by-stage)
   - [Stage 1 — Dataset Loading & Path Setup](#stage-1--dataset-loading--path-setup)
   - [Stage 2 — Template Loading](#stage-2--template-loading)
   - [Stage 3 — Image Standardization & Resizing](#stage-3--image-standardization--resizing)
   - [Stage 4 — Grayscale & HSV Conversion](#stage-4--grayscale--hsv-conversion)
   - [Stage 5 — Median Filtering](#stage-5--median-filtering)
   - [Stage 6 — HSV Color Mask Generation](#stage-6--hsv-color-mask-generation)
   - [Stage 7 — Sobel Edge Detection](#stage-7--sobel-edge-detection)
   - [Stage 8 — Morphological Operations](#stage-8--morphological-operations)
   - [Stage 9 — Connected Component Analysis & Large Region Filtering](#stage-9--connected-component-analysis--large-region-filtering)
   - [Stage 10 — Candidate Region Scoring](#stage-10--candidate-region-scoring)
   - [Stage 11 — Fallback: Blue-Band Expansion](#stage-11--fallback-blue-band-expansion)
   - [Stage 12 — Plate Cropping](#stage-12--plate-cropping)
   - [Stage 13 — Adaptive Plate Binarization](#stage-13--adaptive-plate-binarization)
   - [Stage 14 — Character Segmentation](#stage-14--character-segmentation)
   - [Stage 15 — Numbers / Letters Separation](#stage-15--numbers--letters-separation)
   - [Stage 16 — HOG Feature Extraction & OCR Matching](#stage-16--hog-feature-extraction--ocr-matching)
   - [Stage 17 — Confidence Validation & String Formatting](#stage-17--confidence-validation--string-formatting)
   - [Stage 18 — Visualization Export](#stage-18--visualization-export)
   - [Stage 19 — Excel Export](#stage-19--excel-export)
9. [Function Reference](#9-function-reference)
   - [match\_character()](#match_character)
   - [load\_real\_templates()](#load_real_templates)
   - [robust\_binarize()](#robust_binarize)
   - [remove\_large\_blobs()](#remove_large_blobs)
   - [count\_char\_blobs()](#count_char_blobs)
   - [segment\_characters()](#segment_characters)
10. [OCR Methodology](#10-ocr-methodology)
11. [Detection Scoring Logic](#11-detection-scoring-logic)
12. [Visualization Outputs](#12-visualization-outputs)
13. [Result Export](#13-result-export)
14. [Variable & Threshold Reference](#14-variable--threshold-reference)
15. [Sample Console Output](#15-sample-console-output)
16. [Algorithm Strengths](#16-algorithm-strengths)
17. [Limitations](#17-limitations)
18. [Future Improvements](#18-future-improvements)
19. [Academic Context](#19-academic-context)
20. [References](#20-references)
21. [team](#21-team)
22. [Conclusion](#22-conclusion)

---

## 1. Project Overview

This project implements a **complete, end-to-end Automatic License Plate Recognition (ALPR) pipeline** targeting Egyptian vehicle license plates. Unlike generic ALPR systems, it is purpose-built to handle the unique visual characteristics of Egyptian plates: blue top strip, white background, mixed Arabic letters and Arabic-Indic numerals, and the "مصر / EGYPT" bilingual header.

The system operates on raw vehicle images without any neural network training data. It relies entirely on **classical image processing** and **feature-based machine learning** (HOG descriptors with Euclidean nearest-neighbor matching), making it lightweight, interpretable, and deployable without a GPU.

### What the system does end-to-end:

```
Input Image (JPG)
      ↓
Image Standardization
      ↓
HSV Color Segmentation (Blue + White Masks)
      ↓
Sobel Edge Detection
      ↓
Morphological Filtering
      ↓
Candidate Region Scoring
      ↓
Plate Localization (with Fallback)
      ↓
Adaptive Binarization
      ↓
Character Segmentation (Numbers + Letters)
      ↓
HOG Feature Extraction
      ↓
Template Matching (Arabic OCR)
      ↓
Recognized Plate String (Arabic Numerals + Letters)
      ↓
Visualization PNG + Excel Export
```

---

## 2. Egyptian License Plate Structure

Understanding the physical structure of Egyptian plates is critical for appreciating every design decision in this system.

### Physical Layout

```
╔══════════════════════════════════════════╗
║  [BLUE STRIP]   مصر / EGYPT             ║
╠══════════════════════════════════════════╣
║                                          ║
║   [Arabic Letters]  |  [Arabic Numbers] ║
║   (Right side)      |  (Left side)      ║
║                                          ║
╚══════════════════════════════════════════╝
```

### Plate Characteristics Summary

| Property | Detail |
|---|---|
| **Primary Color** | White background with blue top strip |
| **Top Strip** | Contains "مصر" (Egypt in Arabic) and "EGYPT" (Latin) |
| **Number Zone** | Arabic-Indic numerals (٠–٩), 3–4 digits |
| **Letter Zone** | Arabic letters (ا ب ت ث ج ...), 2–3 letters |
| **Reading Direction** | Right-to-left (Arabic) — letters on right, numbers on left |
| **Plate Aspect Ratio** | Approximately 2.5:1 to 4.5:1 (width:height) |
| **Dimensions (typical)** | ~52 cm × 11 cm on vehicle |

### Arabic Letter Set Used on Egyptian Plates

| Latin Key | Arabic Character | Transliteration |
|---|---|---|
| a | ا | Alef |
| b | ب | Ba |
| t | ت | Ta |
| th | ث | Tha |
| g | ج | Geem |
| h | ح | Ha |
| kh | خ | Kha |
| d | د | Dal |
| z | ذ | Zal |
| r | ر | Ra |
| s | س | Sin |
| sh | ش | Shin |
| sd | ص | Sad |
| dd | ض | Dad |
| ta | ط | Ta (emphatic) |
| za | ظ | Za (emphatic) |
| e | ع | Ain |
| gh | غ | Ghain |
| f | ف | Fa |
| q | ق | Qaf |
| k | ك | Kaf |
| l | ل | Lam |
| m | م | Meem |
| n | ن | Noon |
| he | ه | Ha |
| w | و | Waw |
| y | ى | Ya |

### Arabic-Indic Digit Mapping

| Latin | Arabic-Indic |
|---|---|
| 0 | ٠ |
| 1 | ١ |
| 2 | ٢ |
| 3 | ٣ |
| 4 | ٤ |
| 5 | ٥ |
| 6 | ٦ |
| 7 | ٧ |
| 8 | ٨ |
| 9 | ٩ |

---

## 3. Key Features

- **No deep learning required** — fully classical image processing pipeline
- **HSV-based color discrimination** — exploits Egyptian plate's distinct blue strip
- **Dual edge detection** — vertical + horizontal Sobel with threshold relaxation (`tv * 0.7`)
- **Multi-criteria candidate scoring** — aspect ratio, area, edge density, color overlap, position
- **Fallback detection mechanism** — blue-band expansion when primary detection fails
- **Dual polarity adaptive binarization** — handles both dark-on-white and white-on-dark illumination
- **Arabic character support** — full Arabic letter + Arabic-Indic numeral recognition
- **HOG-based OCR** — Histogram of Oriented Gradients template matching
- **Confidence thresholding** — only accepts matches above score `0.1`
- **Spatial character partitioning** — automatic numbers/letters zone separation at 48% of plate width
- **Visualization pipeline** — 8-panel figure saved per image
- **Excel export** — structured results table in `.xlsx` format
- **Robustness to lighting** — CLAHE + `imadjust` + median filtering combination

---

## 4. System Architecture & Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│  dataset/*.jpg  →  Deduplicated file list (case-insensitive)│
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                       │
│  1. Resize to 800px width (preserve aspect ratio)           │
│  2. rgb2gray  →  grayscale image                            │
│  3. rgb2hsv   →  HSV color space                            │
│  4. medfilt2([3×3])  →  salt-and-pepper noise removal       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                COLOR SEGMENTATION LAYER                     │
│  Blue Mask: H∈(0.52,0.75) ∧ S>0.25 ∧ V>0.12               │
│  White Mask: S<0.20 ∧ V>0.60                                │
│  → imclose (rectangle [3×8])                                │
│  → bwareaopen (min area 50 / 100)                           │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   EDGE DETECTION LAYER                      │
│  Sobel vertical   (threshold × 0.7)                         │
│  Sobel horizontal (threshold × 0.7)                         │
│  → bwareaopen (min 10 pixels)                               │
│  → imdilate [3×15]                                          │
│  → imclose  [4×4]                                           │
│  → imfill holes                                             │
│  → bwareaopen (min 300)                                     │
│  → remove blobs > 5% of image area                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   CANDIDATE SCORING LAYER                   │
│  For each connected component:                              │
│    ├─ Aspect ratio filter: 1.5 ≤ AR ≤ 6.0                  │
│    ├─ Area filter: 300 ≤ area ≤ 5% image                    │
│    ├─ Width filter: 3%–35% of image width                   │
│    ├─ Height filter: 1%–18% of image height                 │
│    ├─ Extent filter: ≥ 0.35                                  │
│    ├─ Position filter: center Y > 15% from top              │
│    ├─ Vertical edge density (vd): score += vd × 100         │
│    ├─ V/H edge ratio bonus: if ratio > 0.8                  │
│    ├─ Blue overlap proximity bonus: +15                      │
│    ├─ Blue pixel ratio in bbox: +br × 30                    │
│    ├─ White pixel ratio in bbox: +wr × 10                   │
│    ├─ Y-position bonus: +5 if yn ∈ (0.4, 0.85)             │
│    ├─ Aspect ratio bonus: +3 if AR ∈ (2.0, 5.0)            │
│    └─ Column intensity std deviation bonus: min(std/10, 5)  │
│  → Select highest score candidate                           │
└────────────────────────────┬────────────────────────────────┘
                             │ (if no candidate found)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  FALLBACK DETECTION LAYER                   │
│  Blue-band expansion: find blue regions with AR > 2.0       │
│  → Expand bbox by: x−5%, y−20%, w×110%, h×450%             │
│  → Validate with vertical edge density > 0.02               │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                PLATE EXTRACTION LAYER                       │
│  Crop detected region with ±5px padding                     │
│  → robust_binarize() → dual-polarity adaptive threshold     │
│  → segment_characters() → bounding boxes                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     OCR LAYER                               │
│  Spatial split at 48% plate width → numbers | letters       │
│  For each character:                                        │
│    → imresize to [42×24]                                    │
│    → extractHOGFeatures (CellSize [8×8])                    │
│    → Euclidean distance vs all templates                    │
│    → score = 1/(1+dist)                                     │
│    → Accept if score > 0.1                                  │
│  Cap: max 4 numbers, max 3 letters                          │
│  → flip order (RTL) → format string                         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│  result_<name>.png  →  8-panel visualization figure        │
│  extracted_chars/*.png  →  individual character images      │
│  alpr_results.xlsx  →  Filename | Detected | Plate String  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Project Structure

```
project_root/
│
├── main_alpr.m                   ← Main MATLAB script (entry point)
│
├── dataset/                      ← Input images (.jpg)
│   ├── car001.jpg
│   ├── car002.jpg
│   └── ...
│
└── dataset/results/              ← Auto-generated output directory
    │
    ├── result_car001.png         ← 8-panel visualization per image
    ├── result_car002.png
    ├── ...
    │
    ├── alpr_results.xlsx         ← Summary table (filename, detected, plate)
    │
    ├── templates/                ← Character template images (.png)
    │   ├── 0-1.png               ← Digit "0", sample 1
    │   ├── 0-2.png               ← Digit "0", sample 2
    │   ├── m-1.png               ← Arabic letter Meem (م)
    │   ├── w-1.png               ← Arabic letter Waw (و)
    │   └── ...
    │
    └── extracted_chars/          ← Individual segmented character images
        ├── img_01_num_1.png
        ├── img_01_num_2.png
        ├── img_01_let_1.png
        └── ...
```

### Template Naming Convention

Templates must follow this naming scheme:

```
<key>-<sample_number>.png
```

Where `<key>` corresponds to entries in the template map:

| Key | Arabic Character | Example Filename |
|---|---|---|
| `0` through `9` | ٠ through ٩ | `5-1.png`, `7-2.png` |
| `a` | ا (Alef) | `a-1.png` |
| `m` | م (Meem) | `m-1.png` |
| `w` | و (Waw) | `w-1.png` |
| `r` | ر (Ra) | `r-1.png` |
| `n` | ن (Noon) | `n-1.png` |
| `sh` | ش (Shin) | `sh-1.png` |

> **Note:** Multiple samples per character (e.g., `m-1.png`, `m-2.png`) improve HOG matching accuracy by averaging across appearance variations.

---

## 6. Requirements & Installation

### MATLAB Toolbox Requirements

| Toolbox | Purpose | Functions Used |
|---|---|---|
| **Image Processing Toolbox** | Core image operations | `rgb2gray`, `rgb2hsv`, `medfilt2`, `edge`, `imdilate`, `imclose`, `imfill`, `bwareaopen`, `bwlabel`, `regionprops`, `imcrop`, `imbinarize`, `adapthisteq`, `imadjust`, `graythresh`, `bwconncomp`, `imresize`, `imwrite`, `imread` |
| **Computer Vision Toolbox** | HOG feature extraction | `extractHOGFeatures` |
| **Statistics and Machine Learning Toolbox** *(optional)* | Some auxiliary statistical operations | `std`, `mean` |

> **Minimum MATLAB version:** R2017b or later (for `extractHOGFeatures` with `CellSize` parameter and `adapthisteq`)

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| MATLAB | R2017b | R2021a+ |
| RAM | 4 GB | 8 GB+ |
| Storage | 500 MB (templates + data) | 2 GB+ |
| OS | Windows / macOS / Linux | Any |

### Installation Steps

**Step 1: Clone or download the repository**
```bash
git clone https://github.com/your-username/egyptian-alpr.git
cd egyptian-alpr
```

**Step 2: Verify MATLAB toolboxes are installed**
```matlab
% In MATLAB Command Window:
ver('images')     % Should return Image Processing Toolbox version
ver('vision')     % Should return Computer Vision Toolbox version
```

**Step 3: Prepare your dataset**
```
Place all vehicle images (.jpg) into:
    egyptian-alpr/dataset/
```

**Step 4: Prepare character templates**

Create character template images and place them in:
```
dataset/results/templates/
```

Templates must be:
- Binary (black character on white background) or grayscale
- Named using the key convention: `<key>-<number>.png`
- Minimum size: 10×10 pixels (system resizes to 42×24 internally)

> **Tip:** You can extract templates from the `extracted_chars/` folder after a first run, then manually label and rename them as templates for subsequent runs.

---

## 7. How to Run

**Method 1: From MATLAB Command Window**
```matlab
% Navigate to the project directory
cd('path/to/egyptian-alpr')

% Run the main script
main_alpr
```

**Method 2: From MATLAB Editor**
1. Open `main_alpr.m` in MATLAB Editor
2. Press **F5** or click **Run**

**Method 3: Batch run via MATLAB script**
```matlab
run('path/to/egyptian-alpr/main_alpr.m')
```

### Dataset Folder Selection

- If `../dataset/` exists relative to the script and contains `.jpg` files, it is used automatically.
- Otherwise, a **folder selection dialog** appears so you can navigate to your dataset folder manually.

### Expected Runtime

| Dataset Size | Approximate Time |
|---|---|
| 1 image | 2–5 seconds |
| 10 images | 20–50 seconds |
| 100 images | 3–8 minutes |

*(Times vary by hardware, image resolution, and template library size.)*

---

## 8. Pipeline Explanation — Stage by Stage

---

### Stage 1 — Dataset Loading & Path Setup

```matlab
script_dir = fileparts(mfilename('fullpath'));
dataset_dir = fullfile(script_dir, '..', 'dataset');
```

**Purpose:** Establish all file paths relative to the script's own location, ensuring portability across different machines.

The script searches for a `dataset/` folder one level above the script directory. If not found or empty, it prompts the user with a GUI folder picker (`uigetdir`). A `results/` subfolder is automatically created if it does not exist.

**Deduplication:** Both `.jpg` and `.JPG` extensions are collected and then deduplicated using `unique(lower({all_files.name}))`, preventing double-processing on case-insensitive filesystems.

---

### Stage 2 — Template Loading

```matlab
templates = load_real_templates(templates_dir);
```

**Purpose:** Pre-load all character reference images into memory as HOG-ready binary arrays, structured in a MATLAB struct array.

Each template is loaded once at startup (not per image), which is critical for performance. The struct has two fields per entry:
- `templates(i).char` — the Unicode Arabic character (e.g., `'م'`, `'٦'`)
- `templates(i).img` — a `42×24` logical (binary) array of the character image

See [load\_real\_templates()](#load_real_templates) for full explanation.

---

### Stage 3 — Image Standardization & Resizing

```matlab
img = imresize(img_orig, 800/orig_w);
```

**Purpose:** Normalize all input images to a fixed width of **800 pixels** while preserving the original aspect ratio.

**Why 800px?**
- Provides sufficient resolution for character detection (typical plate becomes ~60–150px wide)
- Prevents excessive computation on high-resolution images (e.g., 4K photos)
- Ensures all subsequent threshold parameters (pixel counts, areas) apply consistently across all inputs

This is equivalent to computing a scale factor `scale = 800 / original_width` and applying it uniformly to both dimensions.

---

### Stage 4 — Grayscale & HSV Conversion

```matlab
gray_img = rgb2gray(img);
hsv_img  = rgb2hsv(img);
```

**Purpose:** Derive two complementary representations of the image for parallel processing streams:

- **Grayscale** (`gray_img`): Used for edge detection, binarization, and local contrast analysis
- **HSV** (`hsv_img`): Used for color-based segmentation of the plate's blue strip and white background

**Why HSV over RGB for color segmentation?**

RGB mixes color and brightness information. In HSV:
- **Hue (H)** — pure color identity (0–1 circular), unaffected by lighting intensity
- **Saturation (S)** — colorfulness vs. gray; low S = gray/white
- **Value (V)** — brightness; separates dark from bright regions

This makes HSV far more robust for detecting "Egyptian plate blue" under varying ambient light conditions — the Hue remains stable even when the plate is in shadow or overexposed.

---

### Stage 5 — Median Filtering

```matlab
filtered_img = medfilt2(gray_img, [3 3]);
```

**Purpose:** Remove impulse noise (salt-and-pepper noise) from the grayscale image before edge detection.

**How it works:** For each 3×3 neighborhood, replace the central pixel with the **median** value of its 9 neighbors. This preserves sharp edges (since the median is resistant to outliers) while eliminating isolated noisy pixels that would otherwise create spurious edges in the Sobel step.

**Why not Gaussian blur?** Gaussian blur is a linear filter that attenuates all high-frequency content, blurring true edges. Median filtering preserves edge sharpness — critical for detecting the clean horizontal and vertical edges characteristic of license plate boundaries.

---

### Stage 6 — HSV Color Mask Generation

#### Blue Mask

```matlab
blue_mask = (hue > 0.52 & hue < 0.75) & (sat > 0.25) & (val > 0.12);
blue_mask = imclose(blue_mask, strel('rectangle', [3 8]));
blue_mask = bwareaopen(blue_mask, 50);
```

| Parameter | Value | Rationale |
|---|---|---|
| Hue range | 0.52 – 0.75 | Spans cyan-blue to blue-violet, covering Egyptian plate blue |
| Saturation min | 0.25 | Excludes gray/white regions which have near-zero saturation |
| Value min | 0.12 | Excludes near-black regions (night scenes, shadows) |
| `imclose` SE | 3×8 rectangle | Closes horizontal gaps between blue stripe segments |
| `bwareaopen` | 50 pixels | Removes tiny blue blobs from reflections or noise |

**Why these Hue bounds?** In HSV (MATLAB's 0–1 normalized scale), the range 0.52–0.75 maps to approximately 187°–270° in a standard 0°–360° scale, capturing the blue hues typical of Egyptian plate backgrounds.

#### White Mask

```matlab
white_mask = (sat < 0.20) & (val > 0.60);
white_mask = imclose(white_mask, strel('rectangle', [3 8]));
white_mask = bwareaopen(white_mask, 100);
```

| Parameter | Value | Rationale |
|---|---|---|
| Saturation max | 0.20 | White/gray pixels have very low saturation |
| Value min | 0.60 | Ensures we capture bright (not dark gray) regions |
| `bwareaopen` | 100 pixels | Larger minimum than blue mask — white plate area should be substantial |

These two masks work **cooperatively** in the scoring stage: a candidate region with both blue and white content is almost certainly a license plate region.

---

### Stage 7 — Sobel Edge Detection

```matlab
[~, tv] = edge(filtered_img, 'sobel', 'vertical');
edge_v  = edge(filtered_img, 'sobel', 'vertical', tv * 0.7);

[~, th] = edge(filtered_img, 'sobel', 'horizontal');
edge_h  = edge(filtered_img, 'sobel', 'horizontal', th * 0.7);
```

**Purpose:** Detect intensity transitions corresponding to character boundaries and plate frame edges.

**Why two calls per direction?**

MATLAB's `edge()` first computes an automatic threshold (`tv`, `th`) using Otsu-like internal logic. By calling it a second time with `threshold * 0.7`, the system deliberately uses a **30% lower threshold** than the automatic estimate. This relaxed threshold:
- Captures weaker edges from faded plates
- Detects character strokes at plate boundaries in low-contrast conditions
- Recovers edges in underexposed images

**Why separate vertical and horizontal edge maps?**

License plates have a characteristic signature:
- **Vertical edges** (`edge_v`): Character strokes (vertical lines of Arabic characters and numerals)
- **Horizontal edges** (`edge_h`): Plate boundary top/bottom borders

The **V/H edge ratio** (`vhr = vd / (hd + 0.001)`) is used as a scoring criterion: a legitimate plate region has significantly more vertical edges (from characters) than horizontal edges (only the border), so `vhr > 0.8` gives a confidence bonus.

---

### Stage 8 — Morphological Operations

```matlab
edge_combined = bwareaopen(edge_v, 10);
dilated  = imdilate(edge_combined, strel('rectangle', [3 15]));
closed   = imclose(dilated,        strel('rectangle', [4 4]));
filled   = imfill(closed, 'holes');
```

**Purpose:** Transform sparse edge pixels into solid filled regions representing candidate plate locations.

| Operation | Structuring Element | Purpose |
|---|---|---|
| `bwareaopen(10)` | — | Remove tiny isolated edge fragments (noise) |
| `imdilate([3×15])` | Wide horizontal rectangle | Bridge horizontal gaps between character edge columns |
| `imclose([4×4])` | Square | Close small gaps introduced by dilation |
| `imfill('holes')` | — | Fill enclosed interior regions to create solid blobs |

**Intuition for the 3×15 dilation:**

Arabic characters and numerals have vertical strokes separated by small gaps. The 3-row × 15-column rectangle dilates each vertical edge segment **15 pixels horizontally**, connecting adjacent character strokes into a single merged blob that spans the whole plate width. The 3-row height prevents vertical merging with non-plate structures.

---

### Stage 9 — Connected Component Analysis & Large Region Filtering

```matlab
max_area  = img_h * img_w * 0.05;
clean_img = bwareaopen(filled, 300);
labeled   = bwlabel(clean_img);
sp        = regionprops(labeled, 'Area', 'PixelIdxList');
for s = 1:length(sp)
    if sp(s).Area > max_area
        clean_img(sp(s).PixelIdxList) = 0;
    end
end
```

**Purpose:** Isolate meaningful candidate blobs and eliminate those that are too large to be license plates.

**Two-pass filtering:**
1. `bwareaopen(300)` — removes blobs smaller than 300 pixels (noise, small reflections)
2. `sp(s).Area > max_area` — removes blobs larger than 5% of total image area (sky, road, entire vehicle body)

**Why 5%?** A license plate occupies at most a few percent of a typical full vehicle image. Any blob larger than 5% of the total frame is definitively not a plate — it is a large uniform region like a car door, window, or road surface.

---

### Stage 10 — Candidate Region Scoring

This is the most complex and important stage. Each surviving blob is evaluated against a **multi-criteria scoring function** to identify the most likely plate region.

#### Filter Gates (Hard Rejections)

| Criterion | Condition | Reason |
|---|---|---|
| Aspect ratio | 1.5 ≤ AR ≤ 6.0 | Plates are always wider than tall; extreme ratios are rejected |
| Area | 300 ≤ area ≤ max_area | Too-small or too-large regions are rejected |
| Width | 3%–35% of image width | Prevents selecting tiny or full-frame regions |
| Height | 1%–18% of image height | Plates are horizontally narrow |
| Extent | ≥ 0.35 | Extent = area/bounding box area; plate blobs should be relatively filled |
| Vertical position | Center Y > 15% from top | Plates are rarely at the very top of the image |
| Edge density | vd ≥ 0.03 | Minimum vertical edge density; blank regions rejected |

#### Score Accumulation

```
score = vd × 100                          (vertical edge density, primary signal)
      + min(vhr, 5) × 2  [if vhr > 0.8]  (vertical-to-horizontal edge ratio)
      + 15               [blue proximity] (nearby blue region above)
      + br × 30          [if 0.05 < br < 0.5]  (blue pixel ratio inside bbox)
      + wr × 10          [if 0.15 < wr < 0.85]  (white pixel ratio inside bbox)
      + 5                [if yn ∈ (0.4, 0.85)]  (typical plate vertical position)
      + 3                [if AR ∈ (2.0, 5.0)]   (ideal plate aspect ratio)
      + min(std/10, 5)                          (column intensity variation)
```

**Why vertical edge density as the primary signal?** Arabic characters and numerals consist of many strokes, creating a high density of vertical edges within the plate bounding box. Background regions (car body, road) have far lower vertical edge densities.

**Why blue proximity bonus (+15)?** The Egyptian plate has a blue strip directly above the character region. If a detected blue region's bottom edge (`byb`) aligns with the top of a candidate region, the system rewards it with +15 points — a very strong signal.

---

### Stage 11 — Fallback: Blue-Band Expansion

```matlab
if ~plate_detected
    for b = 1:length(blue_stats)
        bb2 = blue_stats(b).BoundingBox;
        if bw2/bh2 > 2.0 && bw2 > img_w*0.03 && blue_stats(b).Area > 80
            exp_bb = [bb2(1)-bw2*0.05, bb2(2)-bh2*0.2, bw2*1.1, bh2*4.5];
            ...
            if d2 > 0.02
                best_bbox = exp_bb;
                plate_detected = true;
            end
        end
    end
end
```

**Purpose:** If the primary scoring approach fails to identify any candidate, attempt plate detection by expanding the known blue strip region downward.

**Logic:** The Egyptian plate's blue strip appears directly above the character zone. If a valid blue strip is detected (aspect ratio > 2.0, width > 3% of image, area > 80px), the system expands it:
- Leftward by 5% of strip width
- Upward by 20% of strip height
- Rightward by 110% of strip width (total)
- Downward by 450% of strip height (to encompass the character zone below)

**Validation:** The expanded region must still contain meaningful vertical edge density (`d2 > 0.02`) to be accepted. This prevents false positives from blue reflections or signage.

**When is this needed?** When the character zone edges are very weak (faded plate, extreme angle) but the blue strip is still clearly visible.

---

### Stage 12 — Plate Cropping

```matlab
pad = 5;
pb = [max(1, best_bbox(1)-pad), max(1, best_bbox(2)-pad),
      min(best_bbox(3)+2*pad, img_w-best_bbox(1)+pad),
      min(best_bbox(4)+2*pad, img_h-best_bbox(2)+pad)];
full_plate = imcrop(gray_img, pb);
```

**Purpose:** Extract the detected plate region from the grayscale image with a 5-pixel padding border on all sides.

**Why add padding?** The scoring bounding box might clip character edges at the plate boundary. The 5px pad ensures complete character shapes are included in the crop, improving segmentation accuracy. `max(1, ...)` and `min(..., img_w/img_h)` clamps prevent the crop from going out of bounds.

---

### Stage 13 — Adaptive Plate Binarization

This stage converts the cropped grayscale plate patch into a binary (black and white) image suitable for character segmentation. It is implemented in `robust_binarize()` — see the [full function explanation below](#robust_binarize).

**Key insight:** Egyptian plates appear in two visual polarities depending on illumination:
- **Normal:** Dark characters on white plate background → Dark-on-bright (standard)
- **Night/reflection:** Plate appears dark, characters appear bright (inverted)

The system generates **two candidate binarizations** (dark-foreground and bright-foreground) and selects the one that contains more valid character-shaped blobs.

---

### Stage 14 — Character Segmentation

```matlab
raw_chars = segment_characters(plate_bw);
raw_chars = sortrows(raw_chars, 1, 'ascend');
```

**Purpose:** Identify individual character bounding boxes from the binary plate image.

`segment_characters()` applies `regionprops()` to extract connected component bounding boxes and filters them using geometric constraints. The results are then sorted left-to-right by X position (ascending), producing an ordered sequence of character regions.

See [segment\_characters()](#segment_characters) for full explanation.

---

### Stage 15 — Numbers / Letters Separation

```matlab
separator_x = plate_width * 0.48;

for g = 1:size(raw_chars, 1)
    char_center_x = raw_chars(g,1) + (raw_chars(g,3)/2);
    if char_center_x < separator_x
        numbers_boxes = [numbers_boxes; raw_chars(g,:)];
    else
        letters_boxes = [letters_boxes; raw_chars(g,:)];
    end
end
```

**Purpose:** Divide detected characters into two groups — numbers (left half) and letters (right half) — before OCR matching.

**Why 48% of plate width as the separator?**

On Egyptian plates, the layout places numbers on the left and letters on the right (when read left-to-right in image coordinates, since Arabic is RTL). The 48% threshold gives a slight leftward bias, accommodating plates where the number zone may be slightly wider than 50%.

**Why separate matching modes?** The OCR function `match_character()` receives a boolean flag `is_number_mode` that restricts template comparison to only digit templates (for numbers) or only letter templates (for letters). This prevents a "6" from being confused with a visually similar Arabic letter, dramatically improving accuracy.

---

### Stage 16 — HOG Feature Extraction & OCR Matching

```matlab
char_img = imcrop(plate_bw, numbers_boxes(c,:));
char_img = imresize(char_img, [42 24]);

[best_match, score] = match_character(char_img, templates, true);
if score > 0.1
    matched_numbers{end+1} = best_match;
end
```

**Purpose:** Recognize each character by comparing its HOG feature vector to all templates using Euclidean distance.

**Normalization to [42×24]:** All character crops are resized to a fixed 42×24 pixel canvas before HOG computation. This ensures:
- Scale invariance (all characters produce the same feature vector length)
- Consistent HOG cell boundaries across characters of different original sizes

**Why 42×24?** This is a common HOG window size for character recognition. With `CellSize = [8 8]`, a 42×24 image produces a well-structured HOG grid. The 42:24 aspect ratio (approximately 1.75:1, height:width) matches the typical Arabic character proportions.

See [match\_character()](#match_character) for detailed HOG methodology.

---

### Stage 17 — Confidence Validation & String Formatting

```matlab
% Cap at 4 numbers
if length(matched_numbers) > 4
    [~, sort_idx] = sort(num_scores, 'descend');
    top_idx = sort(sort_idx(1:4));
    matched_numbers = matched_numbers(top_idx);
end

% Cap at 3 letters
if length(matched_letters) > 3
    [~, sort_idx] = sort(let_scores, 'descend');
    top_idx = sort(sort_idx(1:3));
    matched_letters = matched_letters(top_idx);
end

str_num = strjoin(flip(matched_numbers), ' ');
str_let = strjoin(flip(matched_letters), ' ');
recognized_string = strtrim([str_let, '   ', str_num]);
```

**Purpose:** Apply Egyptian plate format constraints and compose the final recognized string.

**Cap logic:** Egyptian plates have at most 4 digits and 3 letters. If more are detected (due to segmentation noise), the system keeps only the **top-scoring** ones by confidence.

**Why `flip()`?** Character boxes were collected left-to-right in image coordinates. But Arabic text reads right-to-left. `flip()` reverses the order so the characters are presented in natural Arabic reading order.

**String format:** Letters and numbers are joined with spaces between each character and three spaces between the groups, mimicking the typical plate layout.

---

### Stage 18 — Visualization Export

```matlab
fh = figure('Visible', 'off');
subplot(2,4,1); imshow(img);           title('1. Standardized');
subplot(2,4,2); imshow(filtered_img);  title('2. Filtered Gray');
subplot(2,4,3); imshow(blue_mask);     title('3. Blue Mask');
subplot(2,4,4); imshow(clean_img);     title('4. Edge Candidates');
subplot(2,4,5); imshow(img); hold on;
    rectangle('Position', best_bbox, 'EdgeColor', 'g', 'LineWidth', 3);
    title(sprintf('5. Detected (%.1f)', best_score));
subplot(2,4,6); imshow(bottom_plate);  title('6. Plate Bottom');
subplot(2,4,7); imshow(plate_bw);      title('7. Validated (N chars)');
subplot(2,4,8); text(..., recognized_string); title('8. Recognized');
saveas(fh, fullfile(results_dir, sprintf('result_%s.png', ...)));
close(fh);
```

**Purpose:** Generate and save a diagnostic 8-panel figure for each processed image.

`'Visible', 'off'` ensures figures are created in the background without interrupting the user's display — essential for batch processing. Figures are limited to the first 100 images (`if img_idx <= 100`) to prevent excessive disk usage.

#### Visualization Panel Guide

| Panel | Title | Content | Color Coding |
|---|---|---|---|
| 1 | Standardized | Resized RGB image | — |
| 2 | Filtered Gray | Median-filtered grayscale | — |
| 3 | Blue Mask | HSV blue segmentation | White = blue regions |
| 4 | Edge Candidates | Morphologically processed edges | White = candidate blobs |
| 5 | Detected (score) | Original image + plate bounding box | Green rectangle |
| 6 | Plate Bottom | Cropped plate region (grayscale) | — |
| 7 | Validated (N chars) | Binary plate + character boxes | Red = numbers, Blue = letters |
| 8 | Recognized | Arabic text string | Black on white |

---

### Stage 19 — Excel Export

```matlab
T = table(results_filenames, results_detected, results_strings,
          'VariableNames', {'Filename','Is_Detected','Recognized_Plate'});
writetable(T, fullfile(results_dir, 'alpr_results.xlsx'));
```

**Purpose:** Export all recognition results to a structured Excel spreadsheet for downstream analysis or evaluation.

| Column | Type | Description |
|---|---|---|
| `Filename` | String | Original image filename (e.g., `car001.jpg`) |
| `Is_Detected` | Boolean | `true` if a plate was detected, `false` otherwise |
| `Recognized_Plate` | String | Arabic characters (e.g., `و و و   ٦ ٦ ٦ ٦`) |

---

## 9. Function Reference

---

### `match_character()`

```matlab
function [best_match, best_score] = match_character(char_img, templates, is_number_mode)
```

**Purpose:** Match a single character image against the loaded template library using HOG feature similarity.

**Algorithm:**

1. **Mode filtering:** Based on `is_number_mode`, consider only digit templates or only letter templates
2. **Unique character enumeration:** For each unique character label in the template set:
   - Gather all template instances for that character
   - For each instance: extract HOG features with `CellSize = [8 8]`
   - Compute Euclidean distance: `dist = norm(features_char - feat_tpl)`
   - Convert to similarity: `cv = 1 / (1 + dist)`  (score approaches 1 for perfect match, 0 for distant)
   - Average all instance scores: `class_score = mean(char_corrs)`
3. **Best class selection:** Return the character label with the highest average score

**Similarity metric:** `1 / (1 + dist)` maps Euclidean distance to (0, 1]:
- `dist = 0` → score = `1.0` (identical features)
- `dist = 1` → score = `0.5`
- `dist → ∞` → score → `0.0`

**Why HOG?** Histogram of Oriented Gradients captures the **distribution of local edge orientations** within image cells. This makes the descriptor:
- **Scale-invariant** (because inputs are normalized to 42×24)
- **Illumination-robust** (HOG captures relative gradients, not absolute intensities)
- **Shape-discriminative** (different Arabic letters have distinct stroke patterns)

**HOG Parameters:**

| Parameter | Value | Effect |
|---|---|---|
| `CellSize` | `[8 8]` | Each cell covers 8×8 pixels; 42×24 image → ~15 cells |
| Bins per cell | 9 (default) | Captures 9 gradient orientation bins (0°–180°) |
| Block normalization | L2-Hys (default) | Normalizes contrast locally |

---

### `load_real_templates()`

```matlab
function templates = load_real_templates(tpl_dir)
```

**Purpose:** Scan the templates directory, load all character images, map filenames to Arabic Unicode characters, and standardize all images to 42×24 binary arrays.

**Filename parsing:**
```
fname = "m-1.png"
parts = split(fname, {'-', '.'})  →  {"m", "1", "png"}
prefix = lower(parts{1})          →  "m"
mapped_char = map("m")            →  "م"
```

**Key–Value Mapping Table (excerpt):**

| File Prefix | Arabic Output | Notes |
|---|---|---|
| `0`–`9` | `٠`–`٩` | Arabic-Indic digits |
| `a` | `ا` | Alef |
| `m` | `م` | Meem |
| `w` | `و` | Waw |
| `r` | `ر` | Ra |
| `sh` | `ش` | Shin (two-character key) |
| `kh` | `خ` | Kha (two-character key) |
| `th` | `ث` | Tha (two-character key) |

**Image processing pipeline for each template:**
1. `imread()` — load raw image
2. `rgb2gray()` if 3-channel — convert to grayscale
3. `imbinarize()` — convert to binary
4. `regionprops('BoundingBox')` — find tight crop around character
5. `imcrop(bw, st(1).BoundingBox)` — crop to character bounds
6. `imresize(bw, [42 24])` — resize to standard HOG canvas

**Why crop before resize?** Direct resize of a padded image would scale the empty whitespace along with the character, distorting HOG features. Tight-cropping first ensures only the character pixels occupy the HOG window.

**Multiple samples per character:** Multiple template files for the same character (e.g., `m-1.png`, `m-2.png`) produce multiple entries in the `templates` struct. In `match_character()`, the average HOG similarity across all samples is used, making the system more robust to intra-class variation.

---

### `robust_binarize()`

```matlab
function bw = robust_binarize(gray_plate)
```

**Purpose:** Convert a grayscale plate crop to binary with maximum reliability under varying illumination conditions.

**Algorithm — Dual-polarity voting:**

**Step 1 — Enhancement**
```matlab
enh = adapthisteq(gray_plate, 'ClipLimit', 0.02);
enh = imadjust(enh);
```
- `adapthisteq` (CLAHE): Contrast Limited Adaptive Histogram Equalization — enhances local contrast while suppressing noise amplification (ClipLimit = 0.02 is conservative)
- `imadjust`: Stretches pixel range to [0, 255] using percentile-based saturation

**Step 2 — Dark-foreground path (bwA)**
```matlab
bw1a = ~imbinarize(enh, graythresh(enh));        % Otsu inverted
bw2a = ~imbinarize(enh, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.45);
bw3a = enh < 120;                                 % Global hard threshold
bwA  = remove_large_blobs(bwareaopen(
           uint8(bw1a)+uint8(bw2a)+uint8(bw3a) >= 2, 12));
```

**Step 3 — Bright-foreground path (bwB)**
```matlab
bw1b = imbinarize(enh, graythresh(enh));
bw2b = imbinarize(enh, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.45);
bw3b = enh > 135;
bwB  = remove_large_blobs(bwareaopen(
           uint8(bw1b)+uint8(bw2b)+uint8(bw3b) >= 2, 12));
```

**Step 4 — Selection**
```matlab
if count_char_blobs(bwB, h) > count_char_blobs(bwA, h)
    bw = bwB;
else
    bw = bwA;
end
```

**Voting logic:** Each path requires 2-out-of-3 agreement (`>= 2`) between:
- Otsu global threshold
- Adaptive (local neighborhood) threshold
- Hard absolute threshold (120 or 135)

This majority vote is highly resilient: if one method is fooled by local illumination anomalies, the other two override it.

**Polarity selection:** `count_char_blobs()` counts how many valid character-shaped regions exist in each candidate binary image. The system selects whichever polarity produces more characters — a data-driven decision requiring no manual configuration.

| Threshold | Value | Meaning |
|---|---|---|
| Otsu auto | `graythresh(enh)` | Maximizes between-class variance |
| Adaptive sensitivity | 0.45 | 45% local neighborhood sensitivity |
| Dark hard threshold | 120 | Dark pixels (<120) treated as foreground |
| Bright hard threshold | 135 | Bright pixels (>135) treated as foreground |
| Min blob size | 12 pixels | Removes isolated specks |
| Large blob fraction | 40% of image | Removes overwhelming background blobs |

---

### `remove_large_blobs()`

```matlab
function bw = remove_large_blobs(bw)
    cc = bwconncomp(bw);
    for ci = 1:cc.NumObjects
        if length(cc.PixelIdxList{ci}) > 0.40 * numel(bw)
            bw(cc.PixelIdxList{ci}) = 0;
        end
    end
end
```

**Purpose:** Eliminate oversized connected components that represent background regions incorrectly classified as foreground.

**Threshold:** 40% of total image area. Any single blob covering more than 40% of the binary image is almost certainly the plate background (white region) or a large dark region misclassified as a character. These blobs are zeroed out.

**Why called after binarization?** Even with adaptive thresholding, large uniform regions can be incorrectly classified. This post-processing step ensures clean character isolation before counting and recognition.

---

### `count_char_blobs()`

```matlab
function n = count_char_blobs(bw, plate_h)
    st = regionprops(bw, 'BoundingBox');
    for j = 1:length(st)
        cb = st(j).BoundingBox;
        c_ar = cb(3)/cb(4);           % width/height aspect ratio
        chr  = cb(4)/plate_h;          % height relative to plate height
        if c_ar > 0.07 && c_ar < 2.5 && chr > 0.13 && chr < 0.95 && cb(3) > 3 && cb(4) > 4
            n = n + 1;
        end
    end
end
```

**Purpose:** Count the number of regions in a binary image that geometrically match expected character shapes. Used to compare the two binarization polarities.

**Validation criteria:**

| Criterion | Range | Reason |
|---|---|---|
| Aspect ratio | 0.07 – 2.5 | Characters are taller than wide, but not infinitely narrow |
| Height ratio | 0.13 – 0.95 | Character must be at least 13% and at most 95% of plate height |
| Minimum width | > 3 pixels | Reject single-pixel noise columns |
| Minimum height | > 4 pixels | Reject single-pixel noise rows |

**Role:** Acts as the **selection oracle** between bwA and bwB in `robust_binarize()`. The binarization that produces more geometrically valid character candidates is selected.

---

### `segment_characters()`

```matlab
function raw_chars = segment_characters(plate_bw)
```

**Purpose:** Extract character bounding boxes from the binary plate image, applying geometric filtering to reject noise, frame components, and dividers.

**Constants:**

| Variable | Value | Meaning |
|---|---|---|
| `MIN_WIDTH_RATIO` | 0.01 | Character must be ≥1% of plate width |
| `MAX_WIDTH_RATIO` | 0.30 | Character must be ≤30% of plate width |
| `MIN_HEIGHT_RATIO` | 0.10 | Character must be ≥10% of plate height |
| `MAX_HEIGHT_RATIO` | 0.85 | Character must be ≤85% of plate height |
| `ASPECT_RATIO_MIN` | 0.05 | Minimum width/height |
| `ASPECT_RATIO_MAX` | 3.00 | Maximum width/height (not too wide) |

**Special exclusion rules:**

1. **Top-region exclusion:**
   ```matlab
   if center_y < h_plate * 0.3, continue; end
   ```
   The top 30% of the plate image corresponds to the "مصر / EGYPT" header band. All regions in this zone are skipped — they are not license characters.

2. **Edge-column exclusion:**
   ```matlab
   if center_x < w_plate * 0.03 || center_x > w_plate * 0.97, continue; end
   ```
   Characters within 3% of the left or right image boundary are likely plate frame artifacts.

3. **Central divider exclusion:**
   ```matlab
   if c_w < 0.015 && center_x > w_plate*0.48 && center_x < w_plate*0.52, continue; end
   ```
   Very narrow blobs in the center 4% of plate width are likely the vertical divider line between the numbers and letters zones — explicitly removed.

---

## 10. OCR Methodology

### Overview

The system uses a **Histogram of Oriented Gradients (HOG) + Euclidean Nearest-Neighbor** approach for OCR. This is a classical computer vision technique that has been proven effective for character recognition tasks where a labeled template set is available.

### HOG Feature Extraction

HOG captures the **local gradient orientation distribution** within a fixed-size window:

```
Input: 42×24 binary character image
  ↓
Compute gradient magnitudes and orientations (Sobel-like)
  ↓
Divide into cells of 8×8 pixels → grid of ~5 rows × 3 cols = 15 cells
  ↓
For each cell: compute 9-bin orientation histogram (0°–180°)
  ↓
Group cells into overlapping blocks (2×2 cells)
  ↓
L2-Hys normalize each block
  ↓
Concatenate all block histograms → feature vector
```

**Why HOG for Arabic characters?**

- Arabic letters are **stroke-based**: their identity is encoded in the arrangement and orientation of pen strokes, which HOG captures directly
- HOG is **position-tolerant within cells**: small misalignments from imperfect segmentation are absorbed
- HOG is **illumination-robust**: it operates on gradient ratios, not absolute pixel values

### Template Matching Pipeline

```
For each candidate character crop:
    1. Resize to [42×24]
    2. Extract HOG feature vector F_char
    3. For each unique character class C:
        a. For each template T of class C:
            i.  Extract HOG feature vector F_tpl
            ii. dist = ||F_char - F_tpl||₂  (Euclidean distance)
            iii. score_i = 1 / (1 + dist)
        b. class_score(C) = mean(score_i for all templates of C)
    4. best_match = argmax(class_score)
    5. best_score = max(class_score)
    6. Accept if best_score > 0.1
```

### Confidence Threshold

A minimum score of **0.1** is required for a match to be accepted. With the `1/(1+dist)` mapping:
- Score = 0.1 corresponds to `dist = 9.0` (very dissimilar)
- Score = 0.5 corresponds to `dist = 1.0` (moderately similar)
- Score = 0.9 corresponds to `dist ≈ 0.11` (very similar)

The threshold of 0.1 is deliberately loose, accepting even weak matches over rejecting potentially valid characters. This is appropriate because the system later caps the total character count (4 numbers, 3 letters) using score-ranking anyway.

### Number vs. Letter Mode

```matlab
arabic_digits = {'٠','١','٢','٣','٤','٥','٦','٧','٨','٩','0','1',...,'9'};
is_digit_template = ismember(current_char, arabic_digits);

if (is_number_mode && ~is_digit_template), continue; end
if (~is_number_mode && is_digit_template), continue; end
```

This mode switching is critical. Without it, the system might match Arabic digit "٦" (which looks like a reversed hook) against the letter "و" or vice versa. By restricting the comparison set, confusion between visually similar cross-category characters is eliminated.

---

## 11. Detection Scoring Logic

The scoring function is a **weighted multi-evidence linear combiner**. Each evidence type contributes to a cumulative score:

```
score = PRIMARY + RATIO_BONUS + BLUE_PROX_BONUS + BLUE_OVERLAP + WHITE_OVERLAP
      + POSITION_BONUS + ASPECT_BONUS + TEXTURE_BONUS
```

### Evidence Weights Rationale

| Evidence | Max Contribution | Rationale |
|---|---|---|
| Vertical edge density | ~15–30 (vd × 100) | Primary structural cue; ~15–30% of plate pixels are vertical edges |
| V/H edge ratio | up to 10 | Validates plate-like character structure |
| Blue proximity | +15 | Strong prior: blue strip is definitive Egyptian plate marker |
| Blue pixel ratio | up to 15 (0.5 × 30) | Blue content inside bbox confirms plate identity |
| White pixel ratio | up to 8.5 (0.85 × 10) | White background is characteristic |
| Vertical position | +5 | Plates are typically in middle-lower vehicle area |
| Aspect ratio | +3 | Secondary geometric validation |
| Column variance | up to 5 | Texture gradient confirms character pattern |

**Score interpretation:**
- Score > 40: High-confidence detection (typical range for clear plates)
- Score 20–40: Medium confidence
- Score 3–20: Low confidence, accepted but may have errors
- Score < 3: Rejected (treated as no detection)

---

## 12. Visualization Outputs

### Sample Output (result_1.png)

The 8-panel figure shows the full processing chain for a single image:

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  1. Standardized│ 2. Filtered Gray│   3. Blue Mask  │4. Edge Candidates│
│  [RGB image]    │ [Grayscale]     │ [Binary blue]   │ [Morpho blobs]  │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│  5. Detected    │ 6. Plate Bottom │ 7. Validated    │  8. Recognized  │
│  [+green bbox]  │ [Plate crop]    │ [Binary+boxes]  │ [Arabic text]   │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

Each panel directly corresponds to a processing stage, enabling visual debugging of every step.

### Character Box Colors (Panel 7)

- **Red rectangles:** Digit character bounding boxes (numbers zone)
- **Blue rectangles:** Letter character bounding boxes (letters zone)

This color coding visually confirms correct zone separation.

---

## 13. Result Export

### alpr_results.xlsx

```
| Filename     | Is_Detected | Recognized_Plate        |
|--------------|-------------|-------------------------|
| car001.jpg   | true        | و و و   ٦ ٦ ٦ ٦        |
| car002.jpg   | true        | ر ن أ   ٨ ٤ ٦ ٨        |
| car003.jpg   | false       |                         |
```

### extracted_chars/ Directory

Individual character images are saved for inspection and potential use as new templates:

```
img_01_num_1.png   ← 1st digit from image 1
img_01_num_2.png   ← 2nd digit from image 1
img_01_let_1.png   ← 1st letter from image 1
```

All saved at 42×24 pixels in binary format, ready for use as additional training templates.

### Console Output Format

```
[01/10] Processing File : car001.jpg
Detection result        : detected
License plate characters: و و و   ٦ ٦ ٦ ٦
--------------------------------------------------
[02/10] Processing File : car002.jpg
Detection result        : detected
License plate characters: ر ن أ   ٨ ٤ ٦ ٨
--------------------------------------------------
```

---

## 14. Variable & Threshold Reference

| Variable | Value | Stage | Description |
|---|---|---|---|
| `800/orig_w` | dynamic | Resize | Target width scale factor |
| `hue > 0.52 & < 0.75` | — | Blue mask | HSV hue range for Egyptian plate blue |
| `sat > 0.25` | — | Blue mask | Minimum saturation to exclude grays |
| `val > 0.12` | — | Blue mask | Minimum brightness to exclude black |
| `sat < 0.20` | — | White mask | Maximum saturation for white/gray |
| `val > 0.60` | — | White mask | Minimum brightness for white |
| `tv * 0.7` | — | Sobel | 30% threshold relaxation for weaker edges |
| `strel('rectangle',[3 15])` | — | Dilation | Horizontal gap bridging kernel |
| `max_area = h*w*0.05` | 5% of image | Area filter | Maximum blob area allowed |
| `bwareaopen(300)` | — | Noise removal | Min blob area for candidates |
| `AR: 1.5–6.0` | — | Scoring gate | Plate aspect ratio range |
| `vd >= 0.03` | — | Scoring gate | Minimum vertical edge density |
| `score > 3` | — | Final filter | Minimum score for valid detection |
| `separator_x = w * 0.48` | — | OCR | Number/letter spatial split |
| `[42 24]` | — | OCR | Standard HOG character canvas |
| `CellSize [8 8]` | — | HOG | HOG cell size in pixels |
| `score > 0.1` | — | OCR | Minimum match confidence |
| `max 4 numbers` | — | Format | Egyptian plate digit cap |
| `max 3 letters` | — | Format | Egyptian plate letter cap |
| `0.40 * numel(bw)` | 40% | Blob removal | Max blob fraction in binarized plate |
| `ClipLimit 0.02` | — | CLAHE | Conservative enhancement clipping |
| `Sensitivity 0.45` | — | Adaptive bin. | Local thresholding sensitivity |

---

## 15. Sample Console Output

```
Templates loaded: 148
Found 10 images in: /Users/username/project/dataset

[01/10] Processing File : car001.jpg
Detection result        : detected
License plate characters: و و و   ٦ ٦ ٦ ٦
--------------------------------------------------

[02/10] Processing File : car002.jpg
Detection result        : detected
License plate characters: ر ن أ   ٨ ٤ ٦ ٨
--------------------------------------------------

[03/10] Processing File : car003.jpg
Detection result        : detected
License plate characters: م ط ر   ٥ ٥ ٥
--------------------------------------------------

[04/10] Processing File : car004.jpg
Detection result        : not detected
--------------------------------------------------

=== Done === Detected: 9 / 10
Saved: alpr_results.xlsx
```

---

## 16. Algorithm Strengths

- **No training required** — operates entirely from handcrafted rules and HOG templates; no neural network, no GPU, no dataset annotations needed
- **Arabic-native design** — explicitly handles RTL reading order, Arabic-Indic numerals, and Arabic letter inventory
- **Dual-polarity robustness** — handles both normal-illumination and night/reflection plate appearances without manual configuration
- **Multi-evidence scoring** — no single cue can cause a false positive; the scoring system requires consistent evidence from multiple independent sources
- **Fallback mechanism** — blue-strip expansion ensures a second detection opportunity when the primary scoring fails
- **Interpretable pipeline** — every stage is visible, debuggable, and adjustable; no black-box components
- **Lightweight** — runs on any MATLAB installation with Image Processing and Computer Vision toolboxes; no specialized hardware
- **Batch processing** — processes an entire folder automatically with a single execution
- **Rich diagnostic output** — 8-panel figures and extracted character images enable rapid manual review

---

## 17. Limitations

- **Template dependency** — OCR quality is directly proportional to the quality and diversity of the template library; a sparse template set degrades accuracy significantly
- **Fixed plate format** — designed exclusively for the current Egyptian civilian plate format (blue strip, white background, Arabic characters + Arabic-Indic numbers); other Egyptian plate types (police, military, vintage) are not supported
- **Severe angle sensitivity** — perspective distortion beyond ~15° off-axis is not corrected; no perspective homography is applied
- **Night/low-light conditions** — while dual-polarity binarization helps, severely underexposed images degrade edge detection quality
- **Occluded plates** — partial occlusion (mud, dirt, objects) can prevent complete character segmentation
- **High-resolution input penalty** — images are scaled to 800px width, which may lose fine character detail from very high-resolution cameras
- **Connected character segments** — Arabic characters sometimes connect (e.g., ور as one blob), causing under-segmentation; no character splitting logic is implemented
- **No perspective correction** — plates photographed at an angle are not rectified before binarization
- **Static separator threshold** — the 48% plate width split assumes standard layout; unusual plates with wider number zones may misclassify some characters

---

## 18. Future Improvements

### Short-term Enhancements

- **Perspective correction:** Add projective transform (homography estimation from plate corners) to de-skew angled plates before binarization
- **Connected component splitting:** Implement vertical projection histogram analysis to split merged Arabic character blobs
- **Template augmentation:** Add synthetic template variations (rotation, blur, scale) to improve HOG matching robustness
- **Multi-scale detection:** Run the scoring pipeline at multiple image scales to handle plates at different distances

### Medium-term Enhancements

- **SVM classifier:** Replace Euclidean HOG nearest-neighbor with a trained SVM per character class for better discrimination
- **Plate type classification:** Add a plate-type detection stage to support multiple Egyptian plate formats (private, commercial, motorcycle, taxi)
- **LSTM sequence model:** Model the plate string as a sequence with character-to-character transition probabilities (language model), reducing impossible combinations
- **Plate boundary regression:** Train a lightweight regressor to refine the bounding box from the scoring stage

### Long-term / Research Directions

- **End-to-end CNN/Transformer:** Replace the entire pipeline with a YOLO-based detector + CRNN for OCR, trained on a large Egyptian plate dataset
- **Nighttime specialization:** Train a separate low-light preprocessing module using paired day/night images
- **Multi-line plate support:** Handle plates with two character rows (some truck and bus plates)
- **Real-time video processing:** Optimize for frame-by-frame processing at 15+ fps using MATLAB's parallel processing tools or MEX optimization

---

## 19. Academic Context

This project was developed as part of **ECE 228** (as indicated by the source file header `%% ECE 228: ALPR for Egyptian License Plates`), a course in image processing and computer vision engineering.

The implementation demonstrates applied mastery of:

- **Image preprocessing theory** (color space transforms, noise filtering)
- **Feature engineering** (HSV color segmentation, HOG descriptors)
- **Classical computer vision** (Sobel edge detection, morphological operations, connected component analysis)
- **Pattern recognition** (nearest-neighbor classification, confidence thresholding)
- **Systems engineering** (modular function design, error handling, batch processing, result export)
- **Domain knowledge integration** (Egyptian plate structure, Arabic character sets, RTL reading order)

### Methodology Classification

| Component | Category | Technique |
|---|---|---|
| Preprocessing | Signal processing | Median filter, CLAHE, imadjust |
| Color detection | Computer vision | HSV segmentation |
| Plate detection | Feature extraction + scoring | Sobel edges + multi-criteria scoring |
| Binarization | Image processing | Adaptive threshold (local + global) |
| Segmentation | Connected component analysis | regionprops |
| OCR | Feature-based ML | HOG + Euclidean nearest-neighbor |

---

## 20. References

1. **Dalal, N., & Triggs, B.** (2005). *Histograms of Oriented Gradients for Human Detection.* IEEE CVPR 2005. — Original HOG paper; the feature extraction used in this system.

2. **Gonzalez, R. C., & Woods, R. E.** (2018). *Digital Image Processing* (4th ed.). Pearson. — Core reference for Sobel edge detection, morphological operations, and thresholding methods.

3. **Jain, A. K.** (1989). *Fundamentals of Digital Image Processing.* Prentice Hall. — Foundation for connected component analysis and binarization theory.

4. **Otsu, N.** (1979). *A Threshold Selection Method from Gray-Level Histograms.* IEEE Transactions on Systems, Man, and Cybernetics. — Theoretical basis for `graythresh()` (Otsu's method).

5. **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV. — Basis for `adapthisteq()` (CLAHE) used in plate enhancement.

6. **MathWorks MATLAB Documentation.** *Image Processing Toolbox Reference.* https://www.mathworks.com/help/images/ — Reference for all MATLAB built-in functions used.

7. **MathWorks MATLAB Documentation.** *Computer Vision Toolbox — extractHOGFeatures.* https://www.mathworks.com/help/vision/ref/extracthogfeatures.html

8. **Egyptian Traffic Law (Law No. 66 of 1973 and amendments).** — Defines the official Egyptian vehicle license plate format and character set.

---
## 21. team
    # ECE 228: Image Processing - Team Information

# Team Members
1. Ahmed Mohamed Attia Mohamed
2. Ali El-Shawadfy Abdallah El-Sayed
3. Omar Hosny Mohamed Ahmed Abouzeid
4. Anas Ali Hammad El-Sayed
5. Abdelhay Lotfy El-Sayed El-Gawahry

# Affiliation
* **University:** Zagazig University
* **Faculty:** Faculty of Engineering
* **Department:** Electronics and Communications Engineering Department
* **Course:** ECE 228: Image Processing


## 22. Conclusion

This ALPR system demonstrates how a thoughtfully designed **classical image processing pipeline** can solve a real-world computer vision problem — recognizing Arabic license plates — without any deep learning infrastructure. Each stage of the pipeline addresses a specific challenge:

- **HSV color segmentation** exploits the Egyptian plate's distinctive blue strip to guide plate localization
- **Dual Sobel edge detection** with relaxed thresholds captures character strokes even in challenging illumination
- **Multi-criteria scoring** prevents false positives by requiring convergent evidence from multiple independent signals
- **Dual-polarity binarization** with majority voting adapts to both normal and inverted plate appearances
- **HOG template matching** provides a principled, scale-invariant approach to Arabic character recognition
- **Confidence thresholding and format-based caps** enforce the structural constraints of Egyptian plates as a final sanity check

The result is a system that is **transparent, maintainable, and extensible** — qualities that matter greatly in engineering education and real-world deployment where interpretability and debuggability are as important as raw accuracy.

---

*README generated for Zagazig ECE 228 — Egyptian License Plate ALPR Project*
*MATLAB Image Processing & Computer Vision Pipeline*
