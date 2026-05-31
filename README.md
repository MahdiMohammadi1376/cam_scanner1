# Cam Scanner

A document scanner built entirely with classical computer vision — no deep learning required.  
Detects a document in a photo, corrects the perspective, and produces a clean binary scan.

## Demo

![demo](/demo.git)

## How it works

The pipeline has two separate jobs that must never be mixed:

**1. Boundary detection** — finding where the document is
- Morphological close to bridge gaps in the document border
- CLAHE to normalize uneven lighting
- Canny edge detection on the equalized image
- Convex hull + iterative "approxPolyDP" to find the largest quadrilateral

**2. Content enhancement** — making the scan readable
- Perspective warp to rectify the document to a top-down view
- Bilateral filter — removes noise while preserving sharp text edges
- Adaptive threshold — produces a clean black-and-white output
- Morphological opening — removes any remaining isolated noise pixels


## Project structure

```
cam-scanner/
├── cam_scanner/
│   ├── __init__.py
│   ├── scanner.py       # all CV logic
│   └── utils.py         # save and draw helpers
├── images/              # sample input images
├── results/             # output (created at runtime, not tracked by git)
├── main.py              # entry point
├── requirements.txt
└── README.md
```

## Installation

bash
git clone https://github.com/MahdiMohammadi1376/cam_scanner1.git
cd cam-scanner1
pip install -r requirements.txt


## Usage

bash
python main.py images/sample.jpg --output results/


## Results

The scanner outputs two files:
- `img.jpg` — perspective-corrected colour image
- `scan.jpg` — clean black-and-white scan


## Tech stack

Python · OpenCV · NumPy

