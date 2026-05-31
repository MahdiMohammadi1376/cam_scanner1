import argparse
from cam_scanner.scanner import CamScanner
from cam_scanner.utils import save_result, draw_corners
import cv2
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Cam Scanner - document digitiser")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--output", default="results", help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--height", type=int, default=500,
        help="Processing height in pixels (default: 500)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    scanner = CamScanner(target_height=args.height)
    try:
        result = scanner.scan(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit()
    save_result(args.output,result["org_image"],result["final_image"],result["enhanced_image"])
    preview = draw_corners(result["final_image"], result["corners_point"])

    plt.subplot(131);plt.imshow(result["org_image"][...,::-1]);plt.title('original image')
    plt.subplot(132);plt.imshow(preview[...,::-1]);plt.title('correct perspective image')
    plt.subplot(133);plt.imshow(result["enhanced_image"],cmap='gray');plt.title('scan image')
    plt.show()

if __name__ == "__main__":
    main()




