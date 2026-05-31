import os
import cv2
import numpy as np


def save_result(output_dir:str,org_image,final_image,enhanced_image):
    os.makedirs(output_dir,exist_ok=True)
    #cv2.imwrite(os.path.join(output_dir,"original_img2.jpg"),org_image)
    cv2.imwrite(os.path.join(output_dir,"img8.jpg"),final_image)
    cv2.imwrite(os.path.join(output_dir,"scan8.jpg"),enhanced_image)
    print(f"Result saved to: {output_dir}")

def draw_corners(image,corners):
    preview = image.copy()
    for corner in corners:
        x,y = corner
        cv2.circle(preview,(int(x),int(y)),20,(250,0,0),-1)
    return preview


