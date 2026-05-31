 
from pickletools import dis

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class CamScanner:
    def __init__(self,target_height:int = 500):
        self.target_height = target_height
        self.ratio: float = 1.0

    def scan(self, image_path:str):
        img,resized_image = self.resize_image(image_path)
        #binary_image = self.adaptive_threshold(resize_image)
        edges = self.edge_detection(resized_image)
        corners = self.contour_finding(edges)
        if corners is None:
            raise ValueError("try a photo with higher contrast")
        final_img,order_corner = self.perspective_transformation(img,corners)
        enhanced_img = self.enhanced_image(final_img)
        return {
            'org_image':img,
            'final_image':final_img,
            'enhanced_image':enhanced_img,
            'corners_point':order_corner,
        }


    def resize_image(self,image_path:str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"image not found at {image_path}")
        self.ratio = image.shape[0]/self.target_height
        new_width = int(image.shape[1]/self.ratio)
        resized_image = cv2.resize(image,(new_width,self.target_height))
        return image,resized_image
    
    def edge_detection(self,resized_image):
        kernel = np.ones((5,5),np.uint8)
        morph_image = cv2.morphologyEx(resized_image,cv2.MORPH_CLOSE,kernel,iterations=3)
        morph_image_gray = cv2.cvtColor(morph_image,cv2.COLOR_BGR2GRAY)
        clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        morph_image_gray = clahe1.apply(morph_image_gray)
        morph_image_blurred = cv2.GaussianBlur(morph_image_gray,(5,5),0)
        edge_image = cv2.Canny(morph_image_blurred,50,120)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        edge_image = cv2.dilate(edge_image,kernel2,iterations=1)
        edge_image = cv2.morphologyEx(edge_image,cv2.MORPH_CLOSE,kernel2,iterations=3)
        return edge_image

    # def adaptive_threshold(self, resized_image):
    #     gray_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    #     blurred_image = cv2.GaussianBlur(gray_image,(5,5),0)
    #     binary_image = cv2.adaptiveThreshold(blurred_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2)
    #     return binary_image

    def contour_finding(self,processed_image):
        global display_image
        contours,_ = cv2.findContours(processed_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key=cv2.contourArea,reverse=True)
        display_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            if cv2.contourArea(contour)>processed_image.shape[0]*processed_image.shape[1]*0.2:
                hull = cv2.convexHull(contour)
                hull_perimeter = cv2.arcLength(hull,True)
                cv2.drawContours(display_image,[hull],-1,(250,0,250),2)
                for epsilon in [0.01,0.02,0.03,0.04,0.06,0.08]:
                    approx = cv2.approxPolyDP(hull,epsilon*hull_perimeter,True)
                    if len(approx)==4:
                        cv2.drawContours(display_image,[approx],-1,(0,250,0),3)
                        return approx.reshape(4,2).astype("float32")
        return None

    def perspective_transformation(self,original_image,corners):
        if corners is None:
            return None
        order_corners = self.order_points(corners)*self.ratio
        (tl,tr,br,bl) = order_corners
        widthA = np.sqrt(((tl[0]-tr[0])**2)+((tl[1]-tr[1])**2))
        widthB = np.sqrt(((bl[0]-br[0])**2)+((bl[1]-br[1])**2))
        max_width = max(int(widthA),int(widthB))
        heightA = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
        heightB = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
        max_height = max(int(heightA),int(heightB))
        destination_corners = np.array([[0,0],
                                        [max_width,0],
                                        [max_width,max_height],
                                        [0,max_height]],dtype='float32')
        destination_corners2 = np.array([[20,20],
                                        [max_width-20,20],
                                        [max_width-20,max_height-20],
                                        [20,max_height-20]],dtype='float32')
        
        M = cv2.getPerspectiveTransform(order_corners,destination_corners)
        final_image = cv2.warpPerspective(original_image,M,(max_width,max_height))
        return final_image,destination_corners2
    
    def enhanced_image(self,final_image):
        if final_image is None:
            return None
        gray1 = cv2.cvtColor(final_image,cv2.COLOR_BGR2GRAY)
        #NlMeans_denoising = cv2.fastNlMeansDenoising(gray1, h=10,templateWindowSize=7,searchWindowSize=21)
        blurred_bilateral = cv2.bilateralFilter(gray1,1,75,75)
        enhanced = cv2.adaptiveThreshold(blurred_bilateral,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        enhanced = cv2.morphologyEx(enhanced,cv2.MORPH_OPEN,kernel,iterations=1)
        return enhanced

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4,2),dtype='float32')
        s = pts.sum(axis=1)
        rect[0]=pts[np.argmin(s)]
        rect[2]=pts[np.argmax(s)]
        diff = np.diff(pts,axis=1)
        rect[1]=pts[np.argmin(diff)]
        rect[3]=pts[np.argmax(diff)]
        return rect

