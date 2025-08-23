import cv2
import numpy as np
import json
import os
import argparse


class ImageCarrier:
    def __init__(self, img_src:np.ndarray):
        """
        Initializes the image carrier with an image path and rotation.
        :param image_path: Path to the image file.
        :param rotate: Rotation angle (0, 90, 180, or 270 degrees).
        """
        self.img_src = img_src
        self.offset = None

    def rotate_image(self, rotate=0, y_invert=False):
        if rotate not in [0, 90, 180, 270]:
            raise ValueError("Rotate must be one of 0, 90, 180, or 270 degrees.")

        """Loads the image from file."""
        if rotate == 90:
            self.img_src = cv2.rotate(self.img_src, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            self.img_src = cv2.rotate(self.img_src, cv2.ROTATE_180)
        elif rotate == 270:
            self.img_src = cv2.rotate(self.img_src, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate != 0:
            raise ValueError("Rotate must be one of 0, 90, 180, or 270 degrees.")
        if y_invert:
            self.img_src = cv2.flip(self.img_src, 0)
        self.h, self.w, _ = self.img_src.shape
        return self.img_src


class LaserLineDetector:
    def __init__(self, config:dict):
        self.config = config

    def img_sharpening(self, img:np.ndarray):
        """
        Applies sharpening and Gaussian blur to the image.
        :param img: Input image.
        :return: Sharpened and blurred image.
        """
        # Sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 8, -1],
                           [0, -1, 0]])
        # Apply sharpening
        img_sharpened = cv2.filter2D(img, -1, kernel)
        # Slight Gaussian blur to smooth out harsh edges
        img_sharpened = cv2.GaussianBlur(img_sharpened, (5, 5), sigmaX=1)
        return img_sharpened
    

    def analyze(self, image_carrier:ImageCarrier=None):
        # sharpening the image src
        sharpened_image = self.img_sharpening(image_carrier.img_src)        
        # hstack image
        hstack_img = np.hstack((image_carrier.img_src, sharpened_image))

        cv2.imshow("Sharpened Image", hstack_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # hls filter for green
        # img_src_hls = cv2.cvtColor(image_carrier.img_src, cv2.COLOR_BGR2HLS)
        # mask_hls = cv2.inRange(img_src_hls, (0, 30, 0), (255, 255, 255))
        img_src_hls = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2HLS)
        lower = np.array([0, 50, 0])
        upper = np.array([255, 255, 255])
        mask_hls = cv2.inRange(img_src_hls, lower, upper)
        mask_hls_color = cv2.cvtColor(mask_hls, cv2.COLOR_GRAY2BGR)
        hstack_img = np.hstack((hstack_img, mask_hls_color))

        cv2.imshow("HLS Image", hstack_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image_carrier.offset = 0
        return


    def process_image_debug(self, image_carrier:ImageCarrier):
        """Processes the image to extract the laser line."""
        self.img_src = image_carrier.rotate_image(rotate=270, y_invert=False)

        try:
            self.analyze(image_carrier=image_carrier)
            offset = image_carrier.offset
        except Exception as e:
            print(f"Error in analyze: {e}")
            return None
        return offset
        
# === Example Usage ===
# python seam_practice1/2_analyzer_hls.py -p data/light
if __name__ == '__main__':
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Laser Line Detector")
    parser.add_argument("--image_path","-p", type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()
    image_filepath = args.image_path

    detector = LaserLineDetector(config={})

    image_raw = cv2.imread(image_filepath)
    image_carrier=ImageCarrier(image_raw)
    offset_m = detector.process_image_debug(image_carrier=image_carrier)

