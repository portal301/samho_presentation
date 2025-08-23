import cv2
import numpy as np
import argparse

class ImageCarrier:
    def __init__(self, img_src:np.ndarray):
        self.img_src = img_src

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

        
# === Example Usage ===
# python seam_practice1/1_carrier.py -p data/light/img_ref/img_1_0000.jpg
if __name__ == '__main__':
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Laser Line Detector")
    parser.add_argument("--image_path","-p", type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()
    image_filepath = args.image_path

    # image_raw = cv2.imread(os.path.join(base_image_path, image_filepath))
    image_raw = cv2.imread(image_filepath)
    image_carrier=ImageCarrier(image_raw)
    image_carrier.rotate_image(rotate=270, y_invert=False)

    cv2.imshow("Image Carrier", image_carrier.img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
