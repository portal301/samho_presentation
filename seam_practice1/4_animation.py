import cv2
import numpy as np
from scipy.signal import savgol_filter
import json
import os
import sys
import argparse
from dev_assistant_utils import print2, Color


class ImageCarrier:
    def __init__(self, img_src:np.ndarray, laser_id:str):
        """
        Initializes the image carrier with an image path and rotation.
        :param image_path: Path to the image file.
        :param rotate: Rotation angle (0, 90, 180, or 270 degrees).
        """
        self.laser_id = laser_id
        self.img_src = img_src

        self.img_roi = None
        self.img_roi_baseline = None
        self.color_mask = None
        self.roi_window = None
        self.target = None
        self.offset = None
        self.baseline_smooth = None
        self.baseline_curvature = None
        self.roi_bbox_frame = None

        self.img_roi_analyzed = None
        self.img_result = None


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
        """
        Initializes the LaserLineDetector.
        :param image_path: Path to the image.
        :param color: Color to detect ("red" or "green").
        """
        self.config = config

        self.hls_red_h_range_1 = self.config["color_hls_filter"]["red"]["hls_h1_range"]
        self.hls_red_h_range_2 = self.config["color_hls_filter"]["red"]["hls_h2_range"]
        self.hls_red_l_range = self.config["color_hls_filter"]["red"]["hls_l_range"]
        self.hls_red_s_range = self.config["color_hls_filter"]["red"]["hls_s_range"]
        self.hls_green_h_range = self.config["color_hls_filter"]["green"]["hls_h_range"]
        self.hls_green_l_range = self.config["color_hls_filter"]["green"]["hls_l_range"]
        self.hls_green_s_range = self.config["color_hls_filter"]["green"]["hls_s_range"]
        self.hls_blue_h_range = self.config["color_hls_filter"]["blue"]["hls_h_range"]
        self.hls_blue_l_range = self.config["color_hls_filter"]["blue"]["hls_l_range"]
        self.hls_blue_s_range = self.config["color_hls_filter"]["blue"]["hls_s_range"]

        # image filter parameters
        self.baseline_curvature_threshold = 0.14
        self.hls_filter_lightness_threshold = 70


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
    
    def remove_small_islands_from_mask(self, mask:np.ndarray, min_area=20, morphology_filter=True):
        """
        Removes small islands in the mask based on area.
        :param mask: Input binary mask.
        :param min_area: Minimum area of the contour to keep.
        :return: Processed mask with small islands removed.
        """
        # find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filter contours based on area
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                # fill the contour with black color
                cv2.drawContours(mask, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
        mask=mask.astype(np.uint8)

        # optional: erase small noise in the mask
        if morphology_filter is True:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
        return mask

    def get_color_hls_mask(self, image_color:np.ndarray=None, color:str=None):
        """Extracts a binary mask based on the chosen color in HLS color space."""
        if image_color is None:
            raise ValueError("Image color must be provided.")
        if color is None:
            raise ValueError("Color must be specified (red, green, or blue).")

        hls = cv2.cvtColor(image_color, cv2.COLOR_BGR2HLS)
        if color == "green":
            lower = np.array([self.hls_green_h_range[0], self.hls_green_l_range[0], self.hls_green_s_range[0]])
            upper = np.array([self.hls_green_h_range[1], self.hls_green_l_range[1], self.hls_green_s_range[1]])
            mask = cv2.inRange(hls, lower, upper)
        elif color == "red":
            # Using narrowed red range; you can adjust these values as needed.
            lower_red1 = np.array([self.hls_red_h_range_1[0], self.hls_red_l_range[0], self.hls_red_s_range[0]])
            upper_red1 = np.array([self.hls_red_h_range_1[1], self.hls_red_l_range[1], self.hls_red_s_range[1]])
            lower_red2 = np.array([self.hls_red_h_range_2[0], self.hls_red_l_range[0], self.hls_red_s_range[0]])
            upper_red2 = np.array([self.hls_red_h_range_2[1], self.hls_red_l_range[1], self.hls_red_s_range[1]])
            mask1 = cv2.inRange(hls, lower_red1, upper_red1)
            mask2 = cv2.inRange(hls, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == "blue":
            lower = np.array([self.hls_blue_h_range[0], self.hls_blue_l_range[0], self.hls_blue_s_range[0]])
            upper = np.array([self.hls_blue_h_range[1], self.hls_blue_l_range[1], self.hls_blue_s_range[1]])
            mask = cv2.inRange(hls, lower, upper)
        else:
            raise ValueError("Color must be among red, green, and blue. Given: {}".format(color))
        
        return mask


    def analyze(self, image_carrier:ImageCarrier=None):
        color = self.config[image_carrier.laser_id]["color"]
        print2(f"Color: {color}", Color.YELLOW)

        # sharpening the image src
        image_carrier.img_src = self.img_sharpening(image_carrier.img_src)
        image_carrier.color_mask = self.get_color_hls_mask(image_color=image_carrier.img_src, color=color)

        img_src_hls = cv2.cvtColor(image_carrier.img_src, cv2.COLOR_BGR2HLS)
        mask_hls =  cv2.inRange(img_src_hls, (0, 20, 0), (255, 255, 255))
        mask_hls = cv2.bitwise_and(mask_hls, image_carrier.color_mask)
        mask_hls = self.remove_small_islands_from_mask(mask=mask_hls, min_area=20, morphology_filter=True)
        mask_hls2 = cv2.inRange(img_src_hls, (0, self.hls_filter_lightness_threshold, 0), (255, 255, 255))
        # use color mask to filter the hls mask
        mask_hls2 = cv2.bitwise_and(mask_hls2, image_carrier.color_mask)

        max_x = np.max(np.where(mask_hls2 > 0)[1])
        indices2 = np.where(mask_hls2[:, max_x] > 0)[0]
        if indices2.size:
            y_val = int(np.mean(indices2))
        else:
            y_val = None
        endpoint = (max_x, y_val)

        image_carrier.roi_bbox_frame = [self.config[image_carrier.laser_id]["roi_x_range"], self.config[image_carrier.laser_id]["roi_y_range"]]

        print2(f"endpoint: {endpoint}, roi_bbox_frame: {image_carrier.roi_bbox_frame}", Color.YELLOW)

        if self.config[image_carrier.laser_id].get("fix_roi", False) is True:
            roi_x = [self.config[image_carrier.laser_id]["roi_center"][0]+image_carrier.roi_bbox_frame[0][0], self.config[image_carrier.laser_id]["roi_center"][0]+image_carrier.roi_bbox_frame[0][1]]
            roi_y = [self.config[image_carrier.laser_id]["roi_center"][1]+image_carrier.roi_bbox_frame[1][0], self.config[image_carrier.laser_id]["roi_center"][1]+image_carrier.roi_bbox_frame[1][1]]
        else:
            roi_x = [int(endpoint[0]+image_carrier.roi_bbox_frame[0][0]), int(endpoint[0]+image_carrier.roi_bbox_frame[0][1])]
            roi_y = [int(endpoint[1]+image_carrier.roi_bbox_frame[1][0]), int(endpoint[1]+image_carrier.roi_bbox_frame[1][1])]

        image_carrier.roi_window = [roi_x, roi_y]
        # get the image region of interest (ROI)
        image_carrier.img_roi = image_carrier.img_src[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        mask_roi = image_carrier.color_mask[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        mask_roi_hls2 = mask_hls2[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        # apply morphology to the mask_roi_hls_hsv
        kernel = np.ones((3, 3), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
        mask_roi = self.remove_small_islands_from_mask(mask=mask_roi, min_area=20, morphology_filter=True)
        mask_roi_hls = mask_hls[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        mask_roi_hls2 = cv2.morphologyEx(mask_roi_hls2, cv2.MORPH_CLOSE, kernel)
        mask_roi_hls2 = cv2.morphologyEx(mask_roi_hls2, cv2.MORPH_OPEN, kernel)
        indices = np.where(mask_roi_hls != 0)
        indices2 = np.where(mask_roi_hls2 != 0)
        
        mask_roi_overlay = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        mask_roi_overlay[indices[0], indices[1]] = [0, 150, 255]  # yello color for overlay
        mask_roi_overlay[indices2[0], indices2[1]] = [0, 0, 255]  # red color for overlay

        # if debug==True: # debug mode
            # hstack_img = np.hstack((image_carrier.img_roi, mask_roi_overlay, img_roi_sharpened, cv2.cvtColor(mask_roi2, cv2.COLOR_GRAY2BGR)))
        hstack_img = np.hstack((image_carrier.img_roi, mask_roi_overlay))
        hstack_img = cv2.resize(hstack_img, (0, 0), fx=4, fy=4)
        cv2.imshow("ROI and Mask", hstack_img)
        cv2.waitKey(0)

        image_carrier.offset = 0
        return


    def process_image_debug(self, image_carrier:ImageCarrier):
        """Processes the image to extract the laser line."""
        rotate = self.config[image_carrier.laser_id]["rotate"]
        y_invert = self.config[image_carrier.laser_id]["invert_y_axis"]
        self.img_src = image_carrier.rotate_image(rotate=rotate, y_invert=y_invert)

        try:
            self.analyze(image_carrier=image_carrier)
            offset = image_carrier.offset

        except Exception as e:
            print(f"Error in analyze: {e}")
            image_carrier.target = None
            image_carrier.offset = None
            image_carrier.img_roi_analyzed = None
            image_carrier.img_result = None
            return None, None
        
        return image_carrier.offset

# === Example Usage ===
# python seam_practice1/4_animation.py -p data/light
if __name__ == '__main__':

    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Laser Line Detector")
    parser.add_argument("--dir_path","-p", type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()

    with open(os.path.join("data", "hw_calibration", "seamtracker_config.json"), "r") as f:
        seamtracker_config = json.load(f)
    print2(f"seamtracker_config: {seamtracker_config}", Color.YELLOW)
    detector = LaserLineDetector(config=seamtracker_config)

    relative_image_path = args.dir_path
    base_image_path = os.path.join(relative_image_path, "img_ref")
    # get the number of images in the directory
    if not os.path.exists(base_image_path):
        raise FileNotFoundError(f"Image path '{base_image_path}' does not exist.")
    offset_dict = {}
    image_files = [f for f in os.listdir(base_image_path) if f.endswith('.jpg') or f.endswith('.png')]    

    detector.target_lpf = None

    last_seg_id = 0
    idx_seg_start = 0
    for i, image_filepath in enumerate(image_files):
        # split file name img_a_bbbb to seg:a, index: bbbb
        file_name_parts = image_filepath.split('_')
        if len(file_name_parts) < 2:
            print(f"Invalid file name format: {image_filepath}. Skipping.")
            continue
        segment_id = file_name_parts[1]
        if segment_id not in offset_dict:
            offset_dict[segment_id] = []
        if segment_id != last_seg_id:
            if last_seg_id != 0:
                print2(f"Processed segment {last_seg_id} with {i-idx_seg_start} images.", Color.YELLOW)
            last_seg_id = segment_id
            idx_seg_start = i

        image_raw = cv2.imread(os.path.join(base_image_path, image_filepath))
        image_carrier=ImageCarrier(image_raw, laser_id="laser0")
        offset_m = detector.process_image_debug(image_carrier=image_carrier)

