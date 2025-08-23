import cv2
import numpy as np
from scipy.signal import savgol_filter
import json
import os
import sys
import argparse
from dev_assistant_utils import print2, Color
from path_offset_mapper import PathCorrectionMapper


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

        self.target_lpf = None
        self.tau_lpf = 0.6
        self.load_camera_lens_data()

        # image filter parameters
        self.baseline_curvature_threshold = 0.14
        self.hls_filter_lightness_threshold = 70
        self.reward_weight_curvature = 1.0
        self.reward_weight_voting = -1.0
        self.reward_weight_lpf = -20.0 # low pass filter weight


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


    def analyze(self, image_carrier:ImageCarrier=None, save_roi_image=False, debug=False):
        color = self.config[image_carrier.laser_id]["color"]
        print2(f"Color: {color}", Color.YELLOW)

        # sharpening the image src
        image_carrier.img_src = self.img_sharpening(image_carrier.img_src)
        image_carrier.color_mask = self.get_color_hls_mask(image_color=image_carrier.img_src, color=color)

        img_src_hls = cv2.cvtColor(image_carrier.img_src, cv2.COLOR_BGR2HLS)
        mask_hls_pre =  cv2.inRange(img_src_hls, (0, 20, 0), (255, 255, 255))
        mask_hls_pre = cv2.bitwise_and(mask_hls_pre, image_carrier.color_mask)
        mask_hls_pre = self.remove_small_islands_from_mask(mask=mask_hls_pre, min_area=20, morphology_filter=True)
        mask_hls = cv2.inRange(img_src_hls, (0, self.hls_filter_lightness_threshold, 0), (255, 255, 255))
        # use color mask to filter the hls mask
        mask_hls = cv2.bitwise_and(mask_hls, image_carrier.color_mask)

        max_x = np.max(np.where(mask_hls > 0)[1])
        indices2 = np.where(mask_hls[:, max_x] > 0)[0]
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
        mask_roi_hls2 = mask_hls[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        # apply morphology to the mask_roi_hls_hsv
        kernel = np.ones((3, 3), np.uint8)

        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
        mask_roi = self.remove_small_islands_from_mask(mask=mask_roi, min_area=20, morphology_filter=True)
        mask_roi_hls2 = cv2.morphologyEx(mask_roi_hls2, cv2.MORPH_CLOSE, kernel)
        mask_roi_hls2 = cv2.morphologyEx(mask_roi_hls2, cv2.MORPH_OPEN, kernel)
        mask_roi_hls = mask_hls_pre[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        indices = np.where(mask_roi_hls != 0)
        indices2 = np.where(mask_roi_hls2 != 0)
        
        mask_roi_overlay = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        mask_roi_overlay[indices[0], indices[1]] = [0, 150, 255]  # yello color for overlay
        mask_roi_overlay[indices2[0], indices2[1]] = [0, 0, 255]  # red color for overlay

        if debug==True: # debug mode
            # hstack_img = np.hstack((image_carrier.img_roi, mask_roi_overlay, img_roi_sharpened, cv2.cvtColor(mask_roi2, cv2.COLOR_GRAY2BGR)))
            hstack_img = np.hstack((image_carrier.img_roi, mask_roi_overlay))
            hstack_img = cv2.resize(hstack_img, (0, 0), fx=4, fy=4)
            cv2.imshow("ROI and Mask", hstack_img)
            # cv2.waitKey(0)

        mask_roi_color = mask_roi_overlay
        # compute the curvature of the path 'baseline'
        baseline = self.compute_baseline_points(mask=mask_roi)

        x_baseline, y_baseline = zip(*baseline)
        x_baseline = np.array(x_baseline)
        y_baseline = np.array(y_baseline)
        # Smooth the y baseline values using a Savitzky-Golay filter
        # baseline_smooth = savgol_filter(np.array(baseline), window_length=11, polyorder=5, axis=0)
        baseline_smooth = savgol_filter(np.array(baseline), window_length=21, polyorder=1, axis=0)
        y_baseline_smooth = baseline_smooth[:, 1]
        x_baseline_smooth = baseline_smooth[:, 0]

        macro_dy = np.gradient(y_baseline_smooth, x_baseline_smooth) # first derivative (dy/dx)
        macro_ddy = np.gradient(macro_dy, x_baseline_smooth)  # second derivative (d²y/dx²)
        macro_curvature = macro_ddy / (1 + macro_dy**2)**1.5  # curvature: k = |y''| / (1 + (y')^2)^(3/2)

        # get the first point index where dy < 0 and abs(curvature) < 0.01
        idx_major_slope_start = np.where((macro_dy < -0.2) & (np.abs(macro_curvature) < 0.005))[0][0]
        if idx_major_slope_start.size > 0:
            major_slope = macro_dy[idx_major_slope_start]*0.7
            print2(f"major slope: {major_slope}", Color.RED)


        baseline_smooth = savgol_filter(np.array(baseline), window_length=3, polyorder=1, axis=0)
        y_baseline_smooth = baseline_smooth[:, 1]
        x_baseline_smooth = baseline_smooth[:, 0]
        dy = np.gradient(y_baseline_smooth, x_baseline_smooth) # first derivative (dy/dx)
        ddy = np.gradient(dy, x_baseline_smooth)  # second derivative (d²y/dx²)
        curvature = ddy / (1 + dy**2)**1.5  # curvature: k = |y''| / (1 + (y')^2)^(3/2)


        # find the point where curvature > threshold and draw a circle
        self.curvature_threshold = 0.14
        cost = [None] * len(curvature)
        for i in range(len(curvature)):
            if self.target_lpf is None:
                dist = 0
            else:
                dist = np.linalg.norm((baseline_smooth[i]+np.array([roi_x[0], roi_y[0]]).astype(float) - self.target_lpf))
            if curvature[i] > self.curvature_threshold and i > idx_major_slope_start: 
                vector_ref = [1, major_slope]
                vector_displacement = np.array(baseline_smooth[i]) - np.array(baseline_smooth[idx_major_slope_start])
                cross_product = np.cross(vector_ref, vector_displacement)
                cost[i] = cross_product
 

 
        if np.all(np.array(cost) == None):
            print("No valid cost found. Setting target to None.")
            image_carrier.target = None
            image_carrier.offset = None
            return
        optimal_index = np.argmin([value if value is not None else np.inf for value in cost])

        try:
            image_carrier.baseline_smooth = baseline_smooth
            image_carrier.baseline_curvature = curvature
            image_carrier.target = (baseline_smooth[optimal_index]+ np.array([roi_x[0], roi_y[0]])).astype(int)
            image_carrier.offset = image_carrier.target[0] - image_carrier.w //2    
            if self.target_lpf is None:
                    self.target_lpf = image_carrier.target
            else:
                self.target_lpf = (1-self.tau_lpf)*image_carrier.target+self.tau_lpf*self.target_lpf

        except Exception as e:
            print(f"Error in calculating target: {e}")
            image_carrier.target = None
            image_carrier.offset = None
        finally:
            print(f"Target: {image_carrier.target}")
    
        if save_roi_image is True:
            img_scaler = 4
            image_carrier.img_roi_baseline = image_carrier.img_roi.copy()
            img_roi_result = cv2.resize(image_carrier.img_roi_baseline, (0, 0), fx=img_scaler, fy=img_scaler)
            for point in baseline_smooth:
                cv2.circle(img_roi_result, (int(point[0]*img_scaler), int(point[1]*img_scaler)), 2, (220, 150, 0), -1)

            fontsize = 0.4
            text_dir = 1 
            for i in range(len(macro_curvature)):
                if cost[i] is not None:
                    cv2.circle(img_roi_result, (int(img_scaler*x_baseline_smooth[i]), int(img_scaler*y_baseline_smooth[i])), 2, (0, 100, 230), -1, lineType=cv2.LINE_AA)
                    # draw line from passing baseline's y=y(major_slope) with slope dy=dy(major_slope) to the point
                    cv2.putText(img_roi_result, f"{cost[i]:.1f}", (int(img_scaler*x_baseline_smooth[i]), int(img_scaler*y_baseline_smooth[i]) + text_dir*15), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 160, 255), 1, cv2.LINE_AA)
                    cv2.putText(img_roi_result, f"C{macro_curvature[i]:.1f}", (int(img_scaler*x_baseline_smooth[i]), int(img_scaler*y_baseline_smooth[i]) + text_dir*30), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (150, 150, 150), 1, cv2.LINE_AA)
                    cv2.putText(img_roi_result, f"D{dist:.1f}", (int(img_scaler*x_baseline_smooth[i]), int(img_scaler*y_baseline_smooth[i]) + text_dir*45), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (150, 150, 150), 1, cv2.LINE_AA)
                    text_dir *= -1  # alternate text direction for better visibility
            cv2.line(img_roi_result, 
                    (int(img_scaler*x_baseline_smooth[idx_major_slope_start]), int(img_scaler*y_baseline_smooth[idx_major_slope_start])),
                    (int(img_scaler*(x_baseline_smooth[len(x_baseline_smooth)-1])), int(img_scaler*(y_baseline_smooth[idx_major_slope_start]+ major_slope*(x_baseline_smooth[len(x_baseline_smooth)-1]-x_baseline_smooth[idx_major_slope_start])))),
                    (0, 255, 0), 1, lineType=cv2.LINE_AA)

            cv2.circle(img_roi_result, (int(img_scaler*x_baseline_smooth[optimal_index]), int(img_scaler*y_baseline_smooth[optimal_index])), 3, (0, 0, 255), -1)
            mask_roi_color = cv2.resize(mask_roi_color, (0, 0), fx=img_scaler, fy=img_scaler)
            combined = np.vstack((mask_roi_color, img_roi_result))
            image_carrier.img_roi_analyzed = combined.copy()            
        return


    def process_image_debug(self, image_carrier:ImageCarrier ,display:bool=False, display_detail:bool=False, debug:bool=False):
        """Processes the image to extract the laser line."""
        rotate = self.config[image_carrier.laser_id]["rotate"]
        y_invert = self.config[image_carrier.laser_id]["invert_y_axis"]
        self.img_src = image_carrier.rotate_image(rotate=rotate, y_invert=y_invert)

        try:
            self.analyze(image_carrier=image_carrier, save_roi_image=display_detail, debug=debug)
            offset = image_carrier.offset
            offset_metric = float(offset)*self.px_to_m
            print(f"Offset: {offset}px, {offset_metric*1000.0:.3f} mm")

        except Exception as e:
            print(f"Error in analyze: {e}")
            image_carrier.target = None
            image_carrier.offset = None
            image_carrier.img_roi_analyzed = None
            image_carrier.img_result = None
            return None, None
        res_image = self.get_result_image(image_carrier=image_carrier)
        
        if display is True:
            window_name = "result"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, res_image)
            cv2.waitKey(10)
        return offset_metric, res_image

    def compute_baseline_points(self, mask:np.ndarray=None):
        """
        For each x position in the image, computes the mean y-coordinate
        of the detected color pixels.
        """
        baseline_points = []
        for x in range(mask.shape[1]):
            y_indices = np.where(mask[:, x] > 0)[0]
            if len(y_indices) > 0:
                # center_y = int(np.mean(y_indices))
                center_y = int(np.max(y_indices))
                baseline_points.append((x, center_y))
        if len(baseline_points) == 0:
            raise ValueError("error in compute_baseline_points: no baseline points found.")
        return baseline_points # return the last point as a reference for filtering
    
    def save_analyzed_image(self, image_carrier:ImageCarrier, save_path:str):
        cv2.imwrite(save_path, image_carrier.img_result)
        print(f"Annotated image saved to {save_path}")

    def get_result_image(self, image_carrier:ImageCarrier, display=False):
        """Displays the annotated image."""
        img_bbox = image_carrier.img_src.copy()
        img_result = image_carrier.img_src.copy()

        center_x = image_carrier.w // 2
        cv2.line(img_result, (center_x, 0), (center_x, image_carrier.h), (255, 0, 0), 1)
        if image_carrier.target is not None:
            distance = image_carrier.offset
            offset_metric = float(image_carrier.offset)*self.px_to_m
            # text = f"Offset: {distance}px"
            text = f"Offset: {offset_metric*1000.0:.3f} mm"

            cv2.line(img_result, (image_carrier.target[0], 0), (image_carrier.target[0], image_carrier.target[1]-5), (0, 0, 255), 2)
        else:
            text = f"Offset: None"
            distance = None

        cv2.putText(img_result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.rectangle(img_bbox, (int(image_carrier.roi_window[0][0]), int(image_carrier.roi_window[1][0])), (int(image_carrier.roi_window[0][1]), int(image_carrier.roi_window[1][1])), (255, 0, 255), 2)
        image_carrier.img_result = np.hstack((img_bbox, img_result))

        if image_carrier.img_roi_analyzed is not None:
            roi_y2x_ratio = float(image_carrier.img_roi.shape[1])/ float(image_carrier.img_roi.shape[0])
            roi_image = cv2.resize(image_carrier.img_roi_analyzed, (int(float(image_carrier.img_result.shape[0])*roi_y2x_ratio), image_carrier.img_result.shape[0]))
            image_carrier.img_result = np.hstack((image_carrier.img_result, roi_image))
        return image_carrier.img_result

    def load_camera_lens_data(self):
        """ Loads camera lens calibration data from a JSON file."""
        with open(os.path.join("data", "hw_calibration", "camera_lens.json"), "r") as f:
        # with open(os.path.join("..","data", "hw_calibration", "camera_lens.json"), "r") as f:
            camera_lens_data = json.load(f)
            px_to_m = camera_lens_data["settings"]["px_to_m"]
            print(f"px_to_m: {px_to_m}")
        self.px_to_m = px_to_m
        return px_to_m
    
        
# === Example Usage ===
# python seam_practice3/seamtracker.py -p data/light
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
        offset_m, res_image = detector.process_image_debug(image_carrier=image_carrier,display=True, display_detail=True, debug=True)
        print(f"Processed image {i+1}/{len(image_files)}: {image_filepath}")
        offset_dict[segment_id].append(offset_m if offset_m is not None else None)

        if res_image is None:
            print(f"Skipping image {image_filepath} due to processing error.")
            continue
        # save the result image
        save_image_path = os.path.join(base_image_path, "res_image", f"{segment_id}_{i+1-idx_seg_start}.png")
        if not os.path.exists(os.path.dirname(save_image_path)):
            os.makedirs(os.path.dirname(save_image_path))
        cv2.imwrite(save_image_path, res_image)
        print2(f"Result image saved to {save_image_path}", Color.GREEN)

    savepath = os.path.join(base_image_path, "results")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open(os.path.join(savepath, "cv_result_seam.json"), "w") as f:
        json.dump(offset_dict, f, indent=4)
    print2(f" cv_result saved to {os.path.join(savepath, 'cv_result_seam.json')}", Color.GREEN)

    # display the result
    offset_mapper = PathCorrectionMapper()
    offset_mapper.plot_single_trajectory(
        offset_dict=offset_dict)