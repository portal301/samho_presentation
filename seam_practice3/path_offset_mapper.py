import numpy as np
import cv2
import json
import os
from sklearn.linear_model import RANSACRegressor, LinearRegression
# import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dev_assistant_utils import print2, Color
import io
import asyncio

class PathCorrectionMapper:
    def __init__(self, experiment_path="experiment", result_folder="cv_result"):
        self.experiment_path = experiment_path
        self.result_folder = result_folder
        self.offset_file_list = []
        self.offset_diff_stack = []
        self.offset_trends_fitting = []
        self.colors = ['blue', 'orange', 'green', 'red', 'pink', 'purple', 'brown', 'gray']

    def compute_offset_trends(self, offset_diff_data:dict):
        offset_trends_fitting = {}
        endpoint_offsets_fitted = {}
        for segment_id, data in offset_diff_data.items():
            # print("data:")
            # print(data)
            if data.ndim < 2 or data.shape[0] < 2:
                # Not enough data points for fitting segment {segment_id}.
                continue
            xs = data[:, 0]
            ys = data[:, 1]
            
            xs_reshaped = xs.reshape(-1, 1)
            ransac_regressor_linear = RANSACRegressor(LinearRegression())
            ransac_regressor_linear.fit(xs_reshaped, ys)
            fitted_diffs = ransac_regressor_linear.predict(xs_reshaped)
            offset_trends_fitting[segment_id] = np.array(list(zip(xs, fitted_diffs)))
            # print(f"Fitted trend for segment {segment_id}: {offset_trends_fitting[segment_id]}")
            # get endpoint offset from fitting
            endpoint_offsets_fitted[segment_id] = [
                offset_trends_fitting[segment_id][0],
                offset_trends_fitting[segment_id][-1]
            ]
        return offset_trends_fitting, endpoint_offsets_fitted
    
    def find_measure_point_index(self, path_measure_segmented:dict) -> dict:
        measure_point_index = {}
        for segment_id, waypoints in path_measure_segmented.items():
            measure_point_index[segment_id] = []
            for i, waypoint in enumerate(waypoints):
                if waypoint["type"] == "measure":
                    measure_point_index[segment_id].append(i)
        return measure_point_index

    def plot_single_trajectory(self, offset_dict:dict):
        plotable_segments = []

        offsets_aggregated = []
        for segment_id, offset_list in offset_dict.items():
            # check if the all list data in offset_file_list is None
            for offset in offset_list:
                if offset is not None:
                    plotable_segments.append(int(segment_id))
                    offsets_aggregated.extend([offset for offset in offset_list if offset is not None])
                    break


        # list up all offset data in a list
        y_min = min(offsets_aggregated)
        y_max = max(offsets_aggregated)


        plt.figure(figsize=(5*len(plotable_segments), 5))
        # Plot original offsets for segments 1 and 2
        for seg in plotable_segments:
            plt.subplot(1, len(plotable_segments), seg)
            # for i, offset_list in enumerate(offset_file_list):
            segment_id = str(seg)
            #     if segment_id not in offset_list:
            #         print(f"Segment {segment_id} not found in file {i}.")
            #         continue
            #     plt.plot(offset_list[segment_id], label=f"Segment {segment_id}", marker='.', 
            #              color=self.colors[i % len(self.colors)])
            plt.plot(offset_dict[segment_id], label=f"Segment {segment_id} Ref", marker='.', 
                     color="black", linewidth=0.5)
            # set y-axis limits
            plt.ylim(y_min, y_max)
            plt.title(f"Offsets for Segment {seg}")
            plt.xlabel("Sequence Index")
            plt.ylabel("Offset Value")
            plt.grid()
        plt.show()

