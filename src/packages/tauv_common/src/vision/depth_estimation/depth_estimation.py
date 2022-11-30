
import numpy as np 
from geometry_msgs.msg import Point


class DepthEstimator():
  def estimate_relative_depth(depth_image, x, y, w, bbox):
      box = depth_image[max(bbox.ymin, y - w) : min(bbox.ymax, y + w+1), max(bbox.xmin, x - w) : min(bbox.xmax, x + w + 1)]
      return np.nanmean(box)

  def estimate_absolute_depth(depth_image, bbox, depth_camera_info):
    fx = depth_camera_info.K[0]
    cx = depth_camera_info.K[2]
    fy = depth_camera_info.K[4]
    cy = depth_camera_info.K[5]

    width_to_run = 20

    center_x = (bbox.xmin + bbox.xmax) // 2
    center_y = (bbox.ymin + bbox.ymax) // 2
    cur_depth = DepthEstimator.estimate_relative_depth(depth_image, center_x, center_y, width_to_run, bbox)

    if (cur_depth != np.nan):
      cur_x = ((center_x - cx) * cur_depth) / (fx)
      cur_y = ((center_y - cy) * cur_depth) / (fy)
      
      return (cur_x, cur_y, cur_depth * 1.49)
    
    return cur_depth