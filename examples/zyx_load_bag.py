import cv2
import numpy as np
import os
from pyorbbecsdk import Pipeline
from utils import frame_to_bgr_image


bag_paths = ['/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/record/test_0.bag', 
             '/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/record/test_1.bag']

for bag_path in bag_paths: 
    root_path = os.path.dirname(bag_path)

    filename = os.path.basename(bag_path)
    name, _ext = os.path.splitext(filename)
    root_path = os.path.join(root_path, name)
    os.makedirs(root_path, exist_ok=True)

    color_path = os.path.join(root_path, 'color')
    os.makedirs(color_path, exist_ok=True)
    depth_path = os.path.join(root_path, 'depth')
    os.makedirs(depth_path, exist_ok=True)

    pipeline = Pipeline(bag_path)
    pipeline.start()

    idx = 0
    t0 = None
    while True:
        frames = pipeline.wait_for_frames(5000)
        if frames is None:
            break

        depth = frames.get_depth_frame()
        if depth is None:
            print("no depth frame")
            idx += 1
            continue
        color = frames.get_color_frame()
        if color is None:
            print("no color frame")
            idx += 1
            continue

        bgr = frame_to_bgr_image(color)
        if bgr is None:
            print("failed to convert color")
            idx += 1
            continue

        ts = depth.get_timestamp()
        if t0 is None:
            t0 = ts
        dt = ts - t0

        cv2.imwrite(os.path.join(color_path, f"color_{idx}_{dt}.jpg"), bgr)

        h, w = depth.get_height(), depth.get_width()
        scale = depth.get_depth_scale()
        data = (np.frombuffer(depth.get_data(), np.uint16)
                    .reshape((h, w))
                    .astype(np.float32) * scale)
        data.tofile(os.path.join(depth_path, f"depth_{idx}_{dt}.raw"))    

        idx += 1

    pipeline.stop()
