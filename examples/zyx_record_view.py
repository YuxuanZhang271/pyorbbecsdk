import cv2
import os
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from datetime import datetime as dt
import sys
from plyfile import PlyData, PlyElement
import open3d as o3d

ESC_KEY = 27
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

def frame_to_depth_image(depth_frame: DepthFrame): 
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    data = data.reshape((height, width))
    data = data.astype(np.float32) * scale
    data = data.astype(np.uint16)
    return data


def main(argv):
    root_path = "./records"
    os.makedirs(root_path, exist_ok=True)

    now = dt.now()
    data_time = now.strftime("%Y%m%d%H%M%S")
    sub_path = os.path.join(root_path, data_time)
    os.makedirs(sub_path, exist_ok=False)

    pipeline = Pipeline()
    config = Config()

    try: 
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        assert profile_list is not None
        color_profile = profile_list.get_default_video_stream_profile()
        assert color_profile is not None
        config.enable_stream(color_profile)

        color_path = os.path.join(sub_path, "color_images")
        os.makedirs(color_path, exist_ok=False)

        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        config.enable_stream(depth_profile)

        depth_path = os.path.join(sub_path, "depth_images")
        os.makedirs(depth_path, exist_ok=False)

        pcd_path = os.path.join(sub_path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=False)
    except Exception as e:
        print(e)
        return
    
    config.set_align_mode(OBAlignMode.SW_MODE)

    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(e)

    pipeline.start(config)
    pipeline.start_recording(os.path.join(sub_path, "record.bag"))

    camera_param = pipeline.get_camera_param()
    print("Camera param: ", camera_param)

    try: 
        d2c = pipeline.get_d2c_valid_area(depth_profile, color_profile)
        print("d2c: ", d2c)
    except Exception as e: 
        print(e)

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            
            color_frame = frames.get_color_frame()
            if color_frame is None: 
                continue
            timestamp = color_frame.get_timestamp()
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None: 
                continue
            cv2.imwrite(os.path.join(color_path, f"{timestamp}.png"), color_image)

            depth_frame = frames.get_depth_frame()
            if depth_frame is None: 
                continue
            depth_image = frame_to_depth_image(depth_frame)
            if depth_image is None: 
                continue
            depth_image.tofile(os.path.join(depth_path, f"{timestamp}.raw"))

            depth_color_map = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color_map = cv2.applyColorMap(depth_color_map, cv2.COLORMAP_JET)
            
            image = cv2.addWeighted(color_image, 0.5, depth_color_map, 0.5, 0)
            cv2.imshow("Viewer ", image)

            # points = frames.get_color_point_cloud(camera_param)
            # if len(points) == 0:
            #     continue
            # points_np = np.asarray(points)
            # xyz = points_np[:, :3].astype(np.float32)
            # # Ensure colors are in the [0, 1] range for Open3D
            # colors = points_np[:, 3:].astype(np.uint8) / 255.0

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd.colors = o3d.utility.Vector3dVector(colors)

            # points_filename = os.path.join(pcd_path, f"{timestamp}.ply")
            # # Using binary format (write_ascii=False) tends to be faster.
            # o3d.io.write_point_cloud(points_filename, pcd, write_ascii=False)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                pipeline.stop_recording()
                break
        except KeyboardInterrupt:
            pipeline.stop_recording()
            break
    pipeline.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
