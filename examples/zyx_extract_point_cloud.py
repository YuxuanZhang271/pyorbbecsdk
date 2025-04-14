import os
import numpy as np
from plyfile import PlyData, PlyElement
from pyorbbecsdk import *
import argparse


def save_points_to_ply(frames: FrameSet, camera_param: OBCameraParam):
    if frames is None:
        return
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return
    points = frames.get_point_cloud(camera_param)
    if len(points) == 0:
        print("no depth points")
        return
    points_array = np.array([tuple(point) for point in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    points_filename = os.path.join(save_points_dir, "points_{}.ply".format(depth_frame.get_timestamp()))
    el = PlyElement.describe(points_array, 'vertex')
    PlyData([el], text=True).write(points_filename)

def save_color_points_to_ply(frames: FrameSet, camera_param: OBCameraParam):
    if frames is None:
        return
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return
    points = frames.get_color_point_cloud(camera_param)
    if len(points) == 0:
        print("no color points")
        return
    points_array = np.array([tuple(point) for point in points],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                   ('blue', 'u1')])
    points_filename = os.path.join(save_points_dir, "color_points_{}.ply".format(depth_frame.get_timestamp()))
    el = PlyElement.describe(points_array, 'vertex')
    PlyData([el], text=True).write(points_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--folder_path", required=True, type=str)
    root = parser.parse_args().folder_path

    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()
    has_color_sensor = False
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if depth_profile_list is None:
        print("No proper depth profile, can not generate point cloud")
        return
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is not None:
            color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            if device_pid == 0x066B:
                # Femto Mega does not support hardware D2C, and it is changed to software D2C
                config.set_align_mode(OBAlignMode.SW_MODE)
            else:
                config.set_align_mode(OBAlignMode.HW_MODE)
            has_color_sensor = True
    except OBError as e:
        config.set_align_mode(OBAlignMode.DISABLE)
        print(e)

    pipeline.start(config)
    pipeline.enable_frame_sync()
    saved_color_cnt: int = 0
    saved_depth_cnt: int = 0
    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            camera_param = pipeline.get_camera_param()
            saved_depth_cnt += save_points_to_ply(frames, camera_param)
            if has_color_sensor:
                saved_color_cnt += save_color_points_to_ply(frames, camera_param)
            if has_color_sensor:
                if saved_color_cnt >= 1 and saved_depth_cnt >= 1:
                    break
            elif saved_depth_cnt >= 1:
                break
        except OBError as e:
            print(e)
            break


if __name__ == "__main__":
    main()
