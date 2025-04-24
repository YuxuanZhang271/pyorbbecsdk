import cv2
from datetime import datetime as dt
import numpy as np
import os
from pyorbbecsdk import *
from queue import Queue
import sys
from typing import List
from utils import frame_to_bgr_image


MAX_DEVICES = 2
curr_device_cnt = 0

MAX_QUEUE_SIZE = 5

ESC_KEY = 27

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm


color_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
depth_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
stop_rendering = False


def on_new_frame_callback(frames: FrameSet, index: int):
    global color_frames_queue, depth_frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is not None:
        if color_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            color_frames_queue[index].get()
        color_frames_queue[index].put(color_frame)
    if depth_frame is not None:
        if depth_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            depth_frames_queue[index].get()
        depth_frames_queue[index].put(depth_frame)


def start_streams(pipelines: List[Pipeline], configs: List[Config]):
    index = 0
    for pipeline, config in zip(pipelines, configs):
        print(f"Starting device {index}")
        pipeline.start(config, lambda frame_set, 
                       curr_index=index: on_new_frame_callback(frame_set, curr_index))
        index += 1


def stop_streams(pipelines: List[Pipeline]):
    for pipeline in pipelines:
        pipeline.stop()


def main():
    root_path = "./records"
    os.makedirs(root_path, exist_ok=True)

    now = dt.now()
    data_time = now.strftime("%Y%m%d%H%M%S")
    sub_path = os.path.join(root_path, data_time)
    os.makedirs(sub_path, exist_ok=False)

    ctx = Context()
    devices = ctx.query_devices()
    global curr_device_cnt
    curr_device_cnt = devices.get_count()
    if curr_device_cnt <= 0: 
        print("No device connected")
        return
    elif curr_device_cnt > MAX_DEVICES: 
        print("Too many device connected")
        return

    pipelines: List[Pipeline] = []
    configs: List[Config] = []

    color_paths = []
    depth_paths = []

    for i in range(curr_device_cnt): 
        device = devices.get_device_by_index(i)
        pipeline = Pipeline(device)
        config = Config()

        device_path = os.path.join(sub_path, f"device{i}")
        os.makedirs(device_path, exist_ok=False)

        try: 
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            assert profile_list is not None
            color_profile = profile_list.get_default_video_stream_profile()
            assert color_profile is not None
            config.enable_stream(color_profile)

            color_path = os.path.join(device_path, "color_images")
            os.makedirs(color_path, exist_ok=False)
            color_paths.append(color_path)

            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            depth_profile = profile_list.get_default_video_stream_profile()
            assert depth_profile is not None
            config.enable_stream(depth_profile)

            depth_path = os.path.join(device_path, "depth_images")
            os.makedirs(depth_path, exist_ok=False)
            depth_paths.append(depth_path)

        except Exception as e:
            print(e)
            return
        
        config.set_align_mode(OBAlignMode.SW_MODE)
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(e)
        
        pipelines.append(pipeline)
        configs.append(config)
    
    global stop_rendering
    start_streams(pipelines, configs)

    try:
        while not stop_rendering:
            images = []
            for i in range(curr_device_cnt):
                color_frame = None
                depth_frame = None
                if not color_frames_queue[i].empty():
                    color_frame = color_frames_queue[i].get()
                if not depth_frames_queue[i].empty():
                    depth_frame = depth_frames_queue[i].get()
                if color_frame is None or depth_frame is None:
                    continue

                color_image = None
                depth_image = None
                color_path = color_paths[i]
                depth_path = depth_paths[i]

                timestamp = color_frame.get_timestamp()
                color_image = frame_to_bgr_image(color_frame)
                cv2.imwrite(os.path.join(color_path, f"{timestamp}.png"), color_image)

                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))
                depth_data = depth_data.astype(np.float32) * scale
                depth_data.tofile(os.path.join(depth_path, f"{timestamp}.raw"))

                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

                image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
                images.append(image)

                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:
                    stop_rendering = True
                    break

            if not images: 
                continue
            
            full_image = images[0].copy()
            for img in images[1:]:
                full_image = np.hstack((full_image, img))
            cv2.imshow("Devices", full_image)

            if stop_rendering: 
                break

        stop_streams(pipelines)
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        stop_rendering = True
        stop_streams(pipelines)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
