import cv2
import numpy as np
import open3d as o3d


W = 1920 
H = 1080


record_time = 20250418005607
timestamp = 10639122

image = cv2.imread(f'records/{record_time}/color_images/{timestamp}.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

depth = np.fromfile(f'records/{record_time}/depth_images/{timestamp}.raw', dtype=np.uint16)
# print("depth:", depth.shape, depth.dtype,
#       "min/max:", depth.min(), depth.max())
depth = depth.reshape((H, W))  # HxW
depth = depth.astype(np.float32)

fx, fy = 1123.87, 1123.03 
cx, cy = 948.027, 539.649 

us, vs = np.meshgrid(np.arange(W), np.arange(H))
zs = depth / 1000.0
xs = (us - cx) * zs / fx
ys = (vs - cy) * zs / fy
points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)

colors = image.reshape(-1, 3) / 255.0
valid = (zs > 0).reshape(-1)
points = points[valid]
colors = colors[valid]

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

real_point_cloud = o3d.io.read_point_cloud(f'records/{record_time}/point_clouds/{timestamp}.ply')
pts = np.asarray(real_point_cloud.points)
pts /= 1000.0
real_point_cloud.points = o3d.utility.Vector3dVector(pts)

real_point_cloud.paint_uniform_color([1.0, 0.0, 0.0])   # red
point_cloud.paint_uniform_color([0.0, 1.0, 0.0])        # green

o3d.visualization.draw_geometries([point_cloud, real_point_cloud])
# o3d.io.write_point_cloud(f'records/{record_time}/project_img_to_depth/{timestamp}.ply')
