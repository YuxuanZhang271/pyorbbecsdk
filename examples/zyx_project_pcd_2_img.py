import cv2
import numpy as np
import open3d as o3d
from open3d.visualization.rendering import Camera


W = 1920 
H = 1080


record_time = 20250418005607
timestamp = 10639122

point_cloud = o3d.io.read_point_cloud(f'records/{record_time}/point_clouds/{timestamp}.ply')

renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
scene = renderer.scene
scene.set_background([0, 0, 0, 1])

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
scene.add_geometry("point cloud", point_cloud, mat)

cam = renderer.scene.camera
cam.look_at(
    center  = point_cloud.get_center(), 
    eye     = [0, 0, 1], 
    up      = [0, -1, 0]
)
cam.set_projection(60.0, float(W)/H, 0.01, 10.0, Camera.FovType.Vertical)

image_o3d = renderer.render_to_image()
image_np = np.asarray(image_o3d)
img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

real_image = cv2.imread(f'records/{record_time}/color_images/{timestamp}.png')

combined = np.vstack((img_bgr, real_image))

cv2.imshow("Rendered Point Cloud", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
