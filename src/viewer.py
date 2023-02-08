import open3d as o3d
from time import sleep


vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.io.read_point_cloud(f'data/toilet_0001.ply')

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])