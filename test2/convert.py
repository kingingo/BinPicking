import open3d as o3d
import open3d.core as o3c
import numpy as np
import os

def read_dir(path):
    print(f"open path {path}")
    data_directory = os.listdir(path)
    for entry in data_directory:
        new_path = path + '/' + entry
        if os.path.isdir(new_path):
            #read_dir(new_path);
            print(f'found {new_path}')
        else:
            if entry.endswith('.off'):
                convert_off_to_ply(new_path)
                #print(f'found {new_path}')
            elif entry.endswith('.ply'):
                #view(new_path)
                print('')

def view(path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.io.read_point_cloud(path, format="xyz")

    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

def delete(path):
    print(f"deleted {path}")
    os.remove(path)

def convert_off_to_ply(path):
    pcd = o3d.io.read_point_cloud(path, format="xyz")
    new_path = path.replace('.off','.ply')
    
    points = np.asarray(pcd.points)
    points = points.astype('float')
    pcd.points = o3d.utility.Vector3dVector(points)
    
    o3d.io.write_point_cloud(new_path, pcd, True)
    print(f"converted file {new_path}")
    

read_dir('data')