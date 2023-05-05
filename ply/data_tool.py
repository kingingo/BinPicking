import sys
import os
import os
import shutil
import open3d as o3d
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--file', type=str, default="data", help='Filename for the default data file [default: data]')
    parser.add_argument('--vis', type=str, default=None, help='Visualize File [default: None]')
    parser.add_argument('--print_info', type=bool, default=True, help='Print Information [default: True]')

    return parser.parse_args()

def load_data(filepath):
    file = open(filepath, 'r')   
    lines = file.readlines()
    start_reading = False
    
    positions = []
    colors = []
    labels = []
    header = ''
    for line in lines:
        if 'end_header' in line:
            start_reading = True
            header += line + ""
            continue
        
        if not start_reading:
            header += line + ""
        else:
            data = line.split()
            x = data[0]
            y = data[1]
            z = data[2]
            r = data[3]
            g = data[4]
            b = data[5]
            label = data[6]

            positions.append([float(x),float(y),float(z)])
            colors.append([int(r),int(g),int(b)])
            labels.append(int(label))
            
    return {'header': header, 'positions': positions, 'colors': colors, 'labels': labels}

def convert_data(data, callback):
    positions = data['positions']
    colors = data['colors']
    labels = data['labels']
    
    _positions = []
    _colors = []
    _labels = []
    
    length = len(data['positions'])
    for i in range(length):
        pos = positions[i]
        color = colors[i]
        label = labels[i]
        
        _pos, _color, _label = callback(pos, color, label)
        
        _positions.append(_pos)
        _colors.append(_color)
        _labels.append(_label)
        
    return {'header': data['header'], 'positions': _positions, 'colors': _colors, 'labels': _labels}

def down_sampling_data(data, print_info = True):
    positions = data['positions']
    colors = data['colors']
    labels = data['labels']
    length = len(positions)
    
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(positions, o3d.core.float32)
    pcd.point.colors = o3d.core.Tensor(colors, o3d.core.float32)
    pcd.point.labels = o3d.core.Tensor(labels, o3d.core.int32)
    
    filtered_pcd = pcd.voxel_down_sample(voxel_size=0.015)
    positions = []
    colors = []
    labels = []
    
    for i in range(len(filtered_pcd.point.positions)):
        x = filtered_pcd.point.positions[i][0].numpy()
        y = filtered_pcd.point.positions[i][1].numpy()
        z = filtered_pcd.point.positions[i][2].numpy()

        r = filtered_pcd.point.colors[i][0].numpy()
        g = filtered_pcd.point.colors[i][1].numpy()
        b = filtered_pcd.point.colors[i][2].numpy()

        label = filtered_pcd.point.labels[i].numpy()
        positions.append([x,y,z])
        colors.append([int(r), int(g), int(b)])
        labels.append(label)
    
    if print_info:
        _length = len(positions)
        print(f"Sampled data from {length} to {_length} down")    
    
    return {'header': data['header'], 'positions': positions, 'colors': colors, 'labels': labels}
        
def save_data(filepath, data, header = True):
    content = data['header'] if header else ''
    
    positions = data['positions']
    colors = data['colors']
    labels = data['labels']
    length = len(data['positions'])
    for i in range(length):
        pos = positions[i]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        color = colors[i]
        r = color[0]
        g = color[1]
        b = color[2]
        
        label = labels[i]
        
        content += "{} {} {} {} {} {} {}\n".format(x,y,z,r,g,b,label)
        
    f = open(filepath, 'wb')
    f.write(str(content).encode('utf-8'))
    f.close()

def draw_data(data):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(data["positions"], o3d.core.float32)
    pcd.point.colors = o3d.core.Tensor(data["colors"], o3d.core.float32)
    pcd.point.labels = o3d.core.Tensor(data["labels"], o3d.core.int32)
    o3d.visualization.draw([pcd], raw_mode = True)
 
def label_callback(pos, color, label):
    _pos = [pos[0], pos[1], pos[2]]
    _color = [color[0], color[1], color[2]]
    _label = label
    
    
    if int(_label) < 1:
        _label = 1
    
    return _pos, _color, _label 
        
def color_callback(pos, color, label):
    _pos = [pos[0], pos[1], pos[2]]
    _color = [color[0], color[1], color[2]]
    _label = label
    
    if int(_label) > 1:
        _color[0] = 0
        _color[1] = 100
        _color[2] = 0
    else:
        _color[0] = 139
        _color[1] = 0
        _color[2] = 0
        _label = 1
    
    return _pos, _color, _label 

def convert_files():
    folders = os.listdir(os.curdir)
    counter = 0
    for folder in folders:
        if not os.path.isdir(os.curdir+ "/"+folder):
            continue
        path = os.curdir + "/" + folder
        file = parser.file + ".ply"
        filepath = path + "/" + file
        
        splitted = folder.split("_")
        
        if len(splitted) >= 4:
            datetime = splitted[2] + "_" + splitted[3] + "-"  + splitted[5] + "-" + splitted[6]
        else:
            datetime = splitted[2] + "_" + splitted[3]
            
        counter+=1
        print(f"loading data from {filepath}")    
        data = load_data(filepath)
        
        down_sampled_data = down_sampling_data(data, parser.print_info)
        save_data(path + "/down_sampled_" + parser.file + ".ply", down_sampled_data)
        
        color_down_sampled_data = convert_data(down_sampled_data, color_callback)
        save_data(path + "/color_down_sampled_" + parser.file + ".ply", color_down_sampled_data)
        
        label_down_sampled_data = convert_data(down_sampled_data, label_callback)
        save_data(path + "/pre_down_sampled_" + parser.file + ".ply", label_down_sampled_data)
        save_data(path + "/trainings_" + parser.file + "_"+datetime+".txt", label_down_sampled_data)
            
        color_data = convert_data(data, color_callback)
        save_data(path + "/color_" + parser.file + ".ply", color_data)
        
        label_data = convert_data(data, label_callback)
        save_data(path + "/pre_" + parser.file + ".ply", label_data)
        
        print(f"        done")   
    print(f"Converted {counter} files")
    
if __name__ == "__main__":
    parser = parse_arguments()
    
    if parser.vis is not None:
        filepath = parser.vis
        
        if not os.path.isfile(filepath):
            print(f"file {filepath} not found!")
            exit()
        
        data = load_data(filepath)
        
        if parser.print_info:
            print(f"found {len(data['positions'])} points")
        
        draw_data(data)
    else:
        convert_files()
    