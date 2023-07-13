import sys
import os
import os.path as osp
import shutil
import argparse
import numpy as np
from datetime import datetime as dt
import pickle
import ctypes
import math
import struct
import colorsys
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie1976
from colormath.color_conversions import convert_color

OPEN3D = None
try:
    import open3d as o3d
    OPEN3D = True
except:
    OPEN3D = False

PYMESH = None
try:
    import pymesh
    from mesh_to_point import farthest_point_sampling
    PYMESH = True
except:
    PYMESH = False

def parse_arguments():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--format', type=str, default='x,y,z,r,g,b,label', help="Sets the Format for the data load Method")
    parser.add_argument('--file', type=str, default="data", help='Filename for the default data file [default: data]')
    parser.add_argument('--vis', type=str, default=None, help='Visualize File [default: None]')
    parser.add_argument('--vis_coloured', action = "store_true",  help='Visualize Label')
    parser.add_argument('--vis_rgb', action = "store_true",  help='Visualize RGB Convert')
    parser.add_argument('--with_pred_label', action = "store_true", help='Visualize File with Prediction Label')
    parser.add_argument('--clean', action = "store_true", help='Deletes all generated Files [default: False]')
    parser.add_argument('--renamed_old_data', action = "store_true", help='Rename old data files')
    parser.add_argument('--disable_info', action = "store_true", help='Disable print information')
    parser.add_argument('--voxel_size', type=float, default=0.015, help='Voxel Size for down sampling [default: 0.015]')
    parser.add_argument('--nb_neighbors', type=int, default=20, help='Remove statistical outliers neighbours [default: 20]')
    parser.add_argument('--std_ratio', type=float, default=2.0, help='Remove statistical outliers std_ratio [default: 2.0]')
    parser.add_argument('--gather', action = "store_true", help='Gather all training files in folder training/')
    parser.add_argument('--convert_obj', type=str, default=None, help='Convert Objekt [default: None]')
    parser.add_argument('--num_samples', type=int, default=1000, help='Convert Objekt in "num_samples" points [default: 1000]')
    parser.add_argument('--conv_xyz', action = "store_true",  help='Convert MeshLab XYZ File and add RGB')
    parser.add_argument('--conv_ply',action = "store_true",  help='Convert XYZ File to PLY')
    parser.add_argument('--rgb', type=str, default="0,0,0", help='RGB colour [default: 0,0,0]')
    parser.add_argument('--path', type=str, default=None, help='Path to file [default: None]')
    parser.add_argument('--str_fields', type=str, default='x,y,z,r,g,b,label,clabel', help='Data fields loaded [default: x,y,z,r,g,b,label,clabel]')
    parser.add_argument('--info', action = "store_true", help='Shows information about the file')
    parser.add_argument('--labeling', action = "store_true", help='New data labeling')
    parser.add_argument('--labeling_all', action = "store_true", help='New data labeling for all')
    parser.add_argument('--twoD', action = "store_true", help='Show in 2D')
    
    return parser.parse_args()

def callback_filter_points_by_bbox(pos, color, label, clabel):
    '''
    y min:-0.479139149189 max:0.479139208794
    x min:-0.329147785902 max:0.329147785902
    z min:0.00354268550873 max:0.360492372513
    '''
    y_min = -0.329147785902
    y_max = 0.329147785902
    x_min = -0.479139149189
    x_max = 0.479139208794
    
    if (pos[0] < x_min or pos[0] > x_max) \
        or (pos[1] < y_min or pos[1] > y_max):
        return pos, color, label, clabel, True
    return pos, color, label, clabel, False


def load_checkpoint(path):
    checkpoint_path = path + "/checkpoint.p"
    print("\nload checkpoint {}".format(checkpoint_path))
    file = open(checkpoint_path, 'rb')
    checkpoint = pickle.load(file)

    if checkpoint is not None:
      xpoint = checkpoint['xpoint']
      ypoint = checkpoint['ypoint']
      zpoint = checkpoint['zpoint']
      labelpoint = checkpoint['labelpoint']
      rgbpoint = checkpoint['rgbpoint']
      
      points = []
      
      for i in range(len(xpoint)):
          points.append({"x": xpoint[i], "y": ypoint[i], "z": zpoint[i], "rgb": rgbpoint[i], "label": labelpoint[i]})
      return points
    return False

def find_entry_checkpoint(checkpoint_data, x, y, z):
    length = len(checkpoint_data)
    
    for i in range(length):
        data = checkpoint_data[i]
        
        if data['x'] == x and data['y'] == y and data['z'] == z:
            return data
        
    return None

def callback_clabels(pos, color, label, clabel):
    color_codes = {
    '1': [3,67,223], #Blue
    '2': [255,225,20], #Yellow
    '3': [229,0,0], #Red
    '4': [249,115,6], #Orange
    }
    min_diff_label = 0
    min_diff = 10000
    e_min_diff = 1000
    
    for _clabel in color_codes:
        #hdiff = hsv_diff(color, color_codes[_clabel])
        e_diff = deltaE_diff(color, color_codes[_clabel])
        #diff = rgb_diff(color, color_codes[_clabel])
        
        if e_diff < e_min_diff: #and diff < min_diff
            e_min_diff = e_diff
            #min_diff = diff
            min_diff_label = _clabel  
            #h_min_diff = hdiff
    return pos, color, label, int(min_diff_label), False

def color_distance(rgb1, rgb2):
    rm = 0.5 * (rgb1[0] + rgb2[0])
    rd = ((2 + rm) * (rgb1[0] - rgb2[0])) ** 2
    gd = (4 * (rgb1[1] - rgb2[1])) ** 2
    bd = ((3 - rm) * (rgb1[2] - rgb2[2])) ** 2
    return (rd + gd + bd) ** 0.5

def deltaE_diff(c1, c2):
    # Reference color.
    color1 = sRGBColor(c1[0],c1[1],c1[2], True)
    # Color to be compared to the reference.
    color2 = sRGBColor(c2[0],c2[1],c2[2], True)
    
    lab1 = convert_color(color1, LabColor)
    lab2 = convert_color(color2, LabColor)
    # This is your delta E value as a float.
    delta_e = delta_e_cie1976(lab1, lab2)
    
    return delta_e

def hsv_diff(c1, c2):
    c1 = colorsys.rgb_to_hsv(c1[0],c1[1],c1[2]) 
    c2 = colorsys.rgb_to_hsv(c2[0],c2[1],c2[2]) 
    
    h0, s0, v0 = c1
    h1, s1, v1 = c2
    
    dh = min(abs(h1-h0), 360-abs(h1-h0)) / 180.0
    ds = abs(s1-s0)
    dv = abs(v1-v0) / 255.0
    return math.sqrt(dh*dh+ds*ds+dv*dv)

def rgb_diff(c1, c2):
    return math.sqrt(math.pow((c2[0]-c1[0]),2) + math.pow((c2[1]-c1[1]),2) + math.pow((c2[2]-c1[2]),2))

def load_data(filepath, str_fields = "x,y,z,r,g,b,label,clabel"):
    splitted = filepath.split(".")
    file_extension = splitted[len(splitted)-1]
    
    file = open(filepath, 'r')   
    lines = file.readlines()
    start_reading = True
    
    fields = str_fields.split(",")
    
    if file_extension == 'ply':
        start_reading = False
    
    positions = []
    colors = []
    labels = []
    clabels = []
    pred_labels = []
    
    header = ''
    for line in lines:
        if file_extension == 'ply':
            if 'end_header' in line:
                start_reading = True
                header += line + ""
                continue
            
            if not start_reading:
                header += line + ""
                
        if start_reading:     
            data = line.split()
            x = None
            y = None
            z = None
            r = None
            g = None
            b = None
            label = None
            clabel = None
            pred_label = None
            
            for i in range(len(fields)):
                field = fields[i]
                
                if len(data) <= i:
                    print("data doesn't have enough fields! len={}".format(len(fields)))
                    break
                
                if field == 'x':
                    x = data[i]
                elif field == 'y':
                    y = data[i] 
                elif field == 'z':
                    z = data[i] 
                elif field == 'r':
                    r = data[i] 
                elif field == 'g':
                    g = data[i] 
                elif field == 'b':
                    b = data[i] 
                elif field == 'label':
                    label = data[i] 
                elif field == 'clabel':
                    clabel = data[i] 
                elif field == 'pred_label':
                    pred_label = data[i] 
                else:
                    print("field not found! {}".format(data[i]))
                    
            if x is not None and y is not None and z is not None:
                positions.append([float(x),float(y),float(z)])
            if r is not None and g is not None and b is not None:
                colors.append([int(r),int(g),int(b)])
            if label is not None:
                labels.append(int(label))
            if clabel is not None:
                clabels.append(int(clabel))
            if pred_label is not None:
                pred_labels.append(int(pred_label))
            
    data = {'fields':fields}
    
    if len(positions) > 0:
        data['positions'] = positions
    if len(labels) > 0:
        data['labels'] = labels
    if len(clabels) > 0:
        data['clabels'] = clabels
    if len(pred_labels) > 0:
        data['pred_labels'] = pred_labels
    if len(positions) > 0:
        data['colors'] = colors
    if len(header) > 0:
        data['header'] = header
    
    return data

def convert_data(data, callback):
    positions = data['positions']
    colors = data['colors']
    
    if 'labels' in data:
        labels = data['labels']
    else:
        labels = None
    
    if 'clabels' in data:
        clabels = data['clabels']
    else:
        clabels = None
    
    _positions = []
    _colors = []
    _labels = []
    _clabels = []
    
    length = len(data['positions'])
    for i in range(length):
        pos = positions[i]
        color = colors[i]
        label = labels[i] if labels is not None else None
        clabel = clabels[i] if clabels is not None else None
        
        _pos, _color, _label, _clabel, remove = callback(pos, color, label, clabel)
        
        if not remove:
            _positions.append(_pos)
            _colors.append(_color)
            if _label is not None:
                _labels.append(_label)
            if _clabel is not None:
                _clabels.append(_clabel)
        
    length = len(_positions)
    _data = {}
    _data['fields'] = data['fields']
    _data['positions'] = _positions
    _data['colors'] = _colors
    
    if 'header' in data:
        _data['header'] = data['header']
        
    if len(_labels) == length:
        _data['labels'] = _labels
        
    if len(_clabels) == length:
        _data['clabels'] = _clabels
        
    return _data

def down_sampling_data(data, disable_info = False, _voxel_size = 0.015, label_field = 'labels'):
    positions = data['positions']
    colors = data['colors']
    labels = data[label_field]
    length = len(positions)
    
    
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(positions, o3d.core.float32)
    pcd.point.colors = o3d.core.Tensor(colors, o3d.core.float32)
    pcd.point.labels = o3d.core.Tensor(labels, o3d.core.int32)
    
    filtered_pcd = pcd.voxel_down_sample(voxel_size = _voxel_size)
    cl, ind = filtered_pcd.remove_statistical_outliers(nb_neighbors=20,
                                                        std_ratio=2.0)
    filtered_pcd = filtered_pcd.select_by_mask(ind)
    #outlier_cloud = filtered_pcd.select_down_sample(ind, invert=True)
    
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

        label = filtered_pcd.point.labels[i].numpy().tolist()
        positions.append([x,y,z])
        colors.append([int(r), int(g), int(b)])
        labels.append(label)
    
    if not disable_info:
        _length = len(positions)
        print(f"\n Sampled data from {length} to {_length} down")    
    
    data['fields'] = ['x', 'y', 'z', 'r', 'g', 'b', label_field[0:len(label_field)-1]]
    
    return {'header': data['header'], 'fields': data['fields'], 'positions': positions, 'colors': colors, label_field: labels}
        
def save_data(filepath, data, header = True, str_fields = None):
    
    if str_fields is None:
        fields = data['fields']
    else:
        fields = str_fields.split(',')
    
    positions = data['positions']
    length = len(positions)
    
    if header:
        if 'header' in data:
            content = data['header']
        else:
            content = create_ply_header(length,','.join(fields))
    else:
        content = ''
    
    for i in range(length):
        for j in range(len(fields)):
            field = fields[j]
            
            if field == 'x':   
                content += str(data['positions'][i][0])
            elif field == 'y':   
                content += str(data['positions'][i][1]) 
            elif field == 'z':   
                content += str(data['positions'][i][2]) 
            elif field == 'r':   
                content += str(data['colors'][i][0]) 
            elif field == 'g':   
                content += str(data['colors'][i][1]) 
            elif field == 'b':   
                content += str(data['colors'][i][2]) 
            elif field == 'label':   
                content += str(data['labels'][i]) 
            elif field == 'clabel':   
                content += str(data['clabels'][i]) 
            else:
                print("Field not found!! {}".format(field))
                
            content += ' '
        content = content[0:len(content)-1]
        content += '\n'     

    f = open(filepath, 'wb')
    f.write(str(content).encode('utf-8'))
    f.close()

def draw_data(data, draw_callback = None):
    pcd = o3d.t.geometry.PointCloud()
    
    if parser.with_pred_label and draw_callback is None:
        vol = len(data["labels"])
        
        for i in range(vol):
            label = data["labels"][i]
            pred_label = data["pred_labels"][i]
            
            if int(label) == int(pred_label):
                data["colors"][i] = GREEN
            else:
                data["colors"][i] = RED
                
    positions = []
    if draw_callback is not None:
        for i in range(len(data['positions'])):
            pos = data['positions'][i]
            color = data['colors'][i]
            label = data['labels'][i]
            clabel = data['pred_labels'][i] if 'pred_labels' in data else label
            data['colors'][i] = draw_callback(pos,color,label, clabel)
    
    if parser.twoD:
        data = convert_data(data, callback_2d)
        
    pcd.point.positions = o3d.core.Tensor(data["positions"], o3d.core.float32)
    pcd.point.colors = o3d.core.Tensor(data["colors"], o3d.core.float32)
    pcd.point.labels = o3d.core.Tensor(data["labels"], o3d.core.int32)
    
    o3d.visualization.draw(
        [pcd], 
        raw_mode = True,
        lookat = [0,0,0],
        eye = [0,0,-0.1],
        up = [0,1,0.1],
        field_of_view = 10 ,
        point_size = 2,
        bg_color = (1,1,1,0)
    )
 
def callback_2d(pos, color, label, clabel):
    return [pos[0],pos[1],0], color, label, clabel, False

GREEN = [0,255,0]
RED = [229,0,0]
YELLOW = [255,255,0]
BLUE = [0,0,255]
LIGHTBLUE = [0,255,255]
PURPLE = [255,0,255]
WHITE = [255,255,255]
 
def draw_callback_color(pos, color, label, clabel):
    global GREEN, RED, YELLOW, BLUE
    l = clabel
    
    if l <= 0:
        return WHITE
    elif l == 1: #stackingbox
        return BLUE
    elif l == 2: #banana
        return YELLOW
    elif l == 3: #apple
        return GREEN
    elif l == 4: #orange
        return RED
    elif l == 5: #pear
        return LIGHTBLUE
    elif l == 6: #plum
        return PURPLE
    elif l == 7: #HAMMER
        print("HAMMA found how?")
        return [0,0,0]
    else:
        print("unknown label {}".format(l))
        return [0,0,0]
    
def find_data(path):
    files = os.listdir(path)
    data = []
    
    for file in files:
        filepath = path + "/" + file
        if (file.startswith("data") and file.endswith("ply")):
            data.append(filepath)
    return data
            
def convert_files():
    folders = os.listdir(current_dir)
    count_folders = 0
    for folder in folders:
        if not os.path.isdir(current_dir+ "/"+folder):
            continue
        if folder.startswith("synthetic"):
            count_folders+=1
        
    counter = 0
    for folder in folders:
        if not os.path.isdir(current_dir+ "/"+folder):
            continue
        path = current_dir + "/" + folder
        filepaths = find_data(path)
        
        for filepath in filepaths:
            if filepath is None or not os.path.exists(filepath):
                continue
            
            file = os.path.basename(filepath)
            _filename = os.path.splitext(file)[0]
            
            print(f"loading data from {filepath}", end = '')    
            
            try:
                splitted = folder.split("_")
                if "_a" in folder:
                    datetime = splitted[2] + "_" + splitted[3]
                elif len(splitted) == 4:
                    datetime = splitted[2] + "_" + splitted[3]
                elif len(splitted) > 4:
                    datetime = splitted[2] + "_" + splitted[3] + "-"  + splitted[5] + "-" + splitted[6]
                else:
                    datetime = splitted[2] + "_" + splitted[3]
            except:
                print("\ncouldn't extract datetime from {}".format(filepath))
                continue
                
            data = load_data(filepath)
            
            if data is None:
                print("No DATA found... remove folder {}".format(path))
                os.remove(filepath)
                continue
            
            if len(data['positions']) < 50000 and file == 'data.ply':
                os.rename(filepath, path + "/data_min.ply")
                file = "data_min.ply"
            
            data = convert_data(data,callback_filter_points_by_bbox)
            label_data = convert_data(data, callback_fix_label)
            
            #if parser.labeling_all:
            #    label_data = convert_data(label_data, callback_clabels)
                
            if 'min' not in file:
                down_sampled_data = down_sampling_data(label_data, parser.disable_info, parser.voxel_size, label_field='labels')
                save_data(path + "/down_sampled_"+_filename+ ".ply", down_sampled_data)
                save_data(path + "/trainings_"+_filename+"_" +datetime+".txt", down_sampled_data, header=False)
                info_data(down_sampled_data, field='labels', spaces=5)
            else:
                save_data(path + "/trainings_"+_filename+"_" +datetime+".txt", label_data, header=False)
                info_data(label_data, spaces=5)
                
            save_data(path +"/test_"+_filename+"_"+datetime+".txt", label_data, False)
            info_data(label_data, spaces=5)
        counter+=1
        print("{}% done".format(math.ceil(counter * 100.0 / count_folders)))   
    print(f"Converted {counter} files")
    
def callback_fix_label(pos, color, label, clabel):
    is_cl_zero = (int(clabel) == 0)
    is_l_zero = (int(label) == 0)
    
    if is_l_zero and is_cl_zero:
        return pos, color, 1, 1, False
    elif is_cl_zero and not is_l_zero:
        return pos, color, label, label, False
    elif not is_cl_zero and is_l_zero:
        return pos, color, clabel, clabel, False
    return pos, color, label, clabel, False
    
def convert_rgb(pos, c, label):
    s = struct.pack('>f', c)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value
			
    #r = (pack & 0x00FF0000) >> 16
    #g = (pack & 0x0000FF00) >> 8
    #b = (pack & 0x000000FF)
    
    r = (pack >> 16) & 0x0000ff
    g = (pack >> 8)  & 0x0000ff
    b = (pack)       & 0x0000ff
			
    if (r > 255 or g > 255 or b > 255):
        raise ValueError('Invalid packed RGB byte value')
    
    return pos, [r,g,b] , label

def rename_old_data():
    folders = os.listdir(current_dir)
    counter = 0
    for folder in folders:
        path = current_dir+ "/"+folder
        if not os.path.isdir(path):
            continue
        
        found_data = False
        files = os.listdir(path)
        
        for file in files:
            filepath = path + "/" + file
            
            if file.startswith(folder) and file.endswith(".ply"):
                os.rename(filepath, path + "/data.ply")
    print("renamed all")
    
def clean_files():
    folders = os.listdir(current_dir)
    counter = 0
    for folder in folders:
        path = current_dir+ "/"+folder
        if not os.path.isdir(path):
            continue
        
        found_data = False
        files = os.listdir(path)
        
        for file in files:
            filepath = path + "/" + file
            if not os.path.isfile(filepath):
                continue
            
            if (file.startswith("data") and file.endswith("ply")):
                found_data = True
                continue 
            
            if file == 'checkpoint.p':
                continue 
            
            if file.startswith('pre') or file.startswith('color') or file.endswith(".txt") or file.startswith("down"):
                os.remove(filepath)
                counter+=1
            
        if not found_data:
            shutil.rmtree(path)
            print(f"deleted path:{path} bc no data found there")  
        
    print(f"Cleaned {counter} files")
    
def get_testlist_index(list ,date):
    for i in range(len(list)):
        entry = list[i]
        if entry['date'] == date:
            return i
    return -1
    
def find_strptime_index(splitted, format = '%d-%m-%Y'):
    for i in range(len(splitted)):
        try:
            dt.strptime(splitted[i], format)
            return i
        except ValueError:
            continue
    return None
        
    
def gather_all_training_files(to_training_folder = 'training', to_test_folder = 'test'):
    path_to_folder = current_dir + "/" + to_training_folder
    
    if not os.path.isdir(path_to_folder):
        os.makedirs(path_to_folder)
    
    if not os.path.isdir(path_to_folder):
        os.mkdir(path_to_folder)
    
    folders = os.listdir(current_dir)
    test_files = []
    for folder in folders:
        path = current_dir+ "/" + folder
        if not os.path.isdir(path):
            continue
        
        files = os.listdir(path)
        for file in files:
            if file.startswith("trainings") and ('coloured' not in file) and file.endswith(".txt"):
                splitted = (file[0:len(file)-5]).split("_")
                
                date_index = find_strptime_index(splitted, '%d-%m-%Y')
                if date_index is None:
                    print("date_index is none, file: {}".format(file))
                    continue
                
                time_index = find_strptime_index(splitted, '%H-%M-%S')
                if time_index is None:
                    print("time is none, file: {}".format(file))
                    continue
                
                date = dt.strptime(splitted[date_index] , '%d-%m-%Y')
                datetime = dt.strptime(splitted[date_index] + " " + splitted[time_index], '%d-%m-%Y %H-%M-%S')
                index = get_testlist_index(test_files, date) 
                
                if index == -1:
                    test_files.append({'date': date, 'datetime':datetime, 'path': path, 'file': file})
                else:
                    if test_files[index]['datetime'] > datetime:
                        test_files[index]['datetime'] = datetime
                        test_files[index]['date'] = date
                        test_files[index]['path'] = path
                
                if not os.path.exists(path_to_folder + "/" + file):
                    shutil.copyfile(path + "/" + file, path_to_folder + "/" + file)
           
    
    for entry in test_files:
        os.remove(path_to_folder + "/" + entry['file'])
           
    path_to_folder = current_dir + "/" + to_test_folder       
    if not os.path.isdir(path_to_folder):
        os.makedirs(path_to_folder) 
    for entry in test_files:
        path = entry['path']
        files = os.listdir(path)
        
        for file in files:
            if file.startswith("test") and ('coloured' not in file) and file.endswith(".txt"):
                if not os.path.exists(path_to_folder + "/" + file):
                    shutil.copyfile(path + "/" + file, path_to_folder + "/" + file)
        
    print("Gathered all files for training")
    
def info_data(data, field = 'clabels', spaces = 0):
    str_spaces = ' ' * spaces
    name_maping = ['Nothing', 'Stackingbox', 'Banana', 'Apple', 'Orange', 'Pear', 'Plum','Hammer']
    
    if field in data:
        labels = data[field]
    else:
        if 'labels' in data:
            labels = data['labels']
        elif 'clabels' in data:
            labels = data['clabels']
        else:
            print("info_data: No labels found?")
            return
    unique_labels = set(labels)
    counter_label_map = [0] * len(name_maping)
    
    for i in range(len(labels)):
        label = labels[i]
        counter_label_map[label] = counter_label_map[label]+1
        
    for i in range(len(name_maping)):
        print("{}{} found {} labels".format(str_spaces,name_maping[i], counter_label_map[i]))
    
def create_ply_header(volume, format = 'x,y,z,red,green,blue'):
    header = ''
    
    header += 'ply\n'
    header += 'format ascii 1.0\n'
    header += 'element vertex %d\n' % volume
    
    split = format.split(',')
    
    for i in range(len(split)):
        type = ''
        name = split[i]
        
        if name == 'x' or name == 'y' or name == 'z':
            type = 'float'
        elif name == 'r':
            type = 'uchar'
            name = 'red'
        elif name == 'g':
            type = 'uchar'
            name = 'green'
        elif name == 'b':
            type = 'uchar'
            name = 'blue'
        else:
            type = 'uchar'
    
        header += 'property '+type+' '+name+'\n'
    header += 'end_header\n'
    return header
    
    
if __name__ == "__main__":
    parser = parse_arguments()
    current_dir = osp.join(os.curdir, '..', 'data', 'ply')
    
    if parser.conv_ply:
        print("path: {}".format(parser.path))
        print("format: {}".format(parser.format))
        data = load_data(parser.path, str_fields=parser.format)
        save_data(os.path.splitext(parser.path)[0] + ".ply" , data, header = True, str_fields = parser.format)
    elif parser.conv_xyz:
        print("path: {}".format(parser.path))
        print("rgb: {}".format(parser.rgb))
        rgb = parser.rgb.split(',')
        data = load_data(parser.path, str_fields="x,y,z")
        vol = len(data['positions'])
        print("vol: {}".format(vol))
        data['colors'] = [[rgb[0],rgb[1],rgb[2]]]  * vol
        data['labels'] = [rgb[3]] * vol
        #data['clabels'] = [rgb[3]] * vol
        save_data(os.path.splitext(parser.path)[0] + ".txt" , data, header = False, str_fields = 'x,y,z,r,g,b,label')
    elif parser.renamed_old_data:
        rename_old_data()
    elif parser.convert_obj is not None:
        if not PYMESH:
            print("Please install pymesh")
            exit()
        pc = farthest_point_sampling(parser.convert_obj)
        print("PC: {}".format(pc))
    elif parser.gather:
        gather_all_training_files()
    elif parser.clean:
        clean_files()
    elif parser.info:
        data = load_data(parser.path, str_fields=parser.str_fields)
        
        info_data(data)
    elif parser.labeling:
        data = load_data(parser.path, str_fields=parser.str_fields)
        labeled_data = convert_data(data,callback_clabels)
        info_data(labeled_data)
        save_data(parser.path+"_labeled", labeled_data, header=False)
    elif OPEN3D and parser.vis is not None:
        filepath = parser.vis
        
        if not os.path.isfile(filepath):
            print(f"file {filepath} not found!")
            exit()
        
        
        str_format = 'x,y,z,r,g,b,label'
        
        if parser.with_pred_label:
            str_format = str_format + ',pred_label'
        
        data = load_data(filepath, str_format)
        
        if not parser.disable_info:
            print(f"found {len(data['positions'])} points")
        
        if parser.vis_coloured:
            draw_data(data, draw_callback_color)
        else:
            draw_data(data)
            
    elif OPEN3D:
        convert_files()
    else:
        print("Please install open3d")
    