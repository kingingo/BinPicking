import numpy as np
import math

def transform_point(point, translation, rotation):
    # Apply translation
    translated_point = point + translation
    
    # Apply rotation
    rotated_point = np.dot(rotation, translated_point)
    
    return rotated_point

def inverse_matrix(matrix):
    return np.transpose(matrix)

def calculate_translation_vector(pfrom, pto):
    translation_vector = pto - pfrom
    return translation_vector

def create_rotation_matrix(angle):
    # Convert angles to radians
    roll_rad = np.radians(angle[0])
    pitch_rad = np.radians(angle[1])
    yaw_rad = np.radians(angle[2])
    
    # Create rotation matrices for each axis
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(roll_rad), -np.sin(roll_rad)],
                           [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    rotation_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                           [0, 1, 0],
                           [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    rotation_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                           [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                           [0, 0, 1]])
    
    # Combine rotation matrices to get the final rotation matrix
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    return rotation_matrix

world = np.array([0,0,0])  
world_angle = np.array([0, 0, 0])

camera = np.array([0,0,1.9])
#roll pitch yaw
camera_angle = np.array([0,90,0])

correction_angle = np.array([270,0,270])

point = np.array([-0.454140633345, -0.329147785902, 1.53950810432]) # camera point of view so 0,0,1.9 is 0,0,0 for point

translation = calculate_translation_vector(camera, world) # from camera to world transformation vector

correction_rotation = create_rotation_matrix(correction_angle) #https://answers.gazebosim.org//question/4266/gazebo-15-libgazebo_openni_kinectso-pointcloud-data-is-rotated-and-flipped/
cam_rotation = create_rotation_matrix(camera_angle)  # Camera Rotation
world_rotation = create_rotation_matrix(world_angle) # World Rotation

rotation = np.dot(np.dot(cam_rotation, world_rotation), correction_rotation)
rotation = inverse_matrix(rotation)

# Perform the transformation
transformed_point = transform_point(point, translation, rotation)

print("Point: {}".format(point))
print("Transformed Point: {}".format(transformed_point))
#0.329147785902 0.454140633345 0.360491895676 transformation from gazebo static transformator
x_correct = math.isclose(transformed_point[0], 0.329147785902)
y_correct = math.isclose(transformed_point[1], 0.454140633345)
z_correct = math.isclose(transformed_point[2], 0.360491895676)

if x_correct and y_correct and z_correct:
    print("CORRECT TRANSFORMATION!")
else:
    print("X Correct: {}".format(x_correct))
    print("Y Correct: {}".format(y_correct))
    print("Z Correct: {}".format(z_correct))
    
