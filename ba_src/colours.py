import struct
import ctypes

def rgb_to_uint8(color):
    r, g, b = color[0], color[1], color[2]

    pack = (r << 16) | (g << 8) | b

    i = ctypes.c_uint32(pack).value
    s = struct.pack('>l', i)

    float_rgb = struct.unpack('>f', s)[0]

    return float_rgb

def uint8_to_rgb(float_rgb):
    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value
			
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
			
    color = [r,g,b]
			
    return pack, color

color = [128, 64, 255]
eight_bit_color = rgb_to_uint8(color)
color_value,conv_back_color = uint8_to_rgb(eight_bit_color)

def binary(x, n = 32):
    return format(x, 'b').zfill(n)
    #return format(num, '#0{}b'.format(length + 2))

def print_color_operations(pack):
    print_color_operation('RED',pack, 0x00FF0000, 16)
    print_color_operation('GREEN',pack, 0x0000FF00, 8)
    print_color_operation('BLUE',pack, 0x000000FF, 0)

def print_color_operation(name, pack, hex, shift):
    v = (pack & hex)
    vs = v >> shift
    
    print(f"{name}:")
    print(f"  {binary(pack)}")
    print(f"& {binary(hex)}")
    print("----------------------")
    print(f"  {binary(v)}")
    print(f">> {shift}")
    print("----------------------")
    print(f"  {binary(vs)}")
    print(f"= {vs}")
    print("")


print_color_operations(color_value)

print("8-bit color {}".format(eight_bit_color))
print("RGB color {}".format(conv_back_color))

'''

COLOR:
                                     
& 0b 0 0 0 0 0 0 0 0 R  R  R  R  R  R  R  R  G  G  G  G  G  G  G  G  B  B  B  B  B  B  B  B

Example: RGB: 128, 64, 255
   
RED:
  0b 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1
& 0b 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
----------------------
  0b 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
>> 16
----------------------
  0b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
= 128

GREEN:
  0b 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1
& 0b 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
----------------------
  0b 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
>> 8
----------------------
  0b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
= 64

BLUE:
  0b 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1
& 0b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
----------------------
  0b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
>> 0
----------------------
  0b 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
= 255
   

'''