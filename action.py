"""
JongChan Park, DISTORT-AND-RECOVER-CVPR18, (2018), GitHub repository, https://github.com/Jongchan/DISTORT-AND-RECOVER-CVPR18
Modified by: Krishna Kothandaraman, Haobao Li
"""

import sys

import colorsys

import cv2
from PIL import Image, ImageEnhance
import numpy as np
import enum

kelvin_table = [(255, 219, 186),  # 4500
                (255, 228, 206),  # 5000
                (255, 236, 224),  # 5500
                (255, 243, 239),  # 6000
                (255, 249, 253)]  # 6500


class ActionType(enum.Enum):
    """Enum of action types"""
    lower_contrast = enum.auto()
    higher_contrast = enum.auto()
    lower_saturation = enum.auto()
    higher_saturation = enum.auto()
    lower_brightness = enum.auto()
    higher_brightness = enum.auto()
    warmer = enum.auto()
    bit_warmer = enum.auto()
    normal = enum.auto()
    bit_cooler = enum.auto()
    cooler = enum.auto()
    more_red = enum.auto()
    less_red = enum.auto()
    more_green = enum.auto()
    less_green = enum.auto()
    more_blue = enum.auto()
    less_blue = enum.auto()
    shift_hue_up = enum.auto()
    shift_hue_down = enum.auto()


# TODO: FIND OUT WHY COLOUR CHANGES
def take_action(image_np, action_type: ActionType) -> np.array:
    # image_pil = Image.fromarray(np.uint8(image_np))
    """ Take in a np.array of a image, adjust based on action index"""
    """
    0 lower contrast
    1 higher contrast
    2 lower saturation
    3 higher saturation
    4 lower brightness
    5 higher brightness
    6 warmer
    7 bit warmer
    8 normal
    9 bit cooler
    10 cooler
    11 more red
    12 less red
    13 more green
    14 less green
    15 more blue
    16 less blue
    """
    image_pil = Image.fromarray(image_np, "RGB")
    shape = image_np.shape
    i = shape[0]
    j = shape[1]
    # enhance contrast
    if action_type == ActionType.lower_contrast:
        enh = ImageEnhance.Contrast(image_pil)
        image_enh = enh.enhance(0.9)

    elif action_type == ActionType.higher_contrast:
        enh = ImageEnhance.Contrast(image_pil)
        image_enh = enh.enhance(1.1)

    # enhance color
    elif action_type == ActionType.lower_saturation:
        enh = ImageEnhance.Color(image_pil)
        image_enh = enh.enhance(0.9)

    elif action_type == ActionType.higher_saturation:
        enh = ImageEnhance.Color(image_pil)
        image_enh = enh.enhance(1.1)

    # color brightness
    elif action_type == ActionType.lower_brightness:
        enh = ImageEnhance.Brightness(image_pil)
        image_enh = enh.enhance(0.9)

    elif action_type == ActionType.higher_brightness:
        enh = ImageEnhance.Brightness(image_pil)
        image_enh = enh.enhance(1.1)

    # color temperature : http://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image
    # -like-in-photoshop
    elif action_type == ActionType.warmer:
        r, g, b = kelvin_table[0]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)

    elif action_type == ActionType.bit_warmer:
        r, g, b = kelvin_table[1]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)

    elif action_type == ActionType.normal:
        r, g, b = kelvin_table[2]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)

    elif action_type == ActionType.bit_cooler:
        r, g, b = kelvin_table[3]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)

    elif action_type == ActionType.cooler:
        r, g, b = kelvin_table[4]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)

    # Red
    elif action_type == ActionType.more_red:
        for a in range(i):
            for b in range(j):
                temp = image_np[a][b][0]

                image_np[a][b][0] = image_np[a][b][0] * 1.05
        image_enh = Image.fromarray(image_np)

    elif action_type == ActionType.less_red:
        for a in range(i):
            for b in range(j):
                image_np[a][b][0] = image_np[a][b][0] * 0.95
        image_enh = Image.fromarray(image_np)
    # Green
    elif action_type == ActionType.more_green:
        for a in range(i):
            for b in range(j):
                image_np[a][b][1] = image_np[a][b][1] * 1.05
        image_enh = Image.fromarray(image_np)

    elif action_type == ActionType.less_green:
        for a in range(i):
            for b in range(j):
                image_np[a][b][1] = image_np[a][b][1] * 0.95
        image_enh = Image.fromarray(image_np.copy())

    # Blue
    elif action_type == ActionType.more_blue:
        for a in range(i):
            for b in range(j):
                image_np[a][b][2] = image_np[a][b][2] * 1.05
        image_enh = Image.fromarray(image_np)

    elif action_type == action_type.less_blue:
        for a in range(i):
            for b in range(j):
                image_np[a][b][2] = image_np[a][b][2] * .95

        image_enh = Image.fromarray(image_np)

    elif action_type == action_type.shift_hue_up:
        for a in range(i):
            for b in range(j):
                temp = colorsys.rgb_to_hsv(image_np[a][b])
                if temp[0] >= 355:
                    temp[0] += 5
                else:
                    temp[0] = 360

                image_np[a][b] = colorsys.hsv_to_rgb(temp)
        image_enh = Image.fromarray(image_np)

    elif action_type == action_type.shift_hue_down:
        for a in range(i):
            for b in range(j):
                temp = colorsys.rgb_to_hsv(image_np[a][b])
                if temp[0] <= 5:
                    temp[0] -= 5
                else:
                    temp[0] = 0

                image_np[a][b] = colorsys.hsv_to_rgb(temp)
        image_enh = Image.fromarray(image_np)

    else:
        print("Invalid Action")
        sys.exit(1)

    # random_id = str(random.randrange(100000))
    # image_pil.save("%s_%d_raw.jpg" % (random_id, action_type))
    # image_enh.save("%s_%d_enh.jpg" % (random_id, action_type))
    # return np.asarray(image_enh)
    return np.array(image_enh)
