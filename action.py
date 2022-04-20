from PIL import Image, ImageEnhance
import numpy as np

import time

kelvin_table = [(255, 219, 186), # 4500
                (255, 228, 206), # 5000
                (255, 236, 224), # 5500
                (255, 243, 239), # 6000
                (255, 249, 253)] # 6500
    



def take_action(image_np, action_idx):
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
    image_pil = Image.fromarray(image_np)
    shape = image_np.shape
    i = shape[0]
    j = shape[1]
    temp = np.zeros(shape=(data.shape))
    # enhance contrast
    if action_idx == 0:
        enh = ImageEnhance.Contrast(image_pil)
        image_enh = enh.enhance(0.9)
    elif action_idx == 1:
        enh = ImageEnhance.Contrast(image_pil)
        image_enh = enh.enhance(1.1)
    # enhance color
    elif action_idx == 2:
        enh = ImageEnhance.Color(image_pil)
        image_enh = enh.enhance(0.9)
    elif action_idx == 3:
        enh = ImageEnhance.Color(image_pil)
        image_enh = enh.enhance(1.1)
    # color brightness
    elif action_idx == 4:
        enh = ImageEnhance.Brightness(image_pil)
        image_enh = enh.enhance(0.7)
    elif action_idx == 5:
        enh = ImageEnhance.Brightness(image_pil)
        image_enh = enh.enhance(1.3)
    # color temperature : http://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop
    elif action_idx == 6:
        r, g, b = kelvin_table[0]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)
    elif action_idx == 7:
        r, g, b = kelvin_table[1]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)
    elif action_idx == 8:
        r, g, b = kelvin_table[2]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)
    elif action_idx == 9:
        r, g, b = kelvin_table[3]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)
    elif action_idx == 10:
        r, g, b = kelvin_table[4]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        image_enh = image_pil.convert('RGB', matrix)
    # Red
    elif action_idx == 11:
        for a in range(i):
            for b in range(j):
                image_np[a][b][0] = image_np[a][b][0] * 1.05
        image_enh = Image.fromarray(image_np)
    elif action_idx == 12:
        for a in range(i):
            for b in range(j):
                image_np[a][b][0] = image_np[a][b][0] * 0.95
        image_enh = Image.fromarray(image_np)
    # Green
    elif action_idx == 13:
        for a in range(i):
            for b in range(j):
                image_np[a][b][1] = image_np[a][b][1] * 1.05
        image_enh = Image.fromarray(image_np)
    elif action_idx == 14:
        for a in range(i):
            for b in range(j):
                image_np[a][b][1] = image_np[a][b][1] * 0.95
        image_enh = Image.fromarray(image_np.copy())
    # Blue
    elif action_idx == 15:
        for a in range(i):
            for b in range(j):
                image_np[a][b][2] = image_np[a][b][2] * 1.05
        image_enh = Image.fromarray(image_np)
    elif action_idx == 14:
        for a in range(i):
            for b in range(j):
                image_np[a][b][2] = image_np[a][b][2] * 0.95
        image_enh = Image.fromarray(image_np)
    else:
        print("error")

    # random_id = str(random.randrange(100000))
    # image_pil.save("%s_%d_raw.jpg" % (random_id, action_idx))
    # image_enh.save("%s_%d_enh.jpg" % (random_id, action_idx))
    # return np.asarray(image_enh)
    return (np.array(image_enh))


image1 = Image.open('sample-image/example image 2/20220211-000034050008.jpg')

data = np.array(image1)

image = Image.fromarray(take_action(data, 15))

image1.show()
image.show()
