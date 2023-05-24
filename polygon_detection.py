import torch
import torchvision as tv
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import segmentator
import json

def reconstruct(img: np.ndarray, mask: np.ndarray, tup: tuple):
    h = np.where(mask[:,:,] == tup)

    x_begin, x_end = h[0].min(), h[0].max()
    y_begin, y_end = h[1].min(), h[1].max()

    print(x_begin, x_end)
    print(y_begin, y_end)

    return img[x_begin:x_end, y_begin:y_end]

    r = np.where(mask[:, :, 0] == tup[0])
    g = np.where(mask[:, :, 1] == tup[1])
    b = np.where(mask[:, :, 2] == tup[2])

    r_coordinates = list(zip(r[0].tolist(), r[1].tolist()))
    g_coordinates = list(zip(g[0].tolist(), g[1].tolist()))
    b_coordinates = list(zip(b[0].tolist(), b[1].tolist()))

    x, y = [], []

    #Ищем значение кратчайшего списка координат
    length = min(len(r_coordinates), len(g_coordinates), len(b_coordinates))

    for i in range(length):
        #Ориентируемся на кратчайший список. По его координатам ищем те же координаты в остальных списках
        if len(r_coordinates) < len(g_coordinates):
            if r_coordinates[i] in g_coordinates and r_coordinates[i] in b_coordinates:
                x.append(r_coordinates[i][0])
                y.append(r_coordinates[i][1])
        elif len(g_coordinates) < len(b_coordinates):
            if g_coordinates[i] in r_coordinates and g_coordinates[i] in b_coordinates:
                x.append(g_coordinates[i][0])
                y.append(g_coordinates[i][1])
        else:
            if b_coordinates[i] in r_coordinates and b_coordinates[i] in g_coordinates:
                x.append(b_coordinates[i][0])
                x.append(b_coordinates[i][1])
    
    return img[x, y,]

def script(img: np.ndarray, mask: np.ndarray):
    typ = {'chart_title': (1,1,1),
    'axis_title': (2,2,2),
    'tick_label': (3,3,3),
    'plot-bb': (4,4,4),
    'axes': (5,5,5),
    "bars": (6,6,6),
    "boxplots": (7,7,7),
    "dot points": (8,8,8),
    "lines": (9,9,9),
    "scatter points": (10,10,10)} #Цвет каждого полигона

    #Основной скрипт для восстановки полигонов
    ct = reconstruct(img, mask, (1,1,1))#typ["chart_title"])
    cv2.imwrite('./results/chart_title.jpg', ct)
    at = reconstruct(img, mask, typ["axis_title"])
    cv2.imwrite('./results/axis_title.jpg', at)
    pb = reconstruct(img, mask, typ["plot-bb"])
    cv2.imwrite('./results/plot_bb.jpg', pb)
    axes = reconstruct(img, mask, typ["axes"])
    cv2.imwrite('./results/axes.jpg', axes)
    bars = reconstruct(img, mask, typ['bars'])
    cv2.imwrite('./results/bars.jpg', bars)
    try:
        bp = reconstruct(img, mask, typ["boxplots"])
        cv2.imwrite('./results/boxplots.jpg', bp)
    except BaseException:
        pass
    try:
        dp = reconstruct(img, mask, typ["dot points"])
        cv2.imwrite('./results/dot_points.jpg', dp)
    except BaseException:
        pass
    try:
        lines = reconstruct(img, mask, typ["lines"])
        cv2.imwrite('./results/lines.jpg', lines)
    except BaseException:
        pass
    try:
        sp = reconstruct(mask, typ['scatter points'])
        cv2.imwrite('./results/scatter_points.jpg', sp)
    except BaseException:
        pass

# trans = tv.transforms.ToTensor()
# img = Image.open('0000ae6cbdb1.jpg')
# mask = Image.open('mask_tc.jpg')

# img = trans(img)
# mask = trans(mask)

# print(mask.shape)

arr = np.array([0, 1, 0] * (7 * 12))
arr = np.reshape(arr, (7, 12, 3))
print(arr)

coords = np.where(arr[:,:,] == (0, 1, 0))

print(coords[0].min(), coords[0].max())
print(coords[1].min(), coords[1].max())

##7, 277
##12, 463
img = cv2.imread('0000ae6cbdb1.jpg')
mask = cv2.imread('mask_tc.jpg')
shape = np.shape(img)[:2]

mask = segmentator.interesting('0000ae6cbdb1.json', shape)
print(np.shape(mask))
print(np.shape(img))

script(img, mask)