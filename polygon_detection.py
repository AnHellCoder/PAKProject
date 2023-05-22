import torch
import torchvision as tv
import numpy as np
import pandas as pd
from PIL import Image
import cv2

def reconstruct(img: np.ndarray, mask: np.ndarray, tup: tuple):
    #Ищем координаты нужных значений для каждого из каналов
    #Результат - два тензора-вектора с координатами
    r = np.where(mask[0] == tup[0])
    g = np.where(mask[1] == tup[1])
    b = np.where(mask[2] == tup[2])

    #объединяем x и y координаты в пары
    r_coordinates = list(zip(r[0].tolist(), r[1].tolist()))
    g_coordinates = list(zip(g[0].tolist(), g[1].tolist()))
    b_coordinates = list(zip(b[0].tolist(), b[1].tolist()))

    x = []
    y = []

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

    #Итоговый список закидываем в __getitem__ уже изображения - он выдаёт тензор с необходиыми значениями.
    return img[:,x,y]

def script(img: np.ndarray, mask: np.ndarray):
    typ = {'chart_title': (0,1,0),
    'axis_title': (0,2,0),
    'plot-bb': (0,4,0),
    'axes': (0,5,0),
    "bars": (0,6,0),
    "boxplots": (0,7,0),
    "dot points": (0,8,0),
    "lines": (0,9,0),
    "scatter points": (0,10,0)} #Цвет каждого полигона

    #Основной скрипт для восстановки полигонов
    ct = reconstruct(img, mask, typ["chart_title"])
    at = reconstruct(img, mask, typ["axis_title"])
    pb = reconstruct(img, mask, typ["plot-bb"])
    axes = reconstruct(img, mask, typ["axes"])
    bars = reconstruct(img, mask, typ['bars'])
    bp = reconstruct(img, mask, typ["boxplots"])
    dp = reconstruct(img, mask, typ["dot points"])
    lines = reconstruct(img, mask, typ["lines"])
    sp = reconstruct(mask, typ['scatter points'])

# trans = tv.transforms.ToTensor()
# img = Image.open('0000ae6cbdb1.jpg')
# mask = Image.open('mask_tc.jpg')

# img = trans(img)
# mask = trans(mask)

# print(mask.shape)

img = cv2.imread('0000ae6cbdb1.jpg')
mask = cv2.imread('mask_tc.jpg')

print(mask)
print(np.unique(mask))

script(img, mask)

############################Как эта штука работает на примере тензора случайных чисел в диапазоне от 6 до 7
np.random.seed(255)

ten = torch.randint(6, 8, (3, 10, 10))

print(ten)

r = torch.where(ten[0] == 6)
g = torch.where(ten[1] == 7)
b = torch.where(ten[2] == 6)

r = list(zip(r[0].tolist(), r[1].tolist()))
g = list(zip(g[0].tolist(), g[1].tolist()))
b = list(zip(b[0].tolist(), b[1].tolist()))
print(r)
print(g)
print(b)

length = min(len(r), len(g), len(b))

x, y = [], []

for i in range(length):
    if len(r) < len(g):
        if r[i] in g and r[i] in b:
            x.append(r[i][0])
            y.append(r[i][1])
    elif len(g) < len(b):
        if g[i] in r and g[i] in b:
            x.append(g[i][0])
            y.append(g[i][1])
    else:
        if b[i] in r and b[i] in g:
            x.append(b[i][0])
            x.append(b[i][1])

print(x)
print(y)

print(ten[: ,x, y])
#############################