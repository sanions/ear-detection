from json import load
import cv2
import os
from math import dist

from utils import crop_to_ear, load_X, find_landmarks, calculate_size_ratio

INPUT_DIR = './images/input/'
img = cv2.imread('./images/input/saniya-ear-dot-1.jpg')

def main(input_img='none', input_dir=INPUT_DIR):
    if (input_img != 'none'):
        img = input_img
        ratio = calculate_size_ratio(img)
        img = crop_to_ear(img)
        X = load_X(img)
        ot, tr1, tr2 = find_landmarks(X, img)

        d1 = dist(ot, tr1) * ratio
        d2 = dist(ot, tr2) * ratio
        
        return d1, d2
        
    for file in os.scandir(input_dir):
        print(file.name)
        img = cv2.imread(file.path)
        ratio = calculate_size_ratio(img)
        img = crop_to_ear(img)
        X = load_X(img)
        ot, tr1, tr2 = find_landmarks(X, img)

        d1 = dist(ot, tr1) * ratio
        d2 = dist(ot, tr2) * ratio

        print(f'approx. distances: {d1} inches, {d2} inches' )

if __name__ == '__main__':
    main(input_img=img)