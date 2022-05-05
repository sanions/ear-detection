import cv2
import os
import sys
from math import dist

from utils import crop_to_ear, load_X, find_landmarks, calculate_size_ratio

INPUT_DIR = './images/input/'

def main(img):
    ratio = calculate_size_ratio(img)
    img = crop_to_ear(img) 
    X = load_X(img) 
    ot, tr1, tr2 = find_landmarks(X, img)
    # 3 landmarks are found because tr1 and tr2 represent the ear canal -- but they're imprecise.
    # it's best to look at the result (`img_landmarks.jpg`) and decide which distance is better.

    d1 = dist(ot, tr1) * ratio
    d2 = dist(ot, tr2) * ratio

    print(f'Distance: about {d1} inches or {d2} inches' )

    return d1, d2

if __name__ == '__main__':
    args = sys.argv
    filename = args[1]
    img = cv2.imread('./images/input/' + filename)
    main(img)