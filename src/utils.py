import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import matplotlib.pyplot as plt

def calculate_size_ratio(img):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 75, 20])
    upper = np.array([60, 255, 255])
    mask = cv2.inRange(imghsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im = np.copy(img)
    cv2.drawContours(im, contours, -1, (0, 0, 255), 5)

    valid_count = 0
    for contour in contours:
        if len(contour) < 200: 
            continue

        minX = img.shape[1]
        minY = img.shape[0]
        maxX = 0
        maxY = 0

        for coords in contour:
            coord = coords[0]
            x = coord[0]
            y = coord[1]

            if x < minX:
                minX = x
            if x > maxX:
                maxX = x
            if y < minY:
                minY = y
            if y > maxY:
                maxY = y

        h = maxY - minY 
        w = maxX - minX

        if abs(1 - h/w) <= 0.1: 
            valid_count += 1
            cv2.rectangle(im, (minX, minY), (maxX, maxY), (255, 0, 255), 10)

    cv2.imwrite("./images/progress/reference_object.jpg", im)

    if valid_count > 1:
        print('too many reference objects found.')
        return
    if valid_count == 0:
        print('reference object not found.')
        return

    inch_per_pixel = 1/(maxX - minX)
    return inch_per_pixel

def crop_to_ear(img):
    cascades = ['other-cascade.xml', 'ear-cascade.xml', 'last-cascade.xml']

    for casc in cascades:
        ear_cascade = cv2.CascadeClassifier('./pretrained-models/haar-cascades/' + casc)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dims = gray_img.shape
        ears = ear_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(int(dims[1]/5), int(dims[0]/3)))

        if len(ears) > 0:
            break

    for (x, y, w, h) in ears:
        r = 8
        dim = img.shape
        bot_lef_x = int(x - w/r) if (x - w/r >= 0) else 0
        bot_lef_y = int(y - 10) if (y - 10 >=0) else 0
        top_lef_x = int(x + w + w/r) if (x + w + w/r <= dim[0]) else dim[0]
        top_lef_y = int(y + h + 10) if (y + h + 10 <= dim[1]) else dim[1]

        cropped = img[bot_lef_y: top_lef_y, bot_lef_x: top_lef_x]

        cv2.rectangle(img, (bot_lef_x, bot_lef_y), (top_lef_x, top_lef_y), (255, 0, 0), 2)
        break

    cv2.imwrite('./images/progress/detected_ears.jpg', img)
    cv2.imwrite('./images/progress/cropped_img.jpg', cropped)

    return cropped

def load_X(img):
    img = cv2.resize(img, (224, 224)) # first resize the original single image
    cv2.imwrite('images/progress/resized_img.png', img)       # save the resized original single image
    img = image.load_img('images/progress/resized_img.png')      # load the resized original single image
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def find_landmarks(X, img):
    model = load_model('./pretrained-models/cnn-models/cnn-model-2.h5')

    x = X[None, :]
    prediction = model.predict(x)
    pred = prediction[0]

    dims = img.shape

    img_original = plt.imread('./images/progress/cropped_img.jpg')

    coords = []

    for p in range(len(pred)):
        if p < 55:
            pred[p] = int(pred[p] * dims[1])
            pred[p+55] = int(pred[p+55] * dims[0])
        if p in [0, 35, 36]:
            print(str(pred[p]) + ' ' + str(pred[p+55]))
            coords.append((pred[p], pred[p+55]))
            plt.scatter([pred[p]], [pred[p+55]])

    plt.imshow(img_original)
    plt.savefig('./images/result/img_landmarks.jpg')
    plt.close()

    return coords[0], coords[1], coords[2]







