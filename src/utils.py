import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import matplotlib.pyplot as plt

def calculate_size_ratio(img):
    '''Finds the reference object and measures the width/height in 
    pixels to calculate an inch (or any other metric) per pixel ratio. '''
    
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    # lower and upper ranges in HSV colors for the color of the reference object -- currently, GREEN
    # TODO: to find a different color, use https://i.stack.imgur.com/gyuw4.png and instruction here: https://stackoverflow.com/a/48367205
    lower = np.array([30, 75, 20])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(imghsv, lower, upper)

    # find contours/shapes with colors
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im = np.copy(img)

    # draw contours for visualization
    cv2.drawContours(im, contours, -1, (0, 0, 255), 5)

    valid_count = 0    # counter to track number of valid contours found --- should only be 1
    for contour in contours:
        # make sure the number of points that mark this contour/shape is large enough -- filters out random spots of color found in image
        if len(contour) < 800: 
            continue    

        print(len(contour))
        # find the leftmost, rightmost, topmost, bottommost points to use for calculating measurements of the reference object
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

        # calculate height and width
        h = maxY - minY 
        w = maxX - minX

        # since reference object is a circle, the height/width ratio should be close-ish to 1 
        if abs(1 - h/w) <= 0.2: 
            valid_count += 1
            cv2.rectangle(im, (minX, minY), (maxX, maxY), (255, 0, 255), 10) # draw a rectangle around reference object 

    cv2.imwrite("./images/progress/reference_object.jpg", im)

    # errors if the reference object was not properly found
    if valid_count > 1:
        print('too many reference objects found.')
        return
    if valid_count == 0:
        print('reference object not found.')
        return


    # calculate ratio -- TODO: change metric to represent actual size of reference object
    metric = 1 # inch
    inch_per_pixel = metric/h
    return inch_per_pixel

def crop_to_ear(img):
    '''Uses haar cascades to find the ear in the image 
    and crops the image to just the ear.'''

    # list of available existing cascades 
    cascades = ['other-cascade.xml', 'ear-cascade.xml', 'last-cascade.xml']

    # sometimes, a cascade won't find an ear, but another cascade will, so keep on trying cascades until one of them outputs an ear
    for casc in cascades:
        ear_cascade = cv2.CascadeClassifier('./pretrained-models/haar-cascades/' + casc)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dims = gray_img.shape
        # the `minSize` argument below indicates that the ear should be at least 1/5 of the width of full image and 1/3 of the height 
        ears = ear_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(int(dims[1]/5), int(dims[0]/2)))
        if len(ears) > 0:
            break

    for (x, y, w, h) in ears:
        # set box to a little bit larger than identified ear to make sure it covers the whole ear
        r = 8
        dim = img.shape
        bot_lef_x = int(x - w/r) if (x - w/r >= 0) else 0
        bot_lef_y = int(y - 10) if (y - 10 >=0) else 0
        top_lef_x = int(x + w + w/r) if (x + w + w/r <= dim[0]) else dim[0]
        top_lef_y = int(y + h + 10) if (y + h + 10 <= dim[1]) else dim[1]

        cv2.rectangle(img, (bot_lef_x, bot_lef_y), (top_lef_x, top_lef_y), (255, 0, 0), 2) # draw rectangle on original image

        cropped = img[bot_lef_y: top_lef_y, bot_lef_x: top_lef_x] # crop image 
        break

    cv2.imwrite('./images/progress/detected_ears.jpg', img)
    cv2.imwrite('./images/progress/cropped_img.jpg', cropped)

    return cropped

def load_X(img):
    '''Prepares image data for finding the landmarks. '''
    img = cv2.resize(img, (224, 224)) # first resize the original single image
    cv2.imwrite('images/progress/resized_img.png', img) # save the resized original single image
    img = image.load_img('images/progress/resized_img.png') # load the resized original single image
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x

def find_landmarks(X, img):
    '''Use a CNN Model to find the landmarks. The model will find 
    55 landmarks over the entire the ear, but we're only interested 
    in landmarks #0 (the top of the ear -- otobasion) and 
    #35, 36 (the top and middle of the ear canal -- tragus 1 and 2). '''

    # TODO: if you've trained a new model, change the name here to ensure you're using the running the right model.
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







