import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """

    coord1, coord2 = cd_color_segmentation(img)

    x1, y1 = coord1
    x2, y2 = coord2

    bounding_img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imshow("image", bounding_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cd_color_segmentation(img, template=None):
    """
    Implement the cone detection using color segmentation algorithm
    Input:
        img: np.3darray; the input image with a cone to be detected. BGR.
        template_file_path; Not required, but can optionally be used to automate setting hue filter values.
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
                (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
    """
    ########## YOUR CODE STARTS HERE ##########

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    boundaries_high = ([5, 100, 100],[17, 255, 255])

    upper = np.array(boundaries_high[1], dtype='uint8')
    lower = np.array(boundaries_high[0], dtype='uint8')

    mask = cv2.inRange(img_hsv, lower, upper)

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    bounding_box = ((0,0),(0,0))

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x1,y1,xdelta,ydelta = cv2.boundingRect(largest_contour)
        bounding_box = ((x1,y1), (x1+xdelta, y1+ydelta))

    # Return bounding box
    # return bounding_box

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()

        image = cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #img = cv2.imread('test1.jpg')

    #image_print(img)
