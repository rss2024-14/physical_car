import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel
from std_msgs.msg import String

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation

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

class TFDetector(Node):
    def __init__(self):
        super().__init__("tf_detector")

        #subscribe to ZED camera
        self.tf_pub = self.create_publisher(String, "/traffic_light", 10)
        self.debug_pub = self.create_publisher(Image, "/tf_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.tf_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("TF Detector Initialized")

    def convert_to_string_and_publish(self, data):
        msg = String()
        msg.data = data
        self.tf_pub.publish(msg)

    def tf_callback(self, img_msg):
        # Check img type
        img = cv2.imread('tfl6.jpg')
        # img = cv2.resize(img, (0,0), fx=0.2, fy=0.2) 

        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        
        tf_bbox = self.tf_detector(img)
        tf_coord1, tf_coord2 = tf_bbox
        tf_x1,tf_y1 = tf_coord1
        tf_x2,tf_y2 = tf_coord2
        
        green_bbox = self.green_detector(img)
        g_coord1, g_coord2 = green_bbox
        g_x1,g_y1 = g_coord1
        g_x2,g_y2 = g_coord2
        
        red_bbox = self.red_detector(img)
        r_coord1, r_coord2 = red_bbox
        r_x1,r_y1 = r_coord1
        r_x2,r_y2 = r_coord2
        cv2.rectangle(img, (tf_x1, tf_y1), (tf_x2, tf_y2), (255,0,0), 2)
        cv2.rectangle(img, (g_x1, g_y1), (g_x2, g_y2), (0,255,0), 2)
        cv2.rectangle(img, (r_x1, r_y1), (r_x2, r_y2), (0,0,255), 2)
        
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        
        tf_rect = self.get_area((tf_x1,tf_y1,tf_x2,tf_y2))
        # self.get_logger().info("tf_rect %s" % (tf_rect,))
        
        if self.is_inside((tf_x1,tf_y1,tf_x2,tf_y2),(r_x1,r_y1,r_x2,r_y2)) and self.get_area((g_x1,g_y1,g_x2,g_y2)) < self.get_area((r_x1,r_y1,r_x2,r_y2)):
            self.convert_to_string_and_publish("STOP")
            # self.get_logger().info("STOP")
        else: 
            self.convert_to_string_and_publish("GO")
            # self.get_logger().info("GO")
        debug_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.debug_pub.publish(debug_msg)
       

    def tf_detector(self, img, template=None):
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

        boundaries_high = ([0, 0, 20],[180, 255, 70])

        upper = np.array(boundaries_high[1], dtype='uint8')
        lower = np.array(boundaries_high[0], dtype='uint8')

        mask = cv2.inRange(img_hsv, lower, upper)
        #print(mask)
        #cv2.imshow('img', np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_HSV2BGR))))
        #cv2.imshow('mask',mask)
        #cv2.waitKey(0)

        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


        bounding_box = ((0,0),(0,0))

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x1,y1,xdelta,ydelta = cv2.boundingRect(largest_contour)
            bounding_box = ((x1,y1), (x1+xdelta, y1+ydelta))

        return bounding_box
        
    def green_detector(self, img, template=None):
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

        lower_hue = np.array([80, 200, 200])
        upper_hue = np.array([95, 255, 255])


        upper = np.array(upper_hue, dtype='uint8')
        lower = np.array(lower_hue, dtype='uint8')

        mask = cv2.inRange(img_hsv, lower, upper)
        #print(mask)
        #cv2.imshow('img', np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_HSV2BGR))))
        #cv2.imshow('mask',mask)
        #cv2.waitKey(0)

        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


        bounding_box = ((0,0),(0,0))

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x1,y1,xdelta,ydelta = cv2.boundingRect(largest_contour)
            bounding_box = ((x1,y1), (x1+xdelta, y1+ydelta))

        return bounding_box

    def red_detector(self, img, template=None):
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

        lower_hue = np.array([0, 200, 200])
        upper_hue = np.array([10, 255, 255])


        upper = np.array(upper_hue, dtype='uint8')
        lower = np.array(lower_hue, dtype='uint8')

        mask = cv2.inRange(img_hsv, lower, upper)
        #print(mask)
        #cv2.imshow('img', np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_HSV2BGR))))
        #cv2.imshow('mask',mask)
        #cv2.waitKey(0)

        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


        bounding_box = ((0,0),(0,0))

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x1,y1,xdelta,ydelta = cv2.boundingRect(largest_contour)
            bounding_box = ((x1,y1), (x1+xdelta, y1+ydelta))

        return bounding_box

    def overlap_percentage(self, rect1, rect2):
        # Extract coordinates for each rectangle
        x1_tl, y1_tl, x1_br, y1_br = rect1
        x2_tl, y2_tl, x2_br, y2_br = rect2

        # Calculate the area of each rectangle
        area_rect1 = (x1_br - x1_tl) * (y1_br - y1_tl)
        area_rect2 = (x2_br - x2_tl) * (y2_br - y2_tl)

        # Calculate the coordinates of the overlap rectangle
        x_overlap_tl = max(x1_tl, x2_tl)
        y_overlap_tl = max(y1_tl, y2_tl)
        x_overlap_br = min(x1_br, x2_br)
        y_overlap_br = min(y1_br, y2_br)

        # Check if there is no overlap
        if x_overlap_br <= x_overlap_tl or y_overlap_br <= y_overlap_tl:
            return 0

        # Calculate the area of the overlap rectangle
        area_overlap = (x_overlap_br - x_overlap_tl) * (y_overlap_br - y_overlap_tl)

        # Calculate the percentage of overlap for each rectangle
        overlap_percentage1 = (area_overlap / area_rect1) * 100
        overlap_percentage2 = (area_overlap / area_rect2) * 100

        # Return the minimum overlap percentage
        return min(overlap_percentage1, overlap_percentage2) >= 50

    def is_inside(self, rect2, rect1):
        # Extract coordinates for each rectangle
        x1_tl, y1_tl, x1_br, y1_br = rect1
        x2_tl, y2_tl, x2_br, y2_br = rect2

        # Check if all corners of rect1 are inside rect2
        return (x1_tl >= x2_tl and y1_tl >= y2_tl and
                x1_br <= x2_br and y1_br <= y2_br)  

    def get_area(self, rect):
        # Extract coordinates for each rectangle
        x1_tl, y1_tl, x1_br, y1_br = rect

        # Calculate the area of each rectangle
        area_rect = (x1_br - x1_tl) * (y1_br - y1_tl)
        
        return area_rect


def main(args=None):
    rclpy.init(args=args)
    TF_detector = TFDetector()
    rclpy.spin(TF_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# if __name__ == "__main__":
#     img = cv2.imread('tfl6.jpg')
#     img = cv2.resize(img, (0,0), fx=0.2, fy=0.2) 

#     cv2.imshow('img', img)
#     cv2.waitKey(0)
    
#     tf_bbox = tf_detector(img)
#     tf_coord1, tf_coord2 = tf_bbox
#     tf_x1,tf_y1 = tf_coord1
#     tf_x2,tf_y2 = tf_coord2
    
#     green_bbox = green_detector(img)
#     g_coord1, g_coord2 = green_bbox
#     g_x1,g_y1 = g_coord1
#     g_x2,g_y2 = g_coord2
    
#     red_bbox = red_detector(img)
#     r_coord1, r_coord2 = red_bbox
#     r_x1,r_y1 = r_coord1
#     r_x2,r_y2 = r_coord2
#     cv2.rectangle(img, (tf_x1, tf_y1), (tf_x2, tf_y2), (255,0,0), 2)
#     cv2.rectangle(img, (g_x1, g_y1), (g_x2, g_y2), (0,255,0), 2)
#     cv2.rectangle(img, (r_x1, r_y1), (r_x2, r_y2), (0,0,255), 2)
    
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
    
#     tf_rect = get_area((tf_x1,tf_y1,tf_x2,tf_y2),(g_x1,g_y1,g_x2,g_y2))
#     print(tf_rect)
    
#     if is_inside((tf_x1,tf_y1,tf_x2,tf_y2),(g_x1,g_y1,g_x2,g_y2)) or is_inside((tf_x1,tf_y1,tf_x2,tf_y2),(r_x1,r_y1,r_x2,r_y2)):
#         print('traffic cone present')
#     else:
#         print("no traffic cone")
    
    

