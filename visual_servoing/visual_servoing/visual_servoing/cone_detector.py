#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("cone_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

        #rect_bounds = cd_color_segmentation(image_msg)

        

        #################################

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # h, w = image_orig.shape[:2]
        # self.get_logger().info("%s" % (h*w,))
        h, w = image.shape[:2]

        ## Trapezoid ##
                         
        #pts = np.array([[w//2-75, int(h//2.5)], [w//2+75, int(h//2.5)], [w, int((h//4)*2.25)], [w,h], [0,h],[0, int((h//4)*2.25)]])
        #rect = cv2.boundingRect(pts)
        #tx, ty, tw, th = rect
        #cropped = image[ty:ty+th, tx:tx+tw].copy()

        #pts = pts - pts.min(axis=0)

        #mask = np.zeros(cropped.shape[:2], np.uint8)
        #cv2.drawContours(mask, [pts], -1, (255,255,255), -1, cv2.LINE_AA)

        #dst = cv2.bitwise_and(cropped, cropped, mask=mask)

        #bg = np.ones_like(cropped, np.uint8)*0
        #cv2.bitwise_not(bg, bg, mask=mask)
        #image = bg+dst
                         
        ################


        # height, width = image.shape[:2]
        #  # Calculate the width of each segment
        # segment_width = width // 4
    
        # # Extract the second segment from the left
        # left_segment_start = segment_width
        # left_segment_end = segment_width * 2
        # cropped_image = image[:, left_segment_start:left_segment_end]
        # bounds = cd_color_segmentation(cropped_image)
        # coord1, coord2 = bounds
        # coord11, coord21 = bounds2

        # Calculate the coordinates relative to the original image
        # x1_orig = x1 + left_segment_start
        # x2_orig = x2 + left_segment_start
        # y1_orig = y1
        # y2_orig = y2
        bar_image = image[175:225, :]
        left_half = image[175:225,:w//2]
        
        ### NEW CODE ###

        right_half = image[175:225, (w//2):]
        right_bounds = cd_color_segmentation(right_half)
        right_coord1, right_coord2 = right_bounds

        right_x1, right_y1 = right_coord1
        right_x2, right_y2 = right_coord2



               
        # translation_matrix = np.float32([[1,0,-90],[0,1,0]])
        # img2_rows, img2_cols = image2.shape[:2]

        # image3 = image2[:, img2_cols//3:(img2_cols//3) * 2]

        #image3 = cv2.warpAffine(image2, translation_matrix, (img2_cols, img2_rows))
        # bounds = cd_color_segmentation(image3)
        left_bounds = cd_color_segmentation(left_half)
        left_coord1, left_coord2 = left_bounds
        # coord11, coord21 = bounds2

        

        left_x1, left_y1 = left_coord1
        left_x2, left_y2 = left_coord2

        # x11, y11 = coord11
        # x21, y21 = coord21
        self.get_logger().info("colcon build done")
       # self.get_logger().info("COORDS1 %s %s %s %s" % (x1, y1, x2, y2))
        # self.get_logger().info("COORDS2 %s %s %s %s" % (x11, y11, x21, y21))
        #self.get_logger().info("run")

        x1 = (left_x1 + right_x1+w//2)//2
        x2 = (left_x2 + right_x2+w//2)//2
        y1 = (left_y1 + right_y1)//2
        y2 = (left_y2 + right_y2)//2

        pixel_img_msg = ConeLocationPixel()
        # pixel_img_msg.u = float((x1 + x2)//2 + 155) #+ img2_cols//3)
        # pixel_img_msg.u = float((x1 + x2)//2 - 30) #+ img2_cols//3)
        pixel_img_msg.u = float((x1 + x2)//2) #+ img2_cols//3)
        pixel_img_msg.v = float(y2 + 175)

        self.cone_pub.publish(pixel_img_msg)

        #cv2.rectangle(image3, (x1,y1), (x2,y2), (255,0,0),2)
        # cv2.rectangle(image, (x1+img2_cols//3,y1+ 195), (x2+img2_cols//3,y2+ 195), (255,0,0),2)
        # # img_rows, img_cols = image.shape[:2]
        # self.get_logger().info("colcon build done")
        # #cv2.rectangle(image, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
        


        #cv2.rectangle(image, (x1,y1+175), (x2,y2+175), (255,0,0),2)
        cv2.rectangle(image, (right_x1 + w//2, right_y1+175), (right_x2+w//2, right_y2+175), (0,255,0),2)
        cv2.rectangle(image, (left_x1, left_y1+175), (left_x2, left_y2+175), (0,255,0),2)
        #cv2.rectangle(bar_image, (x1, y1 + 175), (x2, y2 + 175), (0,255,0),2)
        #self.get_logger().info("image")
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

