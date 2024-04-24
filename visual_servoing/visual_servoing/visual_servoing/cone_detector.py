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

        # height, width = image.shape[:2]
        #  # Calculate the width of each segment
        # segment_width = width // 4
    
        # # Extract the second segment from the left
        # left_segment_start = segment_width
        # left_segment_end = segment_width * 2
        # cropped_image = image[:, left_segment_start:left_segment_end]
        # bounds = cd_color_segmentation(cropped_image)
        # coord1, coord2 = bounds
        # # coord11, coord21 = bounds2

        # x1, y1 = coord1
        # x2, y2 = coord2
        # # Calculate the coordinates relative to the original image
        # x1_orig = x1 + left_segment_start
        # x2_orig = x2 + left_segment_start
        # y1_orig = y1
        # y2_orig = y2

        image2 = image[195:245,:]
        # translation_matrix = np.float32([[1,0,-160],[0,1,0]])
        # img2_rows, img2_cols = image2.shape[:2]
        # image3 = cv2.warpAffine(image2, translation_matrix, (img2_cols, img2_rows))
        # bounds = cd_color_segmentation(image3)
        bounds = cd_color_segmentation(image2)
        coord1, coord2 = bounds
        # coord11, coord21 = bounds2

        

        x1, y1 = coord1
        x2, y2 = coord2

        # x11, y11 = coord11
        # x21, y21 = coord21

        self.get_logger().info("COORDS1 %s %s %s %s" % (x1, y1, x2, y2))
        # self.get_logger().info("COORDS2 %s %s %s %s" % (x11, y11, x21, y21))

        pixel_img_msg = ConeLocationPixel()

        pixel_img_msg.u = float((x1 + x2)//2)
        pixel_img_msg.v = float(y2 + 195)

        self.cone_pub.publish(pixel_img_msg)

        #cv2.rectangle(image, (x1,y1+ 195), (x2,y2+ 195), (255,0,0),2)
        # img_rows, img_cols = image.shape[:2]

        # image[img_rows//2-5:img_rows//2+6, img_cols//2] = (255,0,0) 
        # image[img_rows//2, img_cols//2-5:img_cols//2+6] = (255,0,0)

        # cv2.rectangle(image, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), 2)
        cv2.rectangle(image, (x1,y1+ 195), (x2,y2+ 195), (255,0,0),2)
        
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

