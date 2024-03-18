#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        #self.declare_parameter("drive_topic")
        #DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        #self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone", 
            self.relative_cone_callback, 1)

        self.velocity = 1.0
        self.parking_distance = 0.75 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        self.relative_distance = np.sqrt(self.relative_x**2 + self.relative_y**2)
        self.relative_angle = np.arctan(self.relative_y/self.relative_x) # rad
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd

        #################################

        # Checking if the cone is behind the car, too close to the car, or too far from the car
        # If the cone is behind the car, back the car up until the cone is in front of the car
        # If the cone is too close to the car, back up the car up
        # If the cone is too far from the car, move towards the cone
        if self.relative_x <= 0:
            drive_cmd.drive.speed = -1.0 * self.velocity
            drive_cmd.drive.steering_angle = 0.0
        elif self.relative_distance < (self.parking_distance - 0.05):
            drive_cmd.drive.speed = -1.0 * self.velocity
            drive_cmd.drive.steering_angle = 0.0
        elif self.relative_distance > (self.parking_distance + 0.05):
            # If the cone is to the right of the car, the steering angle is negative
            # If the cont is to the left of the car, the steering angle is positive
            drive_cmd.drive.speed = self.velocity
            drive_cmd.drive.steering_angle = self.relative_angle
        elif self.relative_distance <= (self.parking_distance + 0.05) and self.relative_distance >= (self.parking_distance - 0.05):
            if self.relative_angle > 0.05:
                drive_cmd.drive.speed = -1.0 * self.velocity
                drive_cmd.drive.steering_angle = -1.0 * self.relative_angle
            elif self.relative_angle < -0.05:
                drive_cmd.drive.speed = -1.0 * self.velocity
                drive_cmd.drive.steering_angle = -1.0 * self.relative_angle
            elif self.relative_angle <= 0.05 and self.relative_angle >= -0.05:
                drive_cmd.drive.speed = 0.0
                drive_cmd.drive.steering_angle = 0.0

        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)

        #################################
        
        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = self.relative_distance
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
