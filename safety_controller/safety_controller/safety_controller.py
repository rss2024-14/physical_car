#!/usr/bin/env python3                                                                                                                                      
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import tf_transformations

class SafetyController(Node):
    def __init__(self):
        super().__init__("safety_controller")
        # Declare parameters to make them available for use                                                                                                                        
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("safety_topic", "default")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")

        # Fetch constants from the ROS parameter server                                                                                                                            
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SAFETY_TOPIC = self.get_parameter('safety_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        # TODO: Initialize your publishers and subscribers here
        self.scan_data = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.safety_callback,
            10)

        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            self.driver_callback,
            10)
        
        self.safety_cmds = self.create_publisher(
            AckermannDriveStamped,
            self.SAFETY_TOPIC,
            1)

        self.subscription  # prevent unused variable warning

    def safety_callback(self, msg):
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value

        num_pts = len(msg.ranges)

        ahead_scan = msg.ranges[num_pts//2-50:num_pts//2+50]
        ahead_distance = np.median(ahead_scan)

        # Braking distance for a car is proportional to the square of the car's speed (Must be changed for physical car)

        if ahead_distance < 0.6:
            # self.get_logger().info("!!!!!!!!!!!!!!! SAFETY INTERCEPT.")
            safety_cmd = AckermannDriveStamped()
            safety_cmd.header.stamp = self.get_clock().now().to_msg()
            safety_cmd.header.frame_id = "/base_link"
            safety_cmd.drive.speed = -0.7
            safety_cmd.drive.steering_angle = -np.pi
            self.safety_cmds.publish(safety_cmd)

    def driver_callback(self, msg):
        pass


def main():
    rclpy.init()
    safety_controller = SafetyController()
    rclpy.spin(safety_controller)
    safety_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

