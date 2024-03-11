#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import tf_transformations

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):
    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        self.FILTERED_SCAN_TOPIC = "/filtered_scan"
        self.ODOM_TOPIC = "/odom"
        self.LOOKAHEAD_TOPIC = "/lookahead"
        self.WALL_TOPIC = "/wall"

        self.subscription = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.driver_callback,
            10)
        
        self.subscription_odom = self.create_subscription(
            Odometry,
            self.ODOM_TOPIC,
            self.odom_callback,
            10)

        self.L1 = 1
        self.L = .3 # car length
        self.heading = 0

        self.Kp = 10
        self.Kd = 0
        self.prev_error = 0
        self.dedt = 0
        self.ref = self.DESIRED_DISTANCE
        self.num_wall_pts = 260
		
	# TODO: Initialize your publishers and subscribers here
        self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 1)
        self.lookahead_pub = self.create_publisher(Marker, self.LOOKAHEAD_TOPIC, 1)
        self.filtered_scan_pub = self.create_publisher(LaserScan, self.FILTERED_SCAN_TOPIC, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 1)
        self.subscription  # prevent unused variable warning
    
    def compute_filtered_msg(self, scan, desired_angles):
        current_angles = np.array([scan.angle_min, scan.angle_max])
        desired_angles *= np.pi/180
        i_range = ( (desired_angles - current_angles) // scan.angle_increment ).astype(int)
        filtered_scan = LaserScan()
        filtered_range = scan.ranges[i_range[0] : i_range[1]]
        filtered_scan.header.stamp = self.get_clock().now().to_msg()
        filtered_scan.header.frame_id = scan.header.frame_id
        filtered_scan.angle_min = scan.angle_min + scan.angle_increment * i_range[0]
        filtered_scan.angle_max = scan.angle_max + scan.angle_increment * i_range[1]
        filtered_scan.angle_increment = scan.angle_increment
        filtered_scan.scan_time = scan.scan_time
        filtered_scan.range_min = float( min(filtered_range) )
        filtered_scan.range_max = float( max(filtered_range) )
        filtered_scan.ranges = filtered_range
        # self.get_logger().info("%s filtered scan" % filtered_scan)
        return filtered_scan

    def driver_callback(self, msg):
        # >>> SCAN FILTERING
        to_left = self.SIDE == 1
        if to_left: 
            desired_angles = np.array([-105., 80.]) #degrees. default: [-135, 135]
        else:
            desired_angles = np.array([-80., 105.]) #degrees. default: [-135, 135]
        scan = self.compute_filtered_msg(msg, desired_angles)
        num_pts = len(scan.ranges)
        self.filtered_scan_pub.publish(scan)

        # >>> SCAN BIASING
        num_biased_pts = 65 # num center pts with distance bias
        left_region = scan.ranges[num_pts//2 + num_biased_pts :]
        bias_region = scan.ranges[num_pts//2 - num_biased_pts : num_pts//2 + num_biased_pts]
        right_region = scan.ranges[: num_pts//2 - num_biased_pts]

        # >>> SCAN HALVING
        bias = .6
        half_scan = []
        if (to_left): #angles > 0
            half_scan = np.concatenate( ( bias * np.array(bias_region), left_region) )
        else:
            half_scan = np.concatenate( ( right_region, bias * np.array(bias_region) ) )

        # >>> SCAN CLOSEST PTS ID
        half_scan = half_scan[half_scan >= .1]
        closest_pts = np.sort(half_scan)[:self.num_wall_pts]
        closest_angles = np.array([])

        if to_left:
            start_i = num_pts//2 - num_biased_pts
            closest_angles = scan.angle_min + (scan.angle_increment * start_i) + scan.angle_increment * np.argsort(half_scan)[:self.num_wall_pts]
        else:
            start_i = 0
            closest_angles = scan.angle_min + scan.angle_increment * np.argsort(half_scan)[:self.num_wall_pts]

        # >>> LIN REG
        closest_xs = closest_pts * np.cos(closest_angles)
        closest_ys = closest_pts * np.sin(closest_angles)
        lin_reg = np.polyfit(closest_xs, closest_ys, 1)
        m = lin_reg[0]
        b = lin_reg[1]
        sample_pts = np.arange(-1, 1, .04)
        y = m * sample_pts + b

        # >>> CONVENIENCE PLOTS
        x_circ_pts = np.linspace(-self.L1, self.L1, 50)
        x_circ_pts_twice = np.tile(x_circ_pts,2)
        y_circ = np.concatenate( (np.sqrt(self.L1**2 - x_circ_pts**2), -np.sqrt(self.L1**2 - x_circ_pts**2)) )
        # VisualizationTools.plot_line(x_circ_pts_twice, y_circ, self.lookahead_pub, color=(1.,1.,1.), frame="/laser")
        VisualizationTools.plot_line(closest_xs, closest_ys, self.lookahead_pub, color=(1.,1.,1.), frame="/laser")

        VisualizationTools.plot_line(sample_pts, y, self.line_pub, color=(0.,0.796,1.), frame="/laser")

        # >>> PURE PURSUIT
        perp_dist = np.abs(b) / np.sqrt(m**2 + 1)

        if (self.L1 <= perp_dist):
            angle_robot_to_intersect = 0
        else:
            angle_robot_to_intersect = np.pi/2 - np.arccos(perp_dist / self.L1)
        
        if (b < 0):
            angle_robot_to_intersect *= -1

        eta = -self.heading + angle_robot_to_intersect
        delta = np.arctan(2 * self.L * np.sin(eta) / self.L1)

        # PD
        error = -self.SIDE * (self.ref - perp_dist)
        dedt = self.SIDE * ( error - self.prev_error )
        u = self.Kp * error + self.Kd * dedt
        delta += u
        self.prev_error = error

        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        drive_cmd.drive.speed = self.VELOCITY
        drive_cmd.drive.steering_angle = delta # np.clip(delta, -60*np.pi/180, 60*np.pi/180)
        # self.get_logger().info("%s perp_dist\n%s eta\n%s delta" % (perp_dist, eta * 180/np.pi, delta*180/np.pi))
        self.drive_pub.publish(drive_cmd)
        
        self.get_logger().info("%s steering angle, %s num_pts" % (drive_cmd.drive.steering_angle*180/np.pi, num_pts) )

    def odom_callback(self, msg):
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        mat = tf_transformations.quaternion_matrix(q)
        z_rot = np.arctan2(mat[1, 0], mat[0, 0]) #* 180/np.pi
        # self.get_logger().info("%s z rot" % z_rot)
        self.heading = z_rot


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()