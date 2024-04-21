import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import math

from nav_msgs.msg import Odometry
import numpy as np

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0.3  # FILL IN #
        self.speed = 0.5  # FILL IN #
        self.wheelbase_length = .3  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(Odometry,
                                                 "/pf/pose/odom",
                                                 self.pose_callback,
                                                 1)
        
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

    def pose_callback(self, odometry_msg):
        q = [
            odometry_msg.pose.pose.orientation.x,
            odometry_msg.pose.pose.orientation.y,
            odometry_msg.pose.pose.orientation.z,
            odometry_msg.pose.pose.orientation.w
        ]

        if not self.initialized_traj:
            return
        
        robot_xy = (q[0], q[1])
        pts = self.trajectory.points
        closest_pts = self.find_closest_points(pts, robot_xy)

        # Linear regression
        x_coords = list( map(lambda pt: pt[0], closest_pts) )
        y_coords = list( map(lambda pt: pt[1], closest_pts) )
        self.get_logger().info("x_coords %s" % (x_coords,))
        lin_reg = np.polyfit(x_coords, y_coords, 1)
        m = lin_reg[0]
        b = lin_reg[1]
        sample_pts = np.arange(-1, 1, .04)
        y = m * sample_pts + b

        # Pure pursuit
        perp_dist = np.abs(b) / np.sqrt(m**2 + 1)

        if (self.lookahead <= perp_dist):
            angle_robot_to_intersect = 0
        else:
            angle_robot_to_intersect = np.pi/2 - np.arccos(perp_dist / self.lookahead)
        
        if (b < 0):
            angle_robot_to_intersect *= -1

        eta = angle_robot_to_intersect
        delta = np.arctan(2 * self.wheelbase_length * np.sin(eta) / self.lookahead)
        
        drive_cmd = self.build_drive_cmd(delta)
        self.drive_pub.publish(drive_cmd)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_closest_points(self, pts, ref_pt):
        return sorted(pts, key=lambda p: self.distance(p, ref_pt))[:15]
    
    def build_drive_cmd(self, delta):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        drive_cmd.drive.speed = self.speed
        drive_cmd.drive.steering_angle = delta
        return drive_cmd


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
