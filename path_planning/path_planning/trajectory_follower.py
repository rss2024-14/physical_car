import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker
from rclpy.node import Node
import numpy as np
from path_planning.visualization_tools import VisualizationTools

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

        self.lookahead = 0  # FILL IN #
        self.speed = 0  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        self.lookahead_pub = self.create_publisher(Marker, '/lookahead', 1)

    def pose_callback(self, odometry_msg):
        
        x_circ_pts = np.linspace(-self.L1, self.L1, 50)
        x_circ_pts_twice = np.tile(x_circ_pts,2)
        y_circ = np.concatenate( (np.sqrt(self.L1**2 - x_circ_pts**2), -np.sqrt(self.L1**2 - x_circ_pts**2)) )
        VisualizationTools.plot_line(x_circ_pts_twice, y_circ, self.lookahead_pub, color=(1.,1.,1.), frame="/laser")

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
