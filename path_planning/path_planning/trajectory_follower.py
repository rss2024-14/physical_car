import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import tf_transformations

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
        self.initialized_traj = False

        self.index_in_path = None

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.odom_sub = self.create_subscription(Odometry,
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
        
        self.line_pub = self.create_publisher(Marker, "/wall", 1)

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj:
            return
        
        robot_xy = [ odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y ]
        q = [
            odometry_msg.pose.pose.orientation.x,
            odometry_msg.pose.pose.orientation.y,
            odometry_msg.pose.pose.orientation.z,
            odometry_msg.pose.pose.orientation.w
        ]
        *_, theta = tf_transformations.euler_from_quaternion([q[0], q[1], q[2], q[3]])

        # mat = tf_transformations.quaternion_matrix(q)
        # z_rot = np.arctan2(mat[1, 0], mat[0, 0])
        heading = theta
        
        pts = self.trajectory.points
        if self.index_in_path is None:
            self.index_in_path = self.find_closest(pts, robot_xy)

        # self.get_logger().info("path index %s" % (self.index_in_path,))

        target = self.trajectory.points[self.index_in_path]
        if self.distance(target, robot_xy) < self.lookahead:
            self.index_in_path += 1
            self.index_in_path = min(self.index_in_path, len(self.trajectory.points)-1)

        target_in_robot_frame = np.linalg.inv( self.tf_matrix(theta, robot_xy[0], robot_xy[1]) ) @ np.array([[target[0]], [target[1]], [1]])
        
        # self.get_logger().info("target x in RF %s, target y in RF %s" % (target_in_robot_frame[0], target_in_robot_frame[1]))

        target_rotation = np.arctan2(target_in_robot_frame[1], target_in_robot_frame[0])[0]
        self.get_logger().info("target rotation %s" % (target_rotation*180/np.pi,))
        self.get_logger().info("heading %s" % (heading*180/np.pi,))
        # self.get_logger().info("steering angle %s" % ( (heading - target_rotation)*180/np.pi,))
        drive_cmd = self.build_drive_cmd(heading - target_rotation)
        self.drive_pub.publish(drive_cmd)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_closest(self, pts, ref_pt):
        distances = [self.distance(pt, ref_pt) for pt in pts]
        min_distance = min(distances)
        return distances.index(min_distance)
    
    def build_drive_cmd(self, delta):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        drive_cmd.drive.speed = self.speed
        drive_cmd.drive.steering_angle = delta
        return drive_cmd
    
    def tf_matrix(self, angle, x, y):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        return np.array([[cos_theta, -sin_theta, x],
                        [sin_theta,  cos_theta, y],
                        [0, 0, 1]])
    
    def plot_line(self, x, y, publisher, color = (1., 0., 0.), frame = "/base_link"):
        """
        Publishes the points (x, y) to publisher
        so they can be visualized in rviz as
        connected line segments.
        Args:
            x, y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """
        # Construct a line
        line_strip = Marker()
        line_strip.type = Marker.CUBE_LIST
        line_strip.header.frame_id = frame

        # Set the size and color
        line_strip.scale.x = 0.5
        line_strip.scale.y = 0.5
        line_strip.scale.z = 0.5
        line_strip.color.a = 1.
        line_strip.color.r = color[0]
        line_strip.color.g = color[1]
        line_strip.color.g = color[2]

        # Fill the line with the desired values
        for xi, yi in zip(x, y):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        # Publish the line
        publisher.publish(line_strip)


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
