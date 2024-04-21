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
        mat = tf_transformations.quaternion_matrix(q)
        z_rot = np.arctan2(mat[1, 0], mat[0, 0])
        self.heading = z_rot
        
        pts = self.trajectory.points
        closest_pts = self.find_closest_points(pts, robot_xy)

        # Linear regression
        # x_coords = list( map(lambda pt: pt[0], closest_pts) )
        # y_coords = list( map(lambda pt: pt[1], closest_pts) )

        m = ( closest_pts[1][1] - closest_pts[0][1] ) / (closest_pts[1][0] - closest_pts[0][0])
        b = -m * closest_pts[0][0] + closest_pts[0][1]

        self.get_logger().info("robot_xy %s" % (robot_xy,))
        # self.get_logger().info("x_coords %s" % (x_coords,))
        self.get_logger().info("closest_pts %s" % (closest_pts,))

        # lin_reg = np.polyfit(x_coords, y_coords, 1)
        # m = lin_reg[0]
        # b = lin_reg[1]
        sample_pts = np.arange(-5, 1, .04)
        y = m * sample_pts + b
        self.get_logger().info("closest points %s" % (closest_pts,))
        self.get_logger().info("line %s %s" % (m, b))
        # self.plot_line(sample_pts, y, self.line_pub, color=(0.,0.796,1.))
        self.plot_line([closest_pts[0][0], closest_pts[1][0]], [closest_pts[0][1], closest_pts[1][1]], self.line_pub, color=(0.,0.796,1.))
        # Pure pursuit
        perp_dist = np.abs(b) / np.sqrt(m**2 + 1)

        if (self.lookahead <= perp_dist):
            angle_robot_to_intersect = 0
        else:
            angle_robot_to_intersect = np.pi/2 - np.arccos(perp_dist / self.lookahead)
        
        if (b < 0):
            angle_robot_to_intersect *= -1

        eta = -self.heading + angle_robot_to_intersect
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
        return sorted(pts, key=lambda p: self.distance(p, ref_pt))[:2]
    
    def build_drive_cmd(self, delta):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        drive_cmd.drive.speed = self.speed
        drive_cmd.drive.steering_angle = delta
        return drive_cmd
    
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
