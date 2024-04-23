import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import tf_transformations
from .utils import log
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

        self.lookahead = 0.6  # FILL IN #
        self.speed = 0.5 # FILL IN #
        self.wheelbase_length = .3  # FILL IN #
        self.initialized_traj = False

        self.index_in_path = None
        self.homogeneous_pts = None
        self.actual_traj = None
        self.print_traj = False

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

        pts = self.trajectory.points            
        robot_xy = [ odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y ]
        q = [
            odometry_msg.pose.pose.orientation.x,
            odometry_msg.pose.pose.orientation.y,
            odometry_msg.pose.pose.orientation.z,
            odometry_msg.pose.pose.orientation.w
        ]
        *_, heading = tf_transformations.euler_from_quaternion([q[0], q[1], q[2], q[3]])

        world_to_robot_tf =  np.linalg.inv( self.tf_matrix(heading, robot_xy[0], robot_xy[1]) ) # world frame to robot frame

        if self.index_in_path is None:
            self.index_in_path = self.find_closest(pts, robot_xy, world_to_robot_tf)

        target = self.trajectory.points[self.index_in_path]
        if self.distance(target, robot_xy) < self.lookahead:
            self.index_in_path = min(self.index_in_path + 1, len(self.trajectory.points)-1)
            log(self, "Seeking new index", self.index_in_path)

        # Collecting points for actual trajectory
        if self.actual_traj is None:
            self.actual_traj = []
        
        self.actual_traj.append(tuple(robot_xy))
        
        if self.index_in_path == len(pts) - 1:
            if not self.print_traj:
                self.get_logger().info(f"Actual Trajectory: {self.actual_traj}")
                self.get_logger().info(f"Planned Trajectory: {pts}")
                self.print_traj = True
        
        target_in_robot_frame = world_to_robot_tf @ self.homogenize_pt(target)
        target_rotation = self.angle(target_in_robot_frame)
        # self.get_logger().info("target rotation %s" % (target_rotation*180/np.pi,))
        # self.get_logger().info("heading %s" % (heading*180/np.pi,))
        # self.get_logger().info("steering angle %s" % ( (target_rotation)*180/np.pi,))
        drive_cmd = self.build_drive_cmd(target_rotation)
        self.drive_pub.publish(drive_cmd)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.homogeneous_pts = [self.homogenize_pt(pt) for pt in self.trajectory.points]
        self.initialized_traj = True

    def homogenize_pt(self, pt):
        """
        Args: 
            pt : xy-point in format [x,y]
        Returns:
            (3,1) np array in format [x,y,1]
        """
        return np.array([pt[0], pt[1], 1]).reshape(3,1)
    
    def distance(self, p1, p2):
        """
        Args:
            p1, p2: pts [x,y]
        Returns:
            Euclidean distance between p1 and p2
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def angle(self, pt):
        """
        Args:
            pt: pt as homogeneous np array in robot frame
        Returns:
          Angle of point relative to robot's heading.
        """
        return np.arctan2(pt[1], pt[0])[0]
    
    def find_closest(self, pts, robot_pt, world_to_robot_tf):
        """
        Args:
            pts: array of pts [x,y] in trajectory
            robot_pt: location of robot [x,y] in world frame
            world_to_robot_tf: 3x3 transformation matrix (rotation + translation)
        Returns:
            index of pt in pts with minimum distance to robot and within reasonable range of heading
        """
        pts_in_robot_frame = [world_to_robot_tf @ pt for pt in self.homogeneous_pts]
        reachable_pts = list( filter(lambda x: abs( self.angle(x) ) <= 0.8 and x[0] >= 2, pts_in_robot_frame) ) # pts within 50 degrees of robot heading
        # log(self, "orig points", pts_in_robot_frame)
        # log(self, "orig points length", len(pts_in_robot_frame))
        # log(self, "reachable points", len(reachable_pts))
        # log(self, "angles", [self.angle(x) for x in pts_in_robot_frame])
        all_distances = [self.distance(pt, robot_pt) for pt in pts_in_robot_frame]
        reachable_distances = [self.distance(pt, robot_pt) for pt in reachable_pts]
        min_distance = min(reachable_distances)
        log(self, "chosen point", all_distances.index(min_distance))
        return all_distances.index(min_distance)
    
    def build_drive_cmd(self, delta):
        """
        Args:
            delta: steering angle in radians
        Returns:
            AckermannDriveStamped ros message
        """
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        drive_cmd.drive.speed = self.speed
        drive_cmd.drive.steering_angle = delta
        return drive_cmd
    
    def tf_matrix(self, angle, x, y):
        """
        Args:
            x,y,angle: translation and rotation angle
        Returns:
            3x3 transformation matrix
        """
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
