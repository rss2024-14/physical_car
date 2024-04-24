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
from std_msgs.msg import Int32
from std_msgs.msg import Float32
import time
from .utils import LineTrajectory
from std_msgs.msg import String

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0.85  # FILL IN #
        self.speed = 0.70 # FILL IN #
        self.wheelbase_length = .3  # FILL IN #
        self.initialized_traj = False

        self.index_in_path = None
        self.homogeneous_pts = None
        self.at_goal_time = None
        self.traffic_stop = False

        self.traffic_lights = [(-12.276, 13.803)]
        self.goal_indices = []


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

        # self.robot_y_pub = self.create_publisher(Float32, "/robot_y", 1)
        self.stop_indices = []
        self.stop_sub = self.create_subscription(Int32,
                                                 "/stop_pub",
                                                 self.stop_cb,
                                                 1)

        self.clicked_indices_sub = self.create_subscription(
            Int32,
            '/clicked_indices',
            self.clicked_indices_cb,
            10
        )

        # self.traffic_sub = self.create_subscription(String,
        #                                          "/traffic_light",
        #                                          self.traffic_callback,
        #                                          1)

    def stop_cb(self, msg):
        self.stop_indices.append(msg.data)
        
    def traffic_callback(self, msg):
        traffic_status = msg.data
        if msg.data == "STOP" and self.traffic_stop:
            self.speed = 0
            self.get_logger().info("TRAFFIC STOP")
        else:
            self.speed = 0.70

    def clicked_indices_cb(self, msg):
        self.goal_indices.append(msg.data)
        # self.get_logger().info("%s", (self.goal_indices,))

    def distance(self, p1, p2):
        return ( (p2[1] - p1[1])**2 + (p2[0] - p2[0])**2 )**0.5

    def convert_to_float_and_publish(self, data):
        msg = Float32()
        msg.data = data
        self.robot_y_pub.publish(msg)
        self.get_logger().info("Robot y %s" % (data,))

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj:
            return
        
        if (self.at_goal_time and time.time() - self.at_goal_time < 5):
            self.speed = 0.0
        else:
            self.speed = 0.70
            self.at_goal_time = None

        pts = self.trajectory.points            
        robot_xy = [ odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y ]
        # self.convert_to_float_and_publish(robot_xy[1])
        
        q = [
            odometry_msg.pose.pose.orientation.x,
            odometry_msg.pose.pose.orientation.y,
            odometry_msg.pose.pose.orientation.z,
            odometry_msg.pose.pose.orientation.w
        ]
        *_, heading = tf_transformations.euler_from_quaternion([q[0], q[1], q[2], q[3]])

        world_to_robot_tf = np.linalg.inv( self.tf_matrix(heading, robot_xy[0], robot_xy[1]) ) # world frame to robot frame

        # for pt in self.traffic_lights:
        #     if self.distance(pt, robot_xy) < 5:
        #         self.traffic_stop = True
        #         self.get_logger().info("TRAFFIC INTERSECTION")
        #     else:
        #         self.traffic_stop = False
                # self.get_logger().info("NO TRAFFIC")

        if self.index_in_path is None:
            self.index_in_path = self.find_closest(pts, robot_xy, world_to_robot_tf)

        target = self.trajectory.points[self.index_in_path]

        if self.distance(target, robot_xy) < self.lookahead:
            if self.index_in_path == self.goal_indices[0] and self.at_goal_time is None:
                self.goal_indices.pop(0)
                self.at_goal_time = time.time()
            
            # if self.index_in_path in self.stop_indices and self.at_goal_time is None:
            #     self.at_goal_time = time.time() - 2

            self.index_in_path = min(self.index_in_path + 1, len(pts)-1)
            target = self.trajectory.points[self.index_in_path]
        
        # if self.index_in_path == len(pts) - 1 and self.distance(target, robot_xy) < self.lookahead:
        #     log(self, "Actual trajectory", self.actual_traj)
        #     self.get_logger().info(f"Planned Trajectory: {pts}")
        #     if not self.saved: 
        #         np.savetxt("actual_traj_curved.csv", self.actual_traj)
        #         # np.savetxt("actual_traj_y.csv", list(map(lambda x: x[1], self.actual_traj)), ",")
        #         np.savetxt("planned_curved.csv", pts)
        #         # np.savetxt("planned_y.csv", list(map(lambda x: x[1], pts)), ",")
        #         self.saved = True
        
        target_in_robot_frame = world_to_robot_tf @ self.homogenize_pt(target)
        target_rotation = self.angle(target_in_robot_frame)
        # log(self, "rotation from robot", target_rotation * 180/np.pi)
        # self.get_logger().info("target rotation %s" % (target_rotation*180/np.pi,))
        # self.get_logger().info("heading %s" % (heading*180/np.pi,))
        # self.get_logger().info("steering angle %s" % ( (target_rotation)*180/np.pi,))

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now

        drive_cmd = self.build_drive_cmd(self.speed, target_rotation * 0.29 + .02)
        self.drive_pub.publish(drive_cmd)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        self.prev_time = self.get_clock().now()
        self.actual_traj = []
        self.saved = False
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
        if not reachable_pts:
            return 1

        all_distances = [self.distance(pt, robot_pt) for pt in pts_in_robot_frame]
        reachable_distances = [self.distance(pt, robot_pt) for pt in reachable_pts]
        min_distance = min(reachable_distances)
        log(self, "chosen point", all_distances.index(min_distance))
        log(self, "number of points", len(all_distances))
        return all_distances.index(min_distance)
    
    def build_drive_cmd(self, speed, delta):
        """
        Args:
            delta: steering angle in radians
        Returns:
            AckermannDriveStamped ros message
        """
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        drive_cmd.drive.speed = speed
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
