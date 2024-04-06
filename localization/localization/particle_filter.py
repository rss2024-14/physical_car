from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy
import numpy as np
import threading

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        #To edit depending on how many particles we think we can handle
        self.declare_parameter('num_particles', "default")
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value


        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.particles = []
        self.lock = threading.Lock()

        self.get_logger().info("=============+READY+=============")


        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def pose_callback(self, pose_data):
        """
        Initialize particles

        ("Ideally with some sort of interactive interface in rviz")
        """
        # Converting pose_data to desired float variables
        x = pose_data.pose.pose.position.x
        y = pose_data.pose.pose.position.y
        theta = 2*np.arccos(pose_data.pose.pose.orientation.w) #Converting from quaternion

        # Create particles based on this pose
            # Need to sample to only have as many as we want based on num_particles
        x_vals = 0
        y_vals = 0
        theta_vals = 0

        self.particles = []

    def odom_callback():
        """
        Whenever we get odometry data, use the motion model to update the particle positions
        """
        pass
        # get x,y, theta
    
    def laser_callback(self, scan):
        """
        Whenever we get sensor data, use the sensor model to compute the particle probabilities. 
        Then resample the particles based on these probabilities
        """
        # Use lock

    ## Other helper functions we might need:
        # Converting particle back to pose
        # 

    



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
