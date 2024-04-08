from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy
import numpy as np
import threading
import time

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('num_particles', "default")
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)
        
        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)


        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.all_pose_pub = self.create_publisher(PoseArray, "/particles", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.average_motion_model = MotionModel(self, deterministic = True)
        self.sensor_model = SensorModel(self)

        self.particles = []
        self.probs = [1] * self.num_particles
        self.received_particles = False
        self.weighted_avg = [0, 0, 0]
        self.prev_time = time.time()
        self.lock = threading.Lock()

        self.get_logger().info("=============+READY+=============")


    def pose_callback(self, pose_data):
        """
        Initialize particles

        ("Ideally with some sort of interactive interface in rviz")
        """
        self.get_logger().info("POSE CALLBACK RUN")

        # Converting pose_data to desired float variables
        x = pose_data.pose.pose.position.x
        y = pose_data.pose.pose.position.y
        theta = 2*np.arccos(pose_data.pose.pose.orientation.w) #Converting from quaternion

        # Create particles based on this pose
        x_vals = np.random.normal(loc=x, scale=.01, size=self.num_particles)
        y_vals = np.random.normal(loc=y, scale=.01, size=self.num_particles)
        theta_vals = np.random.normal(loc=theta, scale=.01, size=self.num_particles)

        self.particles = np.concatenate((x_vals.reshape(-1,1), y_vals.reshape(-1,1), theta_vals.reshape(-1,1)), axis=1)

        # First time initializing, so can now set "prev" as the starting pose and ready to perform other ops
        self.prev_pose = np.array([x, y, theta])
        self.received_particles = True


    def odom_callback(self, odom_data):
        """
        Whenever we get odometry data, use the motion model to update the particle positions
        """

        if self.received_particles:
            with self.lock:
                self.get_logger().info("ODOM CALLBACK")

                # Get new pose
                x = odom_data.twist.twist.linear.x
                y = odom_data.twist.twist.linear.y
                theta = 2*np.arccos(odom_data.twist.twist.angular.z)

                current_pose = np.array([x,y,theta])
                current_time = time.time()

                dt = current_time - self.prev_time

                # Actually calling motion model with particles and new deltaX
                delta_x = ( self.rot(self.prev_pose[2]) @ (current_pose - self.prev_pose).T ) * dt
                self.particles = self.motion_model.evaluate(self.particles, delta_x) #+ np.random.normal(scale=0.001, size=(len(self.particles), 3))

                self.prev_pose = current_pose
                self.prev_time = current_time

                #Because particles have been updated,
                self.publish_pose_info()

    def laser_callback(self, scan_data):
        """
        Whenever we get sensor data, use the sensor model to compute the particle probabilities. 
        Then resample the particles based on these probabilities
        """

        if self.received_particles:
            with self.lock:
                self.get_logger().info("LASER CALLBACK")

                ranges = scan_data.ranges
                if len(ranges) > 100:
                    ranges = ranges[ : : len(ranges) // 100]

                self.probs = self.sensor_model.evaluate(self.particles, ranges)

                # Resampling the Particles
                indices = np.random.choice(len(self.particles), size=self.num_particles, p=self.probs, replace=True)
                self.particles = self.particles[indices]

                #Because particles have been updated,
                self.publish_pose_info()


    def publish_pose_info(self):
        """
        Anytime particles are updated via motion or sensor, publishing particles and the calculated avg
        """

        #Publishing all particles
        poses_msg = PoseArray()
        poses_msg.header.frame_id = "/map"

        poses_msg.poses = self.particles_to_poses(self.particles)

        self.all_pose_pub.publish(poses_msg)

        # Calculate average
        x_mean, y_mean = np.average(self.particles[:,:2], axis=0, weights=self.probs) #Grabbing x,y of each particle and avg
        theta_mean = np.arctan2(
            np.average(np.sin(self.particles[:,2]), weights=self.probs), 
            np.average(np.cos(self.particles[:,2]), weights=self.probs)
            ) #Circular mean eq
                        
        self.weighted_avg = [x_mean, y_mean, theta_mean]
        
        #Publishing average particle pose
        odom_msg = Odometry()
        odom_msg.header.frame_id = "/map"

        odom_msg.pose.pose = self.particles_to_poses([self.weighted_avg])[0]

        self.odom_pub.publish(odom_msg)

    
    def particles_to_poses(self, particles):
        poses = []
        for particle in particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = np.sin(1/2 * particle[2])
            pose.orientation.w = np.cos(1/2 * particle[2])

            poses.append(pose)

        return poses


    def rot(self, angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        return np.array([[cos_theta, -sin_theta, 0],
                        [sin_theta,  cos_theta, 0 ],
                        [0, 0, 1]])

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()



# Important Notes All Moved Here:
    
    #  *Important Note #1 (L27):* It is critical for your particle
    #     filter to obtain the following topic names from the
    #     parameters for the autograder to work correctly. Note
    #     that while the Odometry message contains both a pose and
    #     a twist component, you will only be provided with the
    #     twist component, so you should rely only on that
    #     information, and *not* use the pose component.

    #  *Important Note #2 (L41):* You must respond to pose
    #     initialization requests sent to the /initialpose
    #     topic. You can test that this works properly using the
    #     "Pose Estimate" feature in RViz, which publishes to
    #     /initialpose.

    #  *Important Note #3 (L46):* You must publish your pose estimate to
    #     the following topic. In particular, you must use the
    #     pose field of the Odometry message. You do not need to
    #     provide the twist part of the Odometry message. The
    #     odometry you publish here should be with respect to the
    #     "/map" frame.

    # Make sure you include some way to initialize
    # your particles, ideally with some sort
    # of interactive interface in rviz
    #
    # Publish a transformation frame between the map
    # and the particle_filter_frame.