#!/usr/bin/env python3
import rclpy
import time
from geometry_msgs.msg import PoseArray
from rclpy.node import Node

from path_planning.utils import LineTrajectory


class LoadTrajectory(Node):
    """ Loads a trajectory from the file system and publishes it to a ROS topic.
    """

    def __init__(self):
        super().__init__("trajectory_loader")

        self.declare_parameter("trajectory1", "default")
        self.declare_parameter("trajectory2", "default")
        
        self.path1 = self.get_parameter("trajectory1").get_parameter_value().string_value
        self.path2 = self.get_parameter("trajectory2").get_parameter_value().string_value

        # initialize and load the trajectory
        self.trajectory1 = LineTrajectory(self, "/loaded_trajectory")
        self.trajectory2 = LineTrajectory(self, "/loaded_trajectory")
        self.get_logger().info(f"Loading from {self.path1}")
        self.trajectory1.load(self.path1)
        self.trajectory2.load(self.path2)

        self.pub_topic = "/trajectory/current"
        self.traj_pub = self.create_publisher(PoseArray, self.pub_topic, 1)

        # need to wait a short period of time before publishing the first message
        time.sleep(0.5)

        # visualize the loaded trajectory
        self.trajectory1.publish_viz()
        self.trajectory2.publish_viz()

        # send the trajectory
        self.publish_trajectory()

    def publish_trajectory(self):
        print("Publishing trajectory to:", self.pub_topic)
        self.traj_pub.publish(self.trajectory1.toPoseArray())
        self.traj_pub.publish(self.trajectory2.toPoseArray())


def main(args=None):
    rclpy.init(args=args)
    load_trajectory = LoadTrajectory()
    rclpy.spin(load_trajectory)
    load_trajectory.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
