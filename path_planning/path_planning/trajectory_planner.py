import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

from skimage.morphology import dilation

import matplotlib.pyplot as plt
import tf_transformations
import numpy as np
import random
import math


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        # Set map 2d, initial pose, and goal pose to None (awaiting map 2d, inital pose, and goal pose)
        self.map_2d = None
        self.initial_pose = None
        self.goal_pose = None

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, map_1d):
        # The /map topic only publishes once, so this function will only be ran once
        self.get_logger().info('Received Map')

        self.map_height = map_1d.info.height
        self.map_width = map_1d.info.width
        self.map_res = map_1d.info.resolution
        
        # Getting the origin info
        orientation_q = map_1d.info.origin.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        euler = tf_transformations.euler_from_quaternion(quaternion)
        orientation = euler[2]
        self.map_origin = (map_1d.info.origin.position.x, map_1d.info.origin.position.y, orientation)

        # Reshaping occupancy grid
        self.get_logger().info("Map data %s" % ( set(map_1d.data), ))
        self.map_2d = np.array(map_1d.data).reshape((self.map_height, self.map_width)).T
        structure_elt = np.ones((9,9))
        dilated_map = dilation(self.map_2d == 100, structure_elt)
        self.map_2d[dilated_map] = 100

        self.map_2d[self.map_2d == -1] = -100
        plt.imshow(self.map_2d, cmap='hot', interpolation='nearest')
        plt.show()


    def pose_cb(self, start):
        self.get_logger().info('Received Initial Position')

        x = start.pose.pose.position.x
        y = start.pose.pose.position.y

        self.initial_pose = transform_point((x,y), self.map_res, self.map_origin, pixel_to_world=False) 
        self.initial_pose = (int(self.initial_pose[0]), int(self.initial_pose[1]))

    def goal_cb(self, end):
        self.get_logger().info('Received Goal Position')

        x = end.pose.position.x
        y = end.pose.position.y

        self.goal_pose = transform_point((x,y), self.map_res, self.map_origin, pixel_to_world=False)

        # Find path from initial pose to goal pose once goal pose has been set
        self.plan_path(self.initial_pose, self.goal_pose, self.map_2d)

    def plan_path(self, start_point, end_point, map):
        self.get_logger().info("Finding Trajectory")
        
        start_node = Node(start_point, parent=None) # make the start point a node

        nodes = set() # initializing empty set for nodes
        nodes.add(start_node)

        counter = 0 # keeps track of iterations
        lim = 5000 # number of iterations algorithm should run for
        step = 5.0 # length of the step taken for next_point
        error = 10.0 # valid error around goal pose
        goal_probability = 0.4 # rate at which the goal point is picked

        while counter < lim:
            # Randomly generate a point in map
            if np.random.rand() < goal_probability:
                random_pt = self.goal_pose
            else:
                random_pt = (random.randint(0, self.map_width-1), random.randint(0, self.map_height-1))
            # self.get_logger().info("x_rand %s // y_rand %s" % random_pt)

            nearest_node = find_nearest_node(nodes, random_pt)
            
            # self.get_logger().info("nearest_node %s " % (nearest_node.value,))

            if random_pt == nearest_node.value:
                continue

            next_point = find_next_point(nearest_node.value, random_pt, step)
            # self.get_logger().info("next_point " + str(next_point))

            # Checking if next point is valid(occupied space = 100, unoccupied space = 0, unknown space = -1 )
            # self.get_logger().info("nodes %s" % ( len(nodes), ))
            if (map[next_point[0]][next_point[1]] == 100) or (map[next_point[0]][next_point[1]] == -1): 
                # self.get_logger().info("skipping " + str(map[next_point[0]][next_point[1]]))
                continue

            next_node = Node(next_point, parent=nearest_node)

            nodes.add(next_node)

            # Checking if goal pose has been reached
            if (end_point[0]-error <= next_point[0] <= end_point[0]+error) and (end_point[1]-error <= next_point[1] <= end_point[1]+error):
                path = next_node.path_from_root() # finding the path from initial to goal
                self.get_logger().info("path %s" % ( path, ))
                for point in path:
                    self.trajectory.addPoint(transform_point(point, self.map_res, self.map_origin, pixel_to_world=True)) # adding the points to the trajectory
                
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()

                break
            
            counter += 1


class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent

    def path_from_root(self):
        """ Return the sequence of nodes from the root to this node """
        if self.parent is None:
            return [self.value]
        else:
            path_to_parent = self.parent.path_from_root()
            path_to_parent.append(self.value)
            return path_to_parent


def find_nearest_node(nodes, new_point):
    """
    points: a set of nodes
    new_point: (x, y) of a new point
    """
    nearest_vertex = None
    min_distance = float('inf')  # Start with infinity as the minimum distance
    
    for node in nodes:
        coord = node.value
        # Calculate the Euclidean distance
        distance = math.sqrt((coord[0] - new_point[0])**2 + (coord[1] - new_point[1])**2)
        
        # Update the closest point if a new minimum distance is found
        if distance < min_distance:
            min_distance = distance
            nearest_vertex = node

    return nearest_vertex


def find_next_point(point_a, point_b, length):
    # Calculate the displacement vector
    d = [b - a for a, b in zip(point_a, point_b)]
    
    # Calculate the magnitude of the displacement vector
    magnitude = math.sqrt(sum(comp ** 2 for comp in d))
    
    # Normalize the vector to get the unit vector, then scale it by the desired length
    if magnitude == 0:
        raise ValueError("The points are identical; cannot determine a unique direction vector.")
    scaled_vector = [(comp / magnitude) * length for comp in d]

    next_point = [int(a + b) for a, b in zip(point_a, scaled_vector)]
    
    return tuple(next_point)


def transform_point(coord, res, offsets, pixel_to_world=True):
    """
    coord  : coordinates of the original point (x, y)
    res: scaling factors along the x and y axes
    offsets : (tx, ty, theta)

    Returns tuple of point (x', y')
    """
    # Create the scaling matrix
    S = np.array([[res, 0,  0],
                  [0,  res, 0],
                  [0,  0,  1]])

    # Create the rotation matrix
    R = np.array([[np.cos(offsets[2]), -np.sin(offsets[2]), 0],
                  [np.sin(offsets[2]),  np.cos(offsets[2]), 0],
                  [0,             0,              1]])
    
    # Create the translation matrix
    T = np.array([[1, 0, offsets[0]],
                  [0, 1, offsets[1]],
                  [0, 0,  1]])

    # Create the point vector
    p = np.array([[coord[0]],
                  [coord[1]],
                  [1]])

    # Apply the transformations: first scale, then rotate, then translate
    if pixel_to_world:
        p_prime = T @ (R @ (S @ p))
    else:
        S_inv = np.linalg.inv(S)
        R_inv = np.linalg.inv(R)
        T_inv = np.linalg.inv(T)
        p_prime = S_inv @ (R_inv @ (T_inv @ p))

    # Return the transformed point (x', y')
    return tuple([p_prime[0, 0], p_prime[1, 0]])


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
