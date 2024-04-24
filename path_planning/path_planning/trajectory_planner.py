import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
from .utils import log
from geometry_msgs.msg import PointStamped
import tf_transformations
from std_msgs.msg import Int32

import math
import numpy as np

import json

class PathPlan(Node):
    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('from_start_traj', "default")
        self.declare_parameter('to_start_traj', "default")

        # idk if this is how we want to keep track of which traj on but also I figure we can pull out from the dict if we don't already for the other code
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.from_start_traj = self.get_parameter('from_start_traj').get_parameter_value().string_value
        self.to_start_traj = self.get_parameter('to_start_traj').get_parameter_value().string_value

        self.current_traj = self.parse_traj_file(self.from_start_traj) 
        self.opp_traj = self.parse_traj_file(self.to_start_traj) 
        self.clicked_points = []

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)
        
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.stop_pub = self.create_publisher(
            Int32,
            "/stop_pub",
            10
        )

        self.clicked_indices = self.create_publisher(
            Int32,
            "/clicked_indices",
            10
        )
    
    def parse_traj_file(self, fname):
        # Open the file and load JSON data
        with open(fname) as f:
            data = json.load(f)

        # Extract the "points" data
        points_data = data["points"]

        # Create a list of (x, y) points
        return [(point["x"], point["y"]) for point in points_data]


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

    def clicked_cb(self, msg):
        if len(self.clicked_points) < 3:
            self.clicked_points.append([msg.point.x, msg.point.y])
        
        if len(self.clicked_points) == 3:
            self.main_stuff_happening(self.clicked_points)

    def generate_interpolated_traj(self, trajectory, distance):
        """
        Based on our trajectories, create new ones with interpolated points
        """
        interpolated_traj = []

        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            
            # Calculate the total distance between current pair of points
            total_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Determine the number of segments needed
            num_segments = int(total_distance / distance)
            
            # Generate linear interpolations for x and y coordinates
            x_values = np.linspace(p1[0], p2[0], num_segments + 1)
            y_values = np.linspace(p1[1], p2[1], num_segments + 1)
            
            # Append each generated point to the list, avoiding duplication of the endpoint
            points = np.column_stack((x_values, y_values))

            interpolated_traj.extend(points[:-1].tolist())

        # Add in the final point that was excluded
        interpolated_traj.append(trajectory[-1])

        return interpolated_traj

    def find_closest_point(self, point, trajectory):
        """
        Finds index of the closest point in the trajectory and returns it with its distance to point
        """
        min_distance = float("inf")
        closest_idx = 0
        x1,y1 = point

        for idx, traj_point in enumerate(trajectory):
            x2, y2 = traj_point
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if distance < min_distance:
                closest_idx = idx
                min_distance = distance

        return (closest_idx, min_distance)

    def find_encasing_indices(self, point, trajectory):
        """
        Find which indices in the trajectory the point would come between
        """
        temp_trajectory = trajectory.copy()
                
        #Finding two closest points in original trajectory
        closest_idx, _ = self.find_closest_point(point, temp_trajectory)
        temp_trajectory[closest_idx] = (1000, 1000)   #Arbitrarily large number to remove from consideration
        closest_idx_2, _ = self.find_closest_point(point, temp_trajectory)

        # Returning in order of smaller index
        return sorted([closest_idx, closest_idx_2])

    def transform_point(self, coord, res, offsets, pixel_to_world=True):
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

    def convert_and_publish(self, data):
        msg = Int32()
        msg.data = data
        self.clicked_indices.publish(msg)
        
    def main_stuff_happening(self, goals):
        current_index = 0
        planned_trajectory = [self.current_traj[0]]

        interp_dis = 1 #To play with and decide
        interpolated_current = self.generate_interpolated_traj(self.current_traj, interp_dis)
        interpolated_opp = self.generate_interpolated_traj(self.opp_traj, interp_dis)

        goals.append(self.opp_traj[-1])

        for goal in goals:

            closest_in_current = self.find_closest_point(goal, interpolated_current)
            closest_in_opp = self.find_closest_point(goal, interpolated_opp)

            # Is on this side of the road and ahead (no turns necessary)
            if (closest_in_current[1] <= closest_in_opp[1]) and (closest_in_current[0] >= current_index):
                before, after = self.find_encasing_indices(goal, self.current_traj)

                planned_trajectory.extend(self.current_traj[current_index + 1 : before + 1])
                self.convert_and_publish(len(planned_trajectory))
                planned_trajectory.append(goal) #Goal with STOP marker
                #Iffy, I want it to get back on next point to resume from track but could have issues
                planned_trajectory.append(self.current_traj[after])
                current_index = after


            #Is on other side of the road or behind (turn(s) necessary)
            else:
                current_point = self.current_traj[current_index]
                #If I were on opposite side now, where would I be relative to point
                robot_idx, turn_pt = self.find_encasing_indices(current_point, self.opp_traj)
                goal_idx, _ = self.find_encasing_indices(goal, self.opp_traj)

                same_side_behind = (closest_in_current[1] <= closest_in_opp[1])
                opposite_ahead = (robot_idx < goal_idx)

                #If same-side behind or opposite ahead, make a turn first
                if same_side_behind or opposite_ahead:
                    #Choosing the later point that we are between to uturn chase
                    planned_trajectory.append(self.opp_traj[turn_pt])

                    ## SWITCHED DIRECTIONS ##
                    current_index = turn_pt
                    self.current_traj, self.opp_traj = self.opp_traj, self.current_traj
                    interpolated_current, interpolated_opp = interpolated_opp, interpolated_current

                # Straight away
                before, after = self.find_encasing_indices(goal, self.current_traj)
                planned_trajectory.extend(self.current_traj[current_index + 1 : before + 1])

                if opposite_ahead:
                    self.convert_and_publish(len(planned_trajectory))
                    planned_trajectory.extend([goal, self.current_traj[after]])
                    current_index = after

                # Still needs one final turn
                else:
                    self.convert_and_publish(len(planned_trajectory)+1)
                    planned_trajectory.extend([self.current_traj[after], goal])
                    #Taking it to next point after goal
                    after_goal = self.find_encasing_indices(goal, self.opp_traj)[1]
                    planned_trajectory.append(self.opp_traj[after_goal])
                    current_index = after_goal

                    ## SWITCHED DIRECTIONS ##
                    current_index = after_goal
                    self.current_traj, self.opp_traj = self.opp_traj, self.current_traj
                    interpolated_current, interpolated_opp = interpolated_opp, interpolated_current

        #Publishing stop sign indices
        # stop_sign = (-9.20293807, 26.30563735)
        # stop_indices = [idx for idx,elm in enumerate(planned_trajectory) if elm == stop_sign]

        # for idx in stop_indices:
        #     msg = Int32()
        #     msg.data = idx
        #     self.stop_pub.publish(msg)

        #Publishing Path
        for point in planned_trajectory:
            self.trajectory.addPoint(point) # adding the points to the trajectory
        
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()




#Unaddressed edge cases:s
#   Goal point is not between two trajectory points (aka at very end or sum)
#   Distance from goal to trajectory points is too small 

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
