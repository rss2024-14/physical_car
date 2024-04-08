import numpy as np
import math
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):

        print("STARTING INITIALIZATION")
        node.get_logger().info("STARTING INITIALIZATION")
        self.logger = node.get_logger()

        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = .74
        self.alpha_short = .07
        self.alpha_max = .07
        self.alpha_rand = .12

        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################
        self.resolution = 0.05

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)
        

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        ### Calculate all the phits, put them in the table, normalize across columns, then add the other p values
        ### For each z
        for row in range(self.table_width):
            ### For each d of table, calculate phit
            for column in range(self.table_width):

                ## Calculate phit

                phit = 1.0/(math.sqrt(2.0 * math.pi * self.sigma_hit**2)) * np.exp(-(row-column)**2/(2*self.sigma_hit**2))

                self.sensor_model_table[row][column] = phit

            ### Normalize columns to add up to 1


        column_sums = np.sum(self.sensor_model_table, axis=0)
        self.sensor_model_table /= column_sums 
        self.sensor_model_table *= self.alpha_hit 
        
        for row in range(self.table_width):
            # ### For each d of table, add the remaining probability values
            for column in range(self.table_width):

                ## Calculate pshort
                if (column == 0):
                    pshort = 0
                elif (column != 0 and row <= column):
                    pshort = 2/(column) * (1-(row/column)) 
                else:
                    pshort = 0

                ### Calculate pmax
                if row == self.table_width-1:
                    pmax = 1
                else:
                    pmax = 0

                ## Calculate prand
                prand = 1/(self.table_width-1)

                p = self.alpha_short * pshort + self.alpha_max * pmax + self.alpha_rand * prand

                self.sensor_model_table[row][column] += p
        
        self.sensor_model_table /= self.sensor_model_table.sum(axis=0, keepdims=True)
        
    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.
        Perform ray tracing from all the particles.
        PRoduces a matrix of size N x num_beams_per_particle

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return
        
        positions = self.scan_sim.scan(particles)
        
        observation /= (self.resolution * self.lidar_scale_to_map_scale)
        observation = np.clip(observation, 0, self.table_width - 1).astype(int)

        scans = positions / (self.resolution * self.lidar_scale_to_map_scale)
        scans = np.clip(scans, 0, self.table_width - 1).astype(int)
        
        probs = np.prod(self.sensor_model_table[observation, scans], axis=1)   

        return probs / np.sum(probs)

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
