import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.prev_odom = [0, 0, 0]

        rng = np.random.default_rng()
        self.noise = rng.normal()

        ####################################
    
    def update_pose(self, particle, odom_prime, odom):
        # Current particle position
        x = particle[0]
        y = particle[1]
        theta = particle[2]
        
        # Current odom readings
        dx_prime = odom_prime[0]
        dy_prime = odom_prime[1]
        dtheta_prime = odom_prime[2]

        # Previous odom readings
        dx = odom[0]
        dy = odom[1]
        dtheta = odom[2]

        rot1 = np.arctan2(dy_prime-dy, dx_prime-dx) - dtheta
        trans = np.sqrt((dx-dx_prime)**2 + (dy-dy_prime)**2)
        rot2 = dtheta_prime - dtheta - rot1

        rot1_prime = rot1 + self.noise
        trans_prime = trans + self.noise
        rot2_prime = rot2 + self.noise

        x_prime = x + trans_prime*np.cos(theta + rot1_prime)
        y_prime = y + trans_prime*np.sin(theta + rot1_prime)
        theta_prime = theta + rot1_prime + rot2_prime

        return [x_prime, y_prime, theta_prime]

    def evaluate(self, particles, odom):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################

        new_pose = [self.update_pose(particle, odom, self.prev_odom) for particle in particles]

        self.prev_odom = odom

        return np.array(new_pose)

        ####################################
