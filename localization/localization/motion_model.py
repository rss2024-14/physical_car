import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.prev_odom = [0, 0, 0]

        ####################################

    def sample_normal_distribution(self, b):
        x = np.random.uniform(low=-1, high=1, size=12)
        x_sum = np.sum(x)
        return (b/6.0) * x_sum
    
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

        # Alpha values
        a1 = 0.74
        a2 = 0.07
        a3 = 0.07
        a4 = 0.12

        rot1 = np.arctan2(dy_prime-dy, dx_prime-dx) - dtheta
        trans = np.sqrt((dx-dx_prime)**2 + (dy-dy_prime)**2)
        rot2 = dtheta_prime - dtheta - rot1

        rot1_prime = rot1 - self.sample_normal_distribution(a1*rot1 + a2*trans)
        trans_prime = trans - self.sample_normal_distribution(a3*trans + a4*(rot1+rot2))
        rot2_prime = rot2 - self.sample_normal_distribution(a1*rot2 + a2*trans)

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

        return new_pose

        ####################################
