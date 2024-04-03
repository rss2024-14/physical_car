import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.prev_odom = [0, 0, 0]

        rng = np.random.default_rng()
        self.noise = rng.normal() #alphas 

        ####################################
    
    def rot(angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        return np.array([[cos_theta, -sin_theta, 0],
                         [sin_theta,  cos_theta, 0 ]
                         [0, 0, 1]])


    def update_pose(self, particle, odom_prime, odom):
        # Current particle position
        [x, y, t] = particle
        xk = particle + self.rot(-t) @ odom.T
        
        # Current odom readings
        xp = xk[0] 
        yp = xk[1]
        tp = xk[2]

        rot1 = np.arctan2(yp-y, xp-x) - t
        trans = np.sqrt( (x-xp)**2 + (y-yp)**2 )
        rot2 = tp - t - rot1

        rot1_hat = rot1 - self.noise()
        trans_hat = trans - self.noise()
        rot2_hat = rot2 - self.noise()

        x_prime = x + trans_hat*np.cos(t + rot1_hat)
        y_prime = y + trans_hat*np.sin(t + rot1_hat)
        theta_prime = t + rot1_hat + rot2_hat

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
