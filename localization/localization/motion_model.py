import numpy as np

class MotionModel:

    def __init__(self, node, deterministic = False):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.logger = node.get_logger()
        self.deterministic = deterministic

        # Random Number Generator
        rng = np.random.default_rng()

        # Normal Distribution
        self.noise = rng.normal()

        # Uniform Distribution
        #self.noise = rng.uniform()

        # Triangular Distribution
        # self.noise = rng.triangular()

        # Poisson Distribution
        # self.noise = rng.poisson()

        ####################################
    
    def rot(self, angles):
        cos_theta = np.cos(angles)
        sin_theta = np.sin(angles)
        
        return np.array([[cos_theta, -sin_theta, 0],
                         [sin_theta,  cos_theta, 0 ],
                         [0, 0, 1]])


    def update_pose(self, particle, odom):
        # Current particle position

        [x, y, t] = particle
        
        xk = particle + self.rot(t) @ np.array(odom).T

        [xp, yp, tp] = xk

        if not self.deterministic:
            xp += np.random.normal(scale=.15)#self.noise
            yp += np.random.normal(scale=.15)#self.noise
            tp += np.random.normal(scale=.15)#self.noise

        return [xp, yp, tp]

        # angles = particles[:,2]
        # ct = np.cos(angles)
        # st = np.sin(angles)
                
        # motion = np.array(
        #     [ct * odom[0] - st * odom[1],
        #      st * odom[0] + ct * odom[1],
        #      np.full(np.shape(ct), odom[2]) ])
        
        # return particles + motion + np.random.normal(scale=0.15, size=np.shape(motion)) if self.deterministic else 0

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

        # return self.update_pose(particles, odom)
        new_pose = [self.update_pose(particle, odom) for particle in particles]
        return np.array(new_pose)

        ####################################
