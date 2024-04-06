import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        rng = np.random.default_rng()
        self.noise = rng.normal()

        ####################################
    
    def rot(self, angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        return np.array([[cos_theta, -sin_theta, 0],
                         [sin_theta,  cos_theta, 0 ],
                         [0, 0, 1]])


    def update_pose(self, particle, odom):
        # Current particle position
        [x, y, t] = particle
        
        xk = particle + self.rot(t) @ np.array(odom).T

        [xp, yp, tp] = xk

        # Adding noise
        #xp += self.noise
        #yp += self.noise
        #tp += self.noise

        return [xp, yp, tp]

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

        new_pose = [self.update_pose(particle, odom) for particle in particles]

        return np.array(new_pose)

        ####################################
