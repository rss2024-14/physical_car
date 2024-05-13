import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from wall_follower.visualization_tools import VisualizationTools  # Assuming VisualizationTools is a class in visualization_tools module
from ackermann_msgs.msg import AckermannDriveStamped
import tf_transformations

NUM_PTS_WALL = 15

class ScanReader(Node):
    SCAN_TOPIC = "/scan"
    WALL_TOPIC = "/wall"
    LOOKAHEAD_TOPIC = "/lookahead"
    DRIVE_TOPIC = "/drive"
    ODOM_TOPIC = "/odom"
    L1 = 1
    L = .3 # car length
    heading = np.pi/2

    Kp = 3.5
    Kd = 1
    prev_error = 0
    dedt = 0
    ref = .9

    def __init__(self):
        super().__init__('scan_reader')
        
        self.subscription = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.listener_callback,
            10)
        
        self.subscription_odom = self.create_subscription(
            Odometry,
            self.ODOM_TOPIC,
            self.listener_callback2,
            10)
        
        self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 1)
        self.lookahead_pub = self.create_publisher(Marker, self.LOOKAHEAD_TOPIC, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 1)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        (angle_min, angle_inc) = (msg.angle_min, msg.angle_increment)
        msg_np = np.array(msg.ranges)

        closest_dists = np.sort(msg_np)[:NUM_PTS_WALL]
        closest_angles = angle_min + angle_inc * np.argsort(msg_np)[:NUM_PTS_WALL]

        closest_xs = closest_dists * np.cos(closest_angles)
        closest_ys = closest_dists * np.sin(closest_angles)
        
        lin_reg = np.polyfit(closest_xs, closest_ys, 1)
        m = lin_reg[0]
        b = lin_reg[1]
        sample_pts = np.arange(-2, 2, .1)
        y = m * sample_pts + b

        # lookahead circle
        x_circ_pts = np.linspace(-self.L1, self.L1, 50)
        x_circ_pts_twice = np.tile(x_circ_pts,2)
        y_circ = np.concatenate( (np.sqrt(self.L1**2 - x_circ_pts**2), -np.sqrt(self.L1**2 - x_circ_pts**2)) )
        # self.get_logger().info("%s closest_xs, %s closest_ys %s closest_angles %s %s m b" % (closest_xs, closest_ys, closest_angles, m, b))
        VisualizationTools.plot_line(sample_pts, y, self.line_pub, color=(0.,0.796,1.), frame="/laser")
        VisualizationTools.plot_line(x_circ_pts_twice, y_circ, self.lookahead_pub, color=(1.,1.,1.), frame="/laser")

        perp_dist = calc_perp_dist(m,b)

        if (self.L1 <= perp_dist):
            angle_robot_to_intersect = np.pi/2
        else:
            angle_robot_to_intersect = np.pi/2 - np.arccos(perp_dist / self.L1)
        
        if (b < 0):
            angle_robot_to_intersect *= -1

        eta = -self.heading + angle_robot_to_intersect
        delta = np.arctan(2 * self.L * np.sin(eta) / self.L1)

        # PID
        error = -(self.ref - np.abs(perp_dist))
        dedt = error - self.prev_error
        u = self.Kp * error + self.Kd * dedt
        
        delta += u

        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "/base_link"
        drive_cmd.drive.speed = 1.9
        drive_cmd.drive.steering_angle = delta
        # self.get_logger().info("%s perp_dist\n%s eta\n%s delta" % (perp_dist, eta * 180/np.pi, delta*180/np.pi))
        self.drive_pub.publish(drive_cmd)

        """
        [-0.19113478  0.0665778  -0.02168921 -0.11194623  0.02342293  0.11440338
        0.24860189  0.15968609 -0.15858264  0.2060234 ] closest_xs
        
        [0.88247667 0.92103889 0.94319264 0.94313597 0.95256843 0.95089476
        0.92533793 0.94550112 0.94816235 0.94394799] closest_ys
 
        [1.78409083 1.49863629 1.5937878  1.68893932 1.54621205 1.45106053
        1.30833326 1.40348478 1.73651507 1.35590902] closest_angles
        
        0.04294446594676407 0.9341854802723726 m b
        """
    
    def listener_callback2(self, msg):
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        mat = tf_transformations.quaternion_matrix(q)
        z_rot = np.arctan2(mat[1, 0], mat[0, 0]) #* 180/np.pi
        # self.get_logger().info("%s z rot" % z_rot)
        self.heading = z_rot


def calc_perp_dist(m,b):
    return np.abs(b) / np.sqrt(m**2 + 1)

def main(args=None):

    rclpy.init(args=args)
    scan_reader = ScanReader()
    rclpy.spin(scan_reader)
    scan_reader.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# 1. fit line to collected points
# 2. choose lookahead distance with intersection with created line
# 3. Add PID to adjust speed/steering angle to get desired dist from wall