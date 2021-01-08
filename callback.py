import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose

from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def globalCallback(data):
    message.posx = round(data.position.x, 8)
    message.posy = round(data.position.y, 8)
    message.theta = yawFromQuaternion(data.orientation)

def imuCallback(data):
    message.linear_acceleration = data.linear_acceleration.x
    ang_vel = data.angular_velocity.z

def laserCallback(data):
    message.range = data.ranges
    message.laser_data = data

def goalCallback(data):
    message.goal_pos.x = data.position.x
    message.goal_pos.y = data.position.y

def cameraInfoCallback(data):
    message.camerInfo = data

def imageCallback(data):
    bridge = CvBridge()
    message.image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')