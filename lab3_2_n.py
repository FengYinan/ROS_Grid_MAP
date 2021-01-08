#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import random
import math
import numpy as np

class message():
    def __init__(self):
        self.velx = 0.0
        self.posx = 0.0
        self.posy = 0.0
        self.theta = 0.0

        self.linear_acceleration = 0.0
        self.range = [0 for _ in range(720)]
        self.laser_data = None


def odomCallback(data):
    message.velx = data.twist.twist.linear.x
    message.posx = round(data.pose.pose.position.x, 8)
    message.posy = round(data.pose.pose.position.y, 8)
    message.theta = yawFromQuaternion(data.pose.pose.orientation)


def imuCallback(data):
    message.linear_acceleration = data.linear_acceleration.x
    ang_vel = data.angular_velocity.z


def laserCallback(data):
    message.range = data.ranges
    message.laser_data = data


def yawFromQuaternion(orientation):
    return math.atan2((2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)),
                      (1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)))


def l2_norm(x, y, goalX, goalY):
    return math.sqrt(pow((goalX - x), 2) + pow((goalY - y), 2))


class potential_field():
    def __init__(self):
        self.warning = False
        self. goal_speed = 1.

    def potential_field_force(self, x, y, theta, goalx, goaly, laser_data):
        af = self.AttractivePotentialFiled(goalx, goaly, x, y)

        rf = self.RepulsivePotentialField2(laser_data)

        dx = af[0] + rf[0]
        dy = af[1] + rf[1]

        vel = math.sqrt(dx ** 2 + dy ** 2)

        if np.fabs(vel) > self.goal_speed:
            vel = np.sign(vel) * self.goal_speed
        if vel < 0.5:
            rospy.loginfo("Velocity limited")
            vel = 0.5

        error = np.arctan2(dy, dx) - theta
        rot = math.atan2(math.sin(error), math.cos(error)) * 2

        if np.isnan(vel) or np.isnan(rot):
            rospy.loginfo("NAN: vel: %2f, rot: %2f" % (vel, rot))
        if self.warning:
            vel = 0.
            rot = math.pi/4
            rospy.loginfo("WARNING - OBSTACLE DETECTED")
            self.warning = False

        rospy.loginfo("a0: %.4f, a1: %.4f, r0: %.4f, r1: %.4f, v: %.4f, s: %.4f", af[0], af[1], rf[0], rf[1], vel, rot)

        return vel, rot


    def AttractivePotentialFiled(self, goalx, goaly, x, y):
        epsilon = 1./4

        attractivex = epsilon * (goalx - x)
        attractivey = epsilon * (goaly - y)

        return [attractivex, attractivey]

    def RepulsivePotentialField2(self, laser_data):
        safe_distance = 2
        goal_speed = 0.8

        angle = laser_data.angle_min
        resolution = laser_data.angle_increment

        dx = 0.0
        dy = 0.0

        eta = 1.3

        for l in laser_data.ranges:

            if l < 0.2 or angle < -math.pi/2. or angle > math.pi/2.:
                distance = laser_data.range_max
            else:
                distance = l

            if distance <= 0.7:
                self.warning = True

            if distance <= safe_distance:
                dx += eta * ((safe_distance - distance) * np.cos(angle- math.pi/2.)) * (1+ 1/ distance)
                dy += eta * ((safe_distance - distance) * np.sin(angle- math.pi/2.)) * (1+ 1/ distance)
                assert (np.isfinite(dx) & np.isfinite(dy))

            angle += resolution

        return [dx, dy]


def main():
    random.seed()
    vel_pub = rospy.Publisher('/jackal0/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
    rospy.init_node('jackal_turn', anonymous=True)
    rospy.Subscriber("/jackal0/odometry/local_filtered", Odometry, odomCallback)
    rospy.Subscriber("/jackal0/imu/data", Imu, imuCallback)
    rospy.Subscriber("/jackal0/front/scan", LaserScan, laserCallback)
    rate = rospy.Rate(10)         # rate = 100Hz
    dt = 0.1                      # follow the rate

    global message, controlor
    message = message()

    vel = Twist()
    pf = potential_field()

    goalx = 0.
    goaly = 18.

    rate.sleep()
    previous_range = [message.range, min(message.range)]
    rate.sleep()

    while l2_norm(message.posx, message.posy, goalx, goaly) >= 0.2:
        
        linearx, steering = pf.potential_field_force(message.posx, message.posy, message.theta, goalx, goaly, message.laser_data)

        #rospy.loginfo("a0: %.4f, a1: %.4f, r0: %.4f, r1: %.4f, v: %.4f, s: %.4f, d: %.4f, t: %.4f", attractive[0], attractive[1], repulsive[0], repulsive[1], linearx, steering, distance, angle)

        vel.linear.x = linearx
        vel.angular.z = steering
        vel_pub.publish(vel)
        rate.sleep()

    rospy.loginfo("x: %.4f, y: %.4f, d: %.4f", message.posx, message.posy, l2_norm(message.posx, message.posy, goalx, goaly))
        
        
if __name__ == "__main__":
    main()



