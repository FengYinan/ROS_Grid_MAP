#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import CameraInfo, Image
from PIL import Image

from map_class import maps
from helper_func import *

class position():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

class message():
    def __init__(self):
        self.velx = 0.0
        self.posx = 0.0
        self.posy = 0.0
        self.theta = 0.0

        self.linear_acceleration = 0.0
        self.range = [0 for _ in range(720)]
        self.laser_data = None

        self.goal_pos = position()

        self.camerInfo = None
        self.image = None


class potential_field():
    def __init__(self):
        self.warning = False
        self. goal_speed = 1.5

    def potential_field_force(self, x, y, goalx, goaly, laser_data):
        af = self.AttractivePotentialFiled(goalx, goaly, x, y)

        rf = self.RepulsivePotentialField2(laser_data)

        dx = af[0] + rf[0]
        dy = af[1] + rf[1]

        dx += self.goal_speed
        vel = math.sqrt(dx ** 2 + dy ** 2)

        if np.fabs(vel) > self.goal_speed:
            vel = np.sign(vel) * self.goal_speed
        if vel < 0.1:
            rospy.loginfo("Velocity limited")
            vel = 0.1

        if dx != 0.0:
            rot = (np.arctan2(dy, dx))
        else:
            rospy.loginfo("OBA - DX = 0")
            rot = ((-1 * np.sign(dy)) * (np.pi / 2.0)) / 2.0

        if np.isnan(vel) or np.isnan(rot):
            rospy.loginfo("NAN: vel: %2f, rot: %2f" % (vel, rot))
        if self.warning:
            vel = np.sign(vel) * 0.1
            rot = np.sign(rot) * 0.25
            rospy.loginfo("WARNING - OBSTACLE DETECTED")
            self.warning = False

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
            if l > safe_distance:
                continue

            if l < 0.2 or angle < 0 or angle > math.pi:
                distance = laser_data.range_max
            else:
                distance = l

            if l <= 0.7:
                self.warning = True

            dx += -eta * ((safe_distance - distance) * np.cos(angle - math.pi/2.))
            dy += -eta * ((safe_distance - distance) * np.sin(angle)- math.pi/2.)
            assert (np.isfinite(dx) & np.isfinite(dy))

            angle += resolution

        return [dx, dy]

class pid_controller():
    def __init__(self,ks,kp,ki,kd):
        self.dt = 0.1

        self.value = 0.0

        self.integral = 0.0
        self.error = 0.0

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ks = ks

    def pid(self,error):
        self.integral = self.integral + error*self.dt
        derivative = (error-self.error)/self.dt
        self.error = error
        self.value = self.ks*self.value + self.kp*error + self.ki*self.integral + self.kd*derivative

        return self.value

class race():
    def __init__(self):
        self.v_pid = pid_controller(0.99,0.15,0.002,0.01)
        self.s_pid = pid_controller(0.0,2.7,0.01,0.03)
        self.kr = 0.2
        self.eps = 0.3
        self.dis_min = 3
        self.vel_max = 2.0
        self.vel_min = 1.0

        self.goal = position()
        self.section = [0 for _ in range(9)]

    def update_goal(self, goal):
        self.goal = goal

    def reached(self, robot_pos):
        if dis2d(self.goal, robot_pos) > self.dis_min:
            return 1
        elif dis2d(self.goal, robot_pos) > self.eps:
            return 2
        else:
            return 0

    def going(self, robot_pos, theta, laser_data):
        test = float("inf")
        linearx = self.vel_max
        steering = 0.0 
        turning = 0.0
        thetad = math.atan2((self.goal.y - robot_pos.y), (self.goal.x - robot_pos.x))
        #rospy.loginfo('gx = %.4f, gy = %.4f, rx = %.4f, ry = %.4f', self.goal.x, self.goal.y, robot_pos.x, robot_pos.y)
        #rospy.loginfo('real_gy = %.4f, real_gx = %.4f', message.goal_pos.x, message.goal_pos.y)
        error_min = 10
        error_num = 0
        rospy.loginfo('thetad = %.4f', thetad)
        for i in range(9):
            section_center = self.ranges_to_angle(i*80+40, theta, laser_data)
            rospy.loginfo('section_center %d = %.4f', i, section_center)
            error = section_center - thetad
            if error > 2*math.pi:
                error = 2*math.pi - section_center + thetad
            elif error < -2*math.pi:
                error = 2*math.pi + section_center - thetad
            #error = math.atan2(math.sin(error), math.cos(error))
            #rospy.loginfo('error %d = %.4f', i, error)
            if abs(error) < error_min:
                error_min = abs(error)
                error_num = i
        
        if error_num <= 1:
            turning -= 4
        elif error_num <= 3:
            turning -= 2
        elif error_num >= 7:
            turning += 4
        elif error_num >= 5:
            turning += 2
        rospy.loginfo('section:%d, turning = %.d', error_num, turning)    
        for i in range(9):
            summ = 0
            for j in range(80):
                if laser_data.ranges[i*80+j] == test:
                    summ += 30
                else:
                    summ += laser_data.ranges[i*80+j]
            summ /= 80
            self.section[i] = summ

        #rospy.loginfo('s0-%.4f s1-%.4f s2-%.4f s3-%.4f s4-%.4f s5-%.4f s6-%.4f s7-%.4f s8-%.4f', self.section[0], self.section[1], self.section[2], self.section[3], self.section[4], self.section[5], self.section[6], self.section[7], self.section[8])
        for i in range(9):
            if self.section[i] < 1.5:
                linearx = self.vel_min
                if i < 4:
                    turning += i*2+1
                elif i > 4:
                    turning -= ((8-i)*2+1)
                else:
                    if error_num < 4:
                        turning -= 10
                    else:
                        turning += 10
        rospy.loginfo('t=%.4f', turning)
        steering = turning * self.kr
        

        return linearx, steering

    def goto(self, robot_pos, theta):
        error = dis2d(self.goal, robot_pos)
        linearx = self.v_pid.pid(error)
        linearx = min(self.vel_max, linearx)

        error = math.atan2((self.goal.y - robot_pos.y), (self.goal.x - robot_pos.x)) - theta
        error = math.atan2(math.sin(error), math.cos(error))
        steering = self.s_pid.pid(error)

        #rospy.loginfo('vel=%.4f', linearx)
        rospy.loginfo('gx = %.4f, gy = %.4f, rx = %.4f, ry = %.4f', self.goal.x, self.goal.y, robot_pos.x, robot_pos.y)

        return linearx, steering

    def ranges_to_angle(self, range_id, theta, laser_data):
        #rospy.loginfo('a_m = %.4f, a_i = %.4f', laser_data.angle_min, laser_data.angle_increment)
        return theta + (laser_data.angle_min + (laser_data.angle_max-laser_data.angle_min)*range_id/720)

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

def main():
    global message
    message = message()

    random.seed()
    vel_pub = rospy.Publisher('/jackal0/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
    rospy.init_node('jackal_turn', anonymous=True)
    rospy.Subscriber("/jackal0/global_pos", Pose, globalCallback)
    rospy.Subscriber("/jackal0/imu/data", Imu, imuCallback)
    rospy.Subscriber("/jackal0/front/scan", LaserScan, laserCallback)
    rospy.Subscriber("/jackal0/goal_pos", Pose, goalCallback)
    rate = rospy.Rate(10)         # rate = 100Hz
    dt = 0.1                      # follow the rate

    vel = Twist()
    pf = potential_field()
    solution = race()

    rate.sleep()
    previous_range = [message.range, min(message.range)]
    rate.sleep()

    goal = position()
    goal.x = message.goal_pos.x
    goal.y = message.goal_pos.y
    solution.update_goal(goal)
    robot_pos = position()
    robot_pos.x = message.posx
    robot_pos.y = message.posy
    #rospy.loginfo("goalx:%f, goaly:%f", goalx, goaly)

    map_size = 2000
    prior_pos = np.array([message.posx,message.posy,message.theta])
    map = maps(prior_pos, map_size)

    while solution.reached(robot_pos) != 0:
        if solution.reached(robot_pos) == 1:
            linearx, steering = solution.going(robot_pos, message.theta, message.laser_data)
        else:
            linearx, steering = solution.goto(robot_pos, message.theta)
        #linearx, steering = pf.potential_field_force(message.posx, message.posy, message.theta, goalx, goaly, message.laser_data)

        #rospy.loginfo("a0: %.4f, a1: %.4f, r0: %.4f, r1: %.4f, v: %.4f, s: %.4f, d: %.4f, t: %.4f", attractive[0], attractive[1], repulsive[0], repulsive[1], linearx, steering, distance, angle)

        robot_pos.x = message.posx
        robot_pos.y = message.posy

        prior_pos = np.array([message.posx, message.posy, message.theta])
        map.update(prior_pos, message.laser_data)

        vel.linear.x = linearx
        vel.angular.z = steering
        vel_pub.publish(vel)

        rate.sleep()

        plt.cla()
        plt.imshow(map.grid_map)
        plt.pause(0.0001)

    image_map = maps.grid_map * 255
    im = Image.fromarray(image_map)
    im = im.convert('L')
    im.save('outfile.png')


        
        
if __name__ == "__main__":
    main()



