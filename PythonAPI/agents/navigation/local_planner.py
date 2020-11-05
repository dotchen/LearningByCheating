#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import random

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints

import numpy as np


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
    
    def __int__(self):
        """
        Returns the int value as an integer.

        Args:
            self: (todo): write your description
        """
        return self.value



class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).
    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F
            target_speed -- desired cruise speed in Km/h
            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead
            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}
            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self._init_controller(opt_dict)

    def _init_controller(self, opt_dict):
        """
        Controller initialization.
        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 10.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 1.0,
            'K_D': 0.0,
            'K_I': 0.5,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                    opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                       args_lateral=args_lateral_dict,
                                                       args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.
        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.
        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan):
        """
        Set the waypoints for the waypoints

        Args:
            self: (todo): write your description
            current_plan: (str): write your description
        """
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._global_plan = True

    def run_step(self, debug=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.
        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)


        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        if len(self._waypoints_queue) == 0:
            if not current_waypoint.is_intersection:
                self._target_road_option = RoadOption.LANEFOLLOW

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)
    
        # if self._target_road_option != RoadOption.LANEFOLLOW not current_waypoint.is_intersection:
        #     self._target_road_option = RoadOption.LANEFOLLOW
        return control


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT


class LocalPlannerNew(object):
    def __init__(self, vehicle, resolution=15, threshold_before=2.5, threshold_after=5.0):
        """
        Initialize the target.

        Args:
            self: (todo): write your description
            vehicle: (todo): write your description
            resolution: (todo): write your description
            threshold_before: (float): write your description
            threshold_after: (float): write your description
        """
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

        # Max skip avoids misplanning when route includes both lanes.
        self._max_skip = 20
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after

        self._vehicle = vehicle
        self._map = vehicle.get_world().get_map()
        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._map, resolution))
        self._grp.setup()

        self._route = None
        self._waypoints_queue = deque(maxlen=20000)

        self.target = (None, None)
        self.checkpoint = (None, None)
        self.distance_to_goal = float('inf')
        self.distances = deque(maxlen=20000)

    def set_route(self, start, target):
        """
        Set a target.

        Args:
            self: (todo): write your description
            start: (todo): write your description
            target: (todo): write your description
        """
        self._waypoints_queue.clear()

        self._route = self._grp.trace_route(start, target)

        self.distance_to_goal = 0.0

        prev = None

        for node in self._route:
            self._waypoints_queue.append(node)

            cur = node[0].transform.location

            if prev is not None:
                delta = np.sqrt((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2)

                self.distance_to_goal += delta
                self.distances.append(delta)

            prev = cur

        self.target = self._waypoints_queue[0]
        self.checkpoint = (
                self._map.get_waypoint(self._vehicle.get_location()),
                RoadOption.LANEFOLLOW)

    def run_step(self):
        """
        Run a single waypoints step.

        Args:
            self: (todo): write your description
        """
        assert self._route is not None

        u = self._vehicle.get_transform().location
        max_index = -1

        for i, (node, command) in enumerate(self._waypoints_queue):
            if i > self._max_skip:
                break

            v = node.transform.location
            distance = np.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2)

            if int(self.checkpoint[1]) == 4 and int(command) != 4:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            if distance < threshold:
                self.checkpoint = (node, command)
                max_index = i

        for i in range(max_index + 1):
            if self.distances:
                self.distance_to_goal -= self.distances[0]
                self.distances.popleft()

            self._waypoints_queue.popleft()

        if len(self._waypoints_queue) > 0:
            self.target = self._waypoints_queue[0]

    def calculate_timeout(self, fps=10):
        """
        Calculate the timeout of the current frame.

        Args:
            self: (todo): write your description
            fps: (todo): write your description
        """
        _numpy = lambda p: np.array([p.transform.location.x, p.transform.location.y])

        distance = 0
        node_prev = None

        for node_cur, _ in self._route:
            if node_prev is None:
                node_prev = node_cur

            distance += np.linalg.norm(_numpy(node_cur) - _numpy(node_prev))
            node_prev = node_cur

        timeout_in_seconds = ((distance / 1000.0) / 5.0) * 3600.0 + 20.0
        timeout_in_frames = timeout_in_seconds * fps

        return timeout_in_frames


class LocalPlannerOld(object):
    def __init__(self, vehicle, resolution=1.5):
        """
        Initialize the target.

        Args:
            self: (todo): write your description
            vehicle: (todo): write your description
            resolution: (todo): write your description
        """
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

        self._dt = 1.0 / 10.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * 0.9

        self._vehicle = vehicle
        self._map = vehicle.get_world().get_map()
        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._map, resolution))
        self._grp.setup()

        self._route = None
        self._waypoints_queue = deque(maxlen=20000)

        self.target = (None, None)
        self.checkpoint = (None, None)
        self.distance_to_goal = float('inf')
        self.distances = deque(maxlen=20000)

    def set_route(self, start, target):
        """
        Set a target.

        Args:
            self: (todo): write your description
            start: (todo): write your description
            target: (todo): write your description
        """
        self._waypoints_queue.clear()

        self._route = self._grp.trace_route(start, target)
        self.distance_to_goal = 0.0

        prev = None

        for node in self._route:
            self._waypoints_queue.append(node)

            cur = node[0].transform.location

            if prev is not None:
                delta = np.sqrt((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2)

                self.distance_to_goal += delta
                self.distances.append(delta)

            prev = cur

        self.target = self._waypoints_queue[0]
        self.checkpoint = (
                self._map.get_waypoint(self._vehicle.get_location()),
                RoadOption.LANEFOLLOW)

    def run_step(self):
        """
        Run a single waypoints.

        Args:
            self: (todo): write your description
        """
        assert self._route is not None

        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoints_queue):
            if distance_vehicle(waypoint, vehicle_transform) < self._min_distance:
                max_index = i

        if max_index >= 0:
            for i in range(max_index + 1):
                if self.distances:
                    self.distance_to_goal -= self.distances[0]
                    self.distances.popleft()

                self._waypoints_queue.popleft()

        if len(self._waypoints_queue) > 0:
            self.target = self._waypoints_queue[0]

    def calculate_timeout(self, fps=10):
        """
        Calculate the timeout of the current frame.

        Args:
            self: (todo): write your description
            fps: (todo): write your description
        """
        _numpy = lambda p: np.array([p.transform.location.x, p.transform.location.y])

        distance = 0
        node_prev = None

        for node_cur, _ in self._route:
            if node_prev is None:
                node_prev = node_cur

            distance += np.linalg.norm(_numpy(node_cur) - _numpy(node_prev))
            node_prev = node_cur

        timeout_in_seconds = ((distance / 1000.0) / 5.0) * 3600.0 + 20.0
        timeout_in_frames = timeout_in_seconds * fps

        return timeout_in_frames
