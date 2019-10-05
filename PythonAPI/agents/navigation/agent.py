#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from enum import Enum

import carla
import os.path as osp
import numpy as np
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle, compute_yaw_difference

from skimage.io import imread


WORLD_OFFSETS = {
    'Town01' : (-52.059906005859375, -52.04995942115784),
    'Town02' : (-57.459808349609375, 55.3907470703125)
}
PIXELS_PER_METER = 5

class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class Agent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._last_traffic_light = None

        self._world = None
        self._map = None
        self._debug_info = None

        while self._world is None and self._map is None:
            try:
                self._world = self._vehicle.get_world()
                self._map = self._world.get_map()
            except RuntimeError as e:
                print(e)

        self._road_map = imread(osp.join(osp.dirname(__file__), '%s.png' % self._map.name))

    def run_step(self, inputs=None, debug=False):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        if self._map.name == 'Town01' or self._map.name == 'Town02':
            return self._is_light_red_europe_style(lights_list)
        else:
            return self._is_light_red_us_style(lights_list)

    def _is_light_red_europe_style(self, lights_list):
        """
        This method is specialized to check European style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                  affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us


        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        
        for traffic_light in lights_list:
            location = traffic_light.get_location()
            object_waypoint = self._map.get_waypoint(location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            if object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue
            if not is_within_distance_ahead(
                    location,
                    ego_vehicle_location,
                    self._vehicle.get_transform().rotation.yaw,
                    self._proximity_threshold, degree=60):
                continue
            if traffic_light.state != carla.libcarla.TrafficLightState.Red:
                continue

            return (True, traffic_light)

        return (False, None)

    def _is_light_red_us_style(self, lights_list, debug=False):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            # It is too late. Do not block the intersection! Keep going!
            return (False, None)

        if self._local_planner._target_waypoint is not None:
            if self._local_planner._target_waypoint.is_intersection:
                potential_lights = []
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(
                            loc,
                            ego_vehicle_location,
                            self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 80.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print(
                                '=== Magnitude = {} | Angle = {} | ID = {}'.format(
                                    sel_magnitude, min_angle, sel_traffic_light.id))

                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
                        return (True, self._last_traffic_light)
                else:
                    self._last_traffic_light = None

        return (False, None)

    def _is_walker_hazard(self, walkers_list):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for walker in walkers_list:
            loc = walker.get_location()
            dist = loc.distance(ego_vehicle_location)
            degree = 162 / (np.clip(dist, 1.5, 10.5)+0.3)
            if self._is_point_on_sidewalk(loc):
                continue

            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold, degree=degree):
                return (True, walker)

        return (False, None)

    def _is_vehicle_hazard(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_orientation = self._vehicle.get_transform().rotation.yaw
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            loc = target_vehicle.get_location()
            ori = target_vehicle.get_transform().rotation.yaw

            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())

            # waiting = ego_vehicle_waypoint.is_intersection and target_vehicle.get_traffic_light_state() == carla.TrafficLightState.Red
            # print ("Not our lane: ", other_lane)
            # print ("Waiting", waiting)

            # if the object is not in our lane it's not an obstacle
            # if not ego_vehicle_waypoint.is_intersection and target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id
            #     continue

            # if the object is waiting for it's not an obstacle
            # if waiting:
            #     continue

            if compute_yaw_difference(ego_vehicle_orientation, ori) <= 150 and is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold, degree=45):
                return (True, target_vehicle)

        return (False, None)


    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control


    def _world_to_pixel(self, location, offset=(0, 0)):
        world_offset = WORLD_OFFSETS[self._map.name]
        x = PIXELS_PER_METER * (location.x - world_offset[0])
        y = PIXELS_PER_METER * (location.y - world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]
    
    def _is_point_on_sidewalk(self, loc):
        # Convert to pixel coordinate
        pixel_x, pixel_y = self._world_to_pixel(loc)
        pixel_y = np.clip(pixel_y, 0, self._road_map.shape[0]-1)
        pixel_x = np.clip(pixel_x, 0, self._road_map.shape[1]-1)
        point = self._road_map[pixel_y, pixel_x, 0]

        return point == 0
