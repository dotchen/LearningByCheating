import numpy as np

from agents.navigation.agent import Agent
from agents.navigation.local_planner import LocalPlannerNew

from .controller import PIDController

import carla


TURNING_PID = {
        'K_P': 1.5,
        'K_I': 0.5,
        'K_D': 0.0,
        'fps': 10
        }


class RoamingAgentMine(Agent):
    def __init__(self, vehicle, resolution, threshold_before, threshold_after):
        super().__init__(vehicle)

        self._proximity_threshold = 9.5
        self.speed_control = PIDController(K_P=1.0)
        self.turn_control = PIDController(**TURNING_PID)

        self._local_planner = LocalPlannerNew(self._vehicle, resolution, threshold_before, threshold_after)
        self.set_route = self._local_planner.set_route

        self.debug = dict()

    def run_step(self, inputs=None, debug=False, debug_info=None):
        self._local_planner.run_step()

        ox = self._vehicle.get_transform().get_forward_vector().x
        oy = self._vehicle.get_transform().get_forward_vector().y
        rot = np.array([
            [ox, oy],
            [-oy, ox]])

        target = self._local_planner.target[0].transform.location
        target = np.array([target.x, target.y])
        pos = self._vehicle.get_location()
        pos = np.array([pos.x, pos.y])
        diff = rot.dot(target - pos)

        speed = self._vehicle.get_velocity()
        speed = np.linalg.norm([speed.x, speed.y])

        u = np.array([diff[0], diff[1], 0.0])
        v = np.array([1.0, 0.0, 0.0])
        theta = np.arccos(np.dot(u, v) / np.linalg.norm(u))
        theta = theta if np.cross(u, v)[2] < 0 else -theta
        steer = self.turn_control.step(theta)

        target_speed = 6.0

        if int(self._local_planner.target[1]) not in [3, 4]:
            target_speed *= 0.75

        delta = target_speed - speed
        throttle = self.speed_control.step(delta)

        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = 0.0
        control.manual_gear_shift = False

        self.vehicle = self._vehicle.get_location()
        self.road_option = self._local_planner.target[1]

        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter('*vehicle*')
        lights_list = actor_list.filter('*traffic_light*')
        walkers_list = actor_list.filter('*walker*')

        blocking_vehicle, vehicle = self._is_vehicle_hazard(vehicle_list)
        blocking_light, traffic_light = self._is_light_red(lights_list)
        blocking_walker, walker = self._is_walker_hazard(walkers_list)
        hazard_detected = blocking_vehicle or blocking_light or blocking_walker

        if blocking_vehicle:
            self.waypoint = vehicle.get_location()
        elif traffic_light:
            self.waypoint = traffic_light.get_location()
        elif blocking_walker:
            self.waypoint = walker.get_location()
        else:
            self.waypoint = self._local_planner.target[0].transform.location

        if hazard_detected:
            control = self.emergency_stop()
            control.manual_gear_shift = False

            return control

        self.debug['target'] = (self.waypoint.x, self.waypoint.y)

        return control
        
        # x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        # y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        # return [int(x - offset[0]), int(y - offset[1])]
