import collections
import queue
import weakref
import time
import random

import numpy as np
import math

# import needed due to https://github.com/pytorch/pytorch/issues/36034
import torchvision
import carla

from carla import ColorConverter
from carla import WeatherParameters

from .map_utils import Wrapper as map_utils


PRESET_WEATHERS = {
    1: WeatherParameters.ClearNoon,
    2: WeatherParameters.CloudyNoon,
    3: WeatherParameters.WetNoon,
    4: WeatherParameters.WetCloudyNoon,
    5: WeatherParameters.MidRainyNoon,
    6: WeatherParameters.HardRainNoon,
    7: WeatherParameters.SoftRainNoon,
    8: WeatherParameters.ClearSunset,
    9: WeatherParameters.CloudySunset,
    10: WeatherParameters.WetSunset,
    11: WeatherParameters.WetCloudySunset,
    12: WeatherParameters.MidRainSunset,
    13: WeatherParameters.HardRainSunset,
    14: WeatherParameters.SoftRainSunset,
}

TRAIN_WEATHERS = {
        'clear_noon': WeatherParameters.ClearNoon, #1
        'wet_noon': WeatherParameters.WetNoon, #3
        'hardrain_noon': WeatherParameters.HardRainNoon, #6
        'clear_sunset': WeatherParameters.ClearSunset, #8
        }

WEATHERS = list(TRAIN_WEATHERS.values())


BACKGROUND = [0, 47, 0]
COLORS = [
        (102, 102, 102),
        (253, 253, 17),
        (204, 6, 5),
        (250, 210, 1),
        (39, 232, 51),
        (0, 0, 142),
        (220, 20, 60)
        ]


TOWNS = ['Town01', 'Town02', 'Town03', 'Town04']
VEHICLE_NAME = 'vehicle.ford.mustang'

def is_within_distance_ahead(target_location, current_location, orientation, max_distance, degree=60):
    u = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y])
    distance = np.linalg.norm(u)

    if distance > max_distance:
        return False

    v = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))])

    angle = math.degrees(math.acos(np.dot(u, v) / distance))

    return angle < degree


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 0.1

    world.apply_settings(settings)


def carla_img_to_np(carla_img):
    carla_img.convert(ColorConverter.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype('uint8'))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:,:,:3]
    img = img[:,:,::-1]

    return img


def get_birdview(observations):
    birdview = [
            observations['road'],
            observations['lane'],
            observations['traffic'],
            observations['vehicle'],
            observations['pedestrian']
            ]
    birdview = [x if x.ndim == 3 else x[...,None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview


def process(observations):
    result = dict()
    result['rgb'] = observations['rgb'].copy()
    result['birdview'] = observations['birdview'].copy()
    result['collided'] = observations['collided']

    control = observations['control']
    control = [control.steer, control.throttle, control.brake]

    result['control'] = np.float32(control)

    measurements = [
            observations['position'],
            observations['orientation'],
            observations['velocity'],
            observations['acceleration'],
            observations['command'].value,
            observations['control'].steer,
            observations['control'].throttle,
            observations['control'].brake,
            observations['control'].manual_gear_shift,
            observations['control'].gear
            ]
    measurements = [x if isinstance(x, np.ndarray) else np.float32([x]) for x in measurements]
    measurements = np.concatenate(measurements, 0)

    result['measurements'] = measurements

    return result


def visualize_birdview(birdview):
    """
    0 road
    1 lane
    2 red light
    3 yellow light
    4 green light
    5 vehicle
    6 pedestrian
    """
    h, w = birdview.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND

    for i in range(len(COLORS)):
        canvas[birdview[:,:,i] > 0] = COLORS[i]

    return canvas


def visualize_predicted_birdview(predicted, tau=0.5):
    # mask = np.concatenate([predicted.max(0)[np.newaxis]] * 7, 0)
    # predicted[predicted != mask] = 0
    # predicted[predicted == mask] = 1

    predicted[predicted < tau] = -1

    return visualize_birdview(predicted.transpose(1, 2, 0))


class PedestrianTracker(object):
    def __init__(self, wrapper, peds, ped_controllers, respawn_peds=True, speed_threshold=0.1, stuck_limit=20):
        self._wrapper = wrapper()
        self._peds = peds
        self._ped_controllers = ped_controllers
        
        self._ped_timers = dict()
        for ped in peds:
            self._ped_timers[ped.id] = 0
        
        self._speed_threshold = speed_threshold
        self._stuck_limit = stuck_limit
        self._respawn_peds = respawn_peds
            
    def tick(self):
        for ped in self._peds:
            vel = ped.get_velocity()
            speed = np.linalg.norm([vel.x, vel.y, vel.z])

            if ped.id in self._ped_timers and speed < self._speed_threshold:
                self._ped_timers[ped.id] += 1
            else:
                self._ped_timers[ped.id] = 0
        
        stuck_ped_ids = []
        for ped_id, stuck_time in self._ped_timers.items():
            if stuck_time >= self._stuck_limit and self._respawn_peds:
                stuck_ped_ids.append(ped_id)

        ego_vehicle_location = self._wrapper._player.get_location()

        for ped_controller in self._ped_controllers:
            ped_id = ped_controller.parent.id
            if ped_id not in stuck_ped_ids:
                continue
            
            self._ped_timers.pop(ped_id)

            old_loc = ped.get_location()

            loc = None
            while True:
                _loc = self._wrapper._world.get_random_location_from_navigation()
                if _loc is not None and _loc.distance(ego_vehicle_location) >= 10.0 \
                                    and _loc.distance(old_loc) >= 10.0:
                    loc = _loc
                    break
            
            ped_controller.teleport_to_location(loc)
            print ("Teleported walker %d to %s"%(ped_id, loc))
        

class TrafficTracker(object):
    LANE_WIDTH = 5.0

    def __init__(self, agent, world):
        self._agent = agent
        self._world = world

        self._prev = None
        self._cur = None

        self.total_lights_ran = 0
        self.total_lights = 0
        self.ran_light = False
        
        self.last_light_id = -1

    def tick(self):
        self.ran_light = False
        self._prev = self._cur
        self._cur = self._agent.get_location()

        if self._prev is None or self._cur is None:
            return

        light = TrafficTracker.get_active_light(self._agent, self._world)
        active_light = light
        
        if light is not None and light.id != self.last_light_id:
            self.total_lights += 1
            self.last_light_id = light.id
            
        light = TrafficTracker.get_closest_light(self._agent, self._world)

        if light is None or light.state != carla.libcarla.TrafficLightState.Red:
            return

        light_location = light.get_transform().location
        light_orientation = light.get_transform().get_forward_vector()

        delta = self._cur - self._prev

        p = np.array([self._prev.x, self._prev.y])
        r = np.array([delta.x, delta.y])

        q = np.array([light_location.x, light_location.y])
        s = TrafficTracker.LANE_WIDTH * np.array([-light_orientation.x, -light_orientation.y])

        if TrafficTracker.line_line_intersect(p, r, q, s):
            self.ran_light = True
            self.total_lights_ran += 1


    @staticmethod
    def get_closest_light(agent, world):
        location = agent.get_location()
        closest = None
        closest_distance = float('inf')

        for light in world.get_actors().filter('*traffic_light*'):
            delta = location - light.get_transform().location
            distance = np.sqrt(sum([delta.x ** 2, delta.y ** 2, delta.z ** 2]))

            if distance < closest_distance:
                closest = light
                closest_distance = distance

        return closest

    @staticmethod
    def get_active_light(ego_vehicle, world):
        
        _map = world.get_map()
        ego_vehicle_location = ego_vehicle.get_location()
        ego_vehicle_waypoint = _map.get_waypoint(ego_vehicle_location)
        
        lights_list = world.get_actors().filter('*traffic_light*')
        
        for traffic_light in lights_list:
            location = traffic_light.get_location()
            object_waypoint = _map.get_waypoint(location)
            
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            if object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue
            
            if not is_within_distance_ahead(
                    location,
                    ego_vehicle_location,
                    ego_vehicle.get_transform().rotation.yaw,
                    10., degree=60):
                continue

            return traffic_light

        return None

    @staticmethod
    def line_line_intersect(p, r, q, s):
        r_cross_s = np.cross(r, s)
        q_minus_p = q - p

        if abs(r_cross_s) < 1e-3:
            return False

        t = np.cross(q_minus_p, s) / r_cross_s
        u = np.cross(q_minus_p, r) / r_cross_s

        if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
            return True

        return False


class CarlaWrapper(object):
    def __init__(
            self, town='Town01', vehicle_name=VEHICLE_NAME, port=2000, client=None,
            col_threshold=400, big_cam=False, seed=None, respawn_peds=True, **kwargs):
        
        if client is None:    
            self._client = carla.Client('localhost', port)
        else:
            self._client = client
            
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()
        self._vehicle_bp = np.random.choice(self._blueprints.filter(vehicle_name))
        self._vehicle_bp.set_attribute('role_name', 'hero')

        self._tick = 0
        self._player = None

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)

        self._big_cam = big_cam
        self.col_threshold = col_threshold
        self.collided = False
        self._collided_frame_number = -1

        self.invaded = False
        self._invaded_frame_number = -1

        self.traffic_tracker = None

        self.n_vehicles = 0
        self.n_pedestrians = 0

        self._rgb_queue = None
        self.rgb_image = None
        self._big_cam_queue = None
        self.big_cam_image = None
        
        self.seed = seed

        self._respawn_peds = respawn_peds
        self.disable_two_wheels = False
        

    def spawn_vehicles(self):

        blueprints = self._blueprints.filter('vehicle.*')
        if self.disable_two_wheels:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self._map.get_spawn_points()

        for i in range(self.n_vehicles):
            blueprint = np.random.choice(blueprints)
            blueprint.set_attribute('role_name', 'autopilot')
    
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            
            vehicle = None
            while vehicle is None:
                vehicle = self._world.try_spawn_actor(blueprint, np.random.choice(spawn_points))

            vehicle.set_autopilot(True)
            vehicle.start_dtcrowd()

            self._actor_dict['vehicle'].append(vehicle)

        print ("spawned %d vehicles"%len(self._actor_dict['vehicle']))

    def spawn_pedestrians(self, n_pedestrians):
        SpawnActor = carla.command.SpawnActor

        peds_spawned = 0
        
        walkers = []
        controllers = []
        
        while peds_spawned < n_pedestrians:
            spawn_points = []
            _walkers = []
            _controllers = []
            
            for i in range(n_pedestrians - peds_spawned):
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()
    
                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
    
            blueprints = self._blueprints.filter('walker.pedestrian.*')
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprints)
    
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
    
                batch.append(SpawnActor(walker_bp, spawn_point))
    
            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    print(result.error)
                else:
                    peds_spawned += 1
                    _walkers.append(result.actor_id)
    
            walker_controller_bp = self._blueprints.find('controller.ai.walker')
            batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker) for walker in _walkers]
    
            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    print(result.error)
                else:
                    _controllers.append(result.actor_id)
                    
            controllers.extend(_controllers)
            walkers.extend(_walkers)

        print ("spawned %d pedestrians"%len(controllers))

        return self._world.get_actors(walkers), self._world.get_actors(controllers)
        
    def set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        elif weather_string in TRAIN_WEATHERS:
            weather = TRAIN_WEATHERS[weather_string]
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def init(self, start=0, weather='random', n_vehicles=0, n_pedestrians=0):
        while True:
            self.n_vehicles = n_vehicles or self.n_vehicles
            self.n_pedestrians = n_pedestrians or self.n_pedestrians
            self._start_pose = self._map.get_spawn_points()[start]
    
            self.clean_up()
            self.spawn_player()
            self._setup_sensors()
    
            # Hiding away the gore.
            map_utils.init(self._client, self._world, self._map, self._player)
    
            # Deterministic.
            if self.seed is not None:
                np.random.seed(self.seed)
    
            self.set_weather(weather)
            
            # Spawn vehicles
            self.spawn_vehicles()
            
            # Spawn pedestrians
            peds, ped_controllers = self.spawn_pedestrians(self.n_pedestrians)
            self._actor_dict['pedestrian'].extend(peds)
            self._actor_dict['ped_controller'].extend(ped_controllers)
            
            self.peds_tracker = PedestrianTracker(weakref.ref(self), self.pedestrians, self.ped_controllers, respawn_peds=self._respawn_peds)
    
            self.traffic_tracker = TrafficTracker(self._player, self._world)

            ready = self.ready()
            if ready:
                break

    def spawn_player(self):
        self._player = self._world.spawn_actor(self._vehicle_bp, self._start_pose)
        self._player.set_autopilot(False)
        self._player.start_dtcrowd()
        self._actor_dict['player'].append(self._player)
        

    def ready(self, ticks=50):
        self.tick()
        self.get_observations()

        for controller in self._actor_dict['ped_controller']:
            controller.start()
            controller.go_to_location(self._world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())
        
        for _ in range(ticks):
            self.tick()
            self.get_observations()

        with self._rgb_queue.mutex:
            self._rgb_queue.queue.clear()

        self._time_start = time.time()
        self._tick = 0
        
        print ("Initial collided: %s"%self.collided)
        
        return not self.collided

    def tick(self):
        self._world.tick()
        self._tick += 1

        # More hiding.
        map_utils.tick()

        self.traffic_tracker.tick()
        self.peds_tracker.tick()

        # Put here for speed (get() busy polls queue).
        while self.rgb_image is None or self._rgb_queue.qsize() > 0:
            self.rgb_image = self._rgb_queue.get()
        
        if self._big_cam:
            while self.big_cam_image is None or self._big_cam_queue.qsize() > 0:
                self.big_cam_image = self._big_cam_queue.get()

        return True

    def get_observations(self):
        result = dict()
        result.update(map_utils.get_observations())
        # print ("%.3f, %.3f"%(self.rgb_image.timestamp, self._world.get_snapshot().timestamp.elapsed_seconds))
        result.update({
            'rgb': carla_img_to_np(self.rgb_image),
            'birdview': get_birdview(result),
            'collided': self.collided
            })

        if self._big_cam:
            result.update({
                'big_cam': carla_img_to_np(self.big_cam_image),
            })

        return result

    def apply_control(self, control=None, move_peds=True):
        """
        Applies very naive pedestrian movement.
        """
        if control is not None:
            self._player.apply_control(control)

        return {
                't': self._tick,
                'wall': time.time() - self._time_start,
                'ran_light': self.traffic_tracker.ran_light
                }

    def clean_up(self):
        for vehicle in self._actor_dict['vehicle']:
            # continue
            vehicle.stop_dtcrowd()
        
        for controller in self._actor_dict['ped_controller']:
            controller.stop()

        for sensor in self._actor_dict['sensor']:
            sensor.destroy()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()

        self._tick = 0
        self._time_start = time.time()
        
        if self._player:
            self._player.stop_dtcrowd()
        self._player = None

        # Clean-up cameras
        if self._rgb_queue:
            with self._rgb_queue.mutex:
                self._rgb_queue.queue.clear()
        
        if self._big_cam_queue:
            with self._big_cam_queue.mutex:
                self._big_cam_queue.queue.clear()
    
    @property
    def pedestrians(self):
        return self._actor_dict.get('pedestrian', [])
        
    @property
    def ped_controllers(self):
        return self._actor_dict.get('ped_controller', [])


    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        # Camera.
        self._rgb_queue = queue.Queue()

        if self._big_cam:
            self._big_cam_queue = queue.Queue()
            rgb_camera_bp = self._blueprints.find('sensor.camera.rgb')
            rgb_camera_bp.set_attribute('image_size_x', '800')
            rgb_camera_bp.set_attribute('image_size_y', '600')
            rgb_camera_bp.set_attribute('fov', '90')
            big_camera = self._world.spawn_actor(
                rgb_camera_bp,
                carla.Transform(carla.Location(x=1.0, z=1.4), carla.Rotation(pitch=0)),
                attach_to=self._player)
            big_camera.listen(self._big_cam_queue.put)
            self._actor_dict['sensor'].append(big_camera)
            
        rgb_camera_bp = self._blueprints.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '384')
        rgb_camera_bp.set_attribute('image_size_y', '160')
        rgb_camera_bp.set_attribute('fov', '90')
        rgb_camera = self._world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self._player)

        rgb_camera.listen(self._rgb_queue.put)
        self._actor_dict['sensor'].append(rgb_camera)
        
        

        # Collisions.
        self.collided = False
        self._collided_frame_number = -1

        collision_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.collision'),
                carla.Transform(), attach_to=self._player)
        collision_sensor.listen(
                lambda event: self.__class__._on_collision(weakref.ref(self), event))
        self._actor_dict['sensor'].append(collision_sensor)

        # Lane invasion.
        self.invaded = False
        self._invaded_frame_number = -1

        invasion_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.lane_invasion'),
                carla.Transform(), attach_to=self._player)
        invasion_sensor.listen(
                lambda event: self.__class__._on_invasion(weakref.ref(self), event))
        self._actor_dict['sensor'].append(invasion_sensor)

    @staticmethod
    def _on_collision(weakself, event):
        _self = weakself()

        if not _self:
            return

        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])

        if intensity > _self.col_threshold:
            _self.collided = True
            _self._collided_frame_number = event.frame_number

    @staticmethod
    def _on_invasion(weakself, event):
        _self = weakself()

        if not _self:
            return

        _self.invaded = True
        _self._invaded_frame_number = event.frame_number

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self.clean_up()

        set_sync_mode(self._client, False)

    def render_world(self):
        return map_utils.render_world()

    def world_to_pixel(self, pos):
        return map_utils.world_to_pixel(pos)
