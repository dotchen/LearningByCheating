from pathlib import Path

import queue

import numpy as np

import carla

from agents.navigation.local_planner import RoadOption, LocalPlannerNew, LocalPlannerOld

from .base_suite import BaseSuite


def from_file(poses_txt):
    pairs_file = Path(__file__).parent / poses_txt
    pairs = pairs_file.read_text().strip().split('\n')
    pairs = [(int(x[0]), int(x[1])) for x in map(lambda y: y.split(), pairs)]

    return pairs


class PointGoalSuite(BaseSuite):
    def __init__(
            self, success_dist=5.0, col_is_failure=False,
            viz_camera=False, planner='new', poses_txt='', **kwargs):
        super().__init__(**kwargs)

        self.success_dist = success_dist
        self.col_is_failure = col_is_failure
        self.planner = planner
        self.poses = from_file(poses_txt)

        self.command = RoadOption.LANEFOLLOW

        self.timestamp_active = 0
        self._timeout = float('inf')

        self.viz_camera = viz_camera
        self._viz_queue = None

    def init(self, target=1, **kwargs):
        self._target_pose = self._map.get_spawn_points()[target]

        super().init(**kwargs)

    def ready(self):
        # print (self.planner)
        if self.planner == 'new':
            self._local_planner = LocalPlannerNew(self._player, 2.5, 9.0, 1.5)
        else:
            self._local_planner = LocalPlannerOld(self._player)

        self._local_planner.set_route(self._start_pose.location, self._target_pose.location)
        self._timeout = self._local_planner.calculate_timeout()

        return super().ready()

    def tick(self):
        result = super().tick()

        self._local_planner.run_step()
        self.command = self._local_planner.checkpoint[1]
        self.node = self._local_planner.checkpoint[0].transform.location
        self._next = self._local_planner.target[0].transform.location

        return result

    def get_observations(self):
        result = dict()
        result.update(super().get_observations())
        result['command'] = int(self.command)
        result['node'] = np.array([self.node.x, self.node.y])
        result['next'] = np.array([self._next.x, self._next.y])

        return result

    @property
    def weathers(self):
        return self._weathers

    @property
    def pose_tasks(self):
        return self.poses

    def clean_up(self):
        super().clean_up()

        self.timestamp_active = 0
        self._timeout = float('inf')
        self._local_planner = None

        # Clean-up cameras
        if self._viz_queue:
            with self._viz_queue.mutex:
                self._viz_queue.queue.clear()

    def is_failure(self):
        if self.timestamp_active >= self._timeout or self._tick >= 10000:
            return True
        elif self.col_is_failure and self.collided:
            return True

        return False

    def is_success(self):
        location = self._player.get_location()
        distance = location.distance(self._target_pose.location)
        
        return distance <= self.success_dist

    def apply_control(self, control):
        result = super().apply_control(control)

        # is_light_red = self._is_light_red(agent)
        #
        # if is_light_red:
        #     self.timestamp_active -= 1

        self.timestamp_active += 1

        # Return diagnostics
        location = self._player.get_location()
        orientation = self._player.get_transform().get_forward_vector()
        velocity = self._player.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        info = {
                'x': location.x,
                'y': location.y,
                'z': location.z,
                'ori_x': orientation.x,
                'ori_y': orientation.y,
                'speed': speed,
                'collided': self.collided,
                'invaded': self.invaded,
                'distance_to_goal': self._local_planner.distance_to_goal,
                'viz_img': self._viz_queue.get() if self.viz_camera else None
                }

        info.update(result)

        return info

    def _is_light_red(self, agent):
        lights_list = self._world.get_actors().filter('*traffic_light*')
        is_light_red, _ = agent._is_light_red(lights_list)

        return is_light_red

    def _setup_sensors(self):
        super()._setup_sensors()

        if self.viz_camera:
            viz_camera_bp = self._blueprints.find('sensor.camera.rgb')
            viz_camera = self._world.spawn_actor(
                viz_camera_bp,
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self._player)
            viz_camera_bp.set_attribute('image_size_x', '640')
            viz_camera_bp.set_attribute('image_size_y', '480')

            # Set camera queues
            self._viz_queue = queue.Queue()
            viz_camera.listen(self._viz_queue.put)

            self._actor_dict['sensor'].append(viz_camera)

    def render_world(self):
        import matplotlib.pyplot as plt

        from matplotlib.patches import Circle

        plt.clf()
        plt.tight_layout()
        plt.axis('off')

        fig, ax = plt.subplots(1, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        world = super().render_world()
        world[np.all(world == (255, 255, 255), axis=-1)] = [100, 100, 100]
        world[np.all(world == (0, 0, 0), axis=-1)] = [255, 255, 255]

        ax.imshow(world)

        prev_command = -1

        for i, (node, command) in enumerate(self._local_planner._route):
            command = int(command)
            pixel_x, pixel_y = self.world_to_pixel(node.transform.location)

            if command != prev_command and prev_command != -1:
                _command = {1: 'L', 2: 'R', 3: 'S', 4: 'F'}.get(command, '???')
                ax.text(pixel_x, pixel_y, _command, fontsize=8, color='black')
                ax.add_patch(Circle((pixel_x, pixel_y), 5, color='black'))
            elif i == 0 or i == len(self._local_planner._route)-1:
                text = 'start' if i == 0 else 'end'
                ax.text(pixel_x, pixel_y, text, fontsize=8, color='blue')
                ax.add_patch(Circle((pixel_x, pixel_y), 5, color='blue'))
            elif i % (len(self._local_planner._route) // 10) == 0:
                ax.add_patch(Circle((pixel_x, pixel_y), 3, color='red'))

            prev_command = int(command)

        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()

        return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
