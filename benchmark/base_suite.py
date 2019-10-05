from itertools import product

from bird_view.utils import carla_utils as cu


class BaseSuite(cu.CarlaWrapper):
    def __init__(self, weathers=[0], n_vehicles=0, n_pedestrians=0, disable_two_wheels=False, **kwargs):
        super().__init__(**kwargs)

        self._weathers = weathers
        self.n_vehicles = n_vehicles
        self.n_pedestrians = n_pedestrians
        self.disable_two_wheels = disable_two_wheels

    def get_spawn_point(self, pose_num):
        return self._spawn_points[pose_num]

    def is_failure(self):
        raise NotImplementedError

    def is_success(self):
        raise NotImplementedError

    @property
    def pose_tasks(self):
        raise NotImplementedError

    @property
    def weathers(self):
        return self._weathers

    @property
    def all_tasks(self):
        for (start, target), weather in product(self.pose_tasks, self.weathers):
            run_name = 's%d_t%d_w%d' % (start, target, weather)

            yield weather, (start, target), run_name
