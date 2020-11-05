from itertools import product

from bird_view.utils import carla_utils as cu


class BaseSuite(cu.CarlaWrapper):
    def __init__(self, weathers=[0], n_vehicles=0, n_pedestrians=0, disable_two_wheels=False, **kwargs):
        """
        Initialize weather objects.

        Args:
            self: (todo): write your description
            weathers: (todo): write your description
            n_vehicles: (int): write your description
            n_pedestrians: (str): write your description
            disable_two_wheels: (todo): write your description
        """
        super().__init__(**kwargs)

        self._weathers = weathers
        self.n_vehicles = n_vehicles
        self.n_pedestrians = n_pedestrians
        self.disable_two_wheels = disable_two_wheels

    def get_spawn_point(self, pose_num):
        """
        Returns the number of the currently running point

        Args:
            self: (todo): write your description
            pose_num: (int): write your description
        """
        return self._spawn_points[pose_num]

    def is_failure(self):
        """
        Checks if the failure is a failure.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def is_success(self):
        """
        Check if the request.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    @property
    def pose_tasks(self):
        """
        : return :. tasks.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    @property
    def weathers(self):
        """
        The weather weather weather object.

        Args:
            self: (todo): write your description
        """
        return self._weathers

    @property
    def all_tasks(self):
        """
        Yield all weather tasks.

        Args:
            self: (todo): write your description
        """
        for (start, target), weather in product(self.pose_tasks, self.weathers):
            run_name = 's%d_t%d_w%d' % (start, target, weather)

            yield weather, (start, target), run_name
