"""
Stores tuples of (birdview, measurements, rgb).

Run from top level directory.
Sample usage -

python3 bird_view/data_collector.py \
        --dataset_path $PWD/data \
        --frame_skip 10 \
        --frames_per_episode 1000 \
        --n_episodes 100 \
        --port 3000 \
        --n_vehicles 0 \
        --n_pedestrians 0
"""
import argparse

from pathlib import Path

import numpy as np
import tqdm
import lmdb

import carla

from benchmark import make_suite
from bird_view.utils import carla_utils as cu
from bird_view.utils import bz_utils as bu

from bird_view.models.common import crop_birdview
from bird_view.models.controller import PIDController
from bird_view.models.roaming import RoamingAgentMine


def _debug(observations, agent_debug):
    import cv2

    processed = cu.process(observations)

    control = observations['control']
    control = [control.steer, control.throttle, control.brake]
    control = ' '.join(str('%.2f' % x).rjust(5, ' ') for x in control)
    real_control = observations['real_control']
    real_control = [real_control.steer, real_control.throttle, real_control.brake]
    real_control = ' '.join(str('%.2f' % x).rjust(5, ' ') for x in real_control)

    canvas = np.uint8(observations['rgb']).copy()
    rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
    cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

    WHITE = (255, 255, 255)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    def _write(text, i, j):
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(int(observations['command']), '???')

    _write('Command: ' + _command, 1, 0)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0)
    _write('Real: %s' % control, -5, 0)
    _write('Control: %s' % control, -4, 0)

    r = 2
    birdview = cu.visualize_birdview(crop_birdview(processed['birdview']))

    def _dot(x, y, color):
        x = int(x)
        y = int(y)
        birdview[176-r-x:176+r+1-x,96-r+y:96+r+1+y] = color

    _dot(0, 0, [255, 255, 255])

    ox, oy = observations['orientation']
    R = np.array([
        [ox,  oy],
        [-oy, ox]])

    u = np.array(agent_debug['waypoint']) - np.array(agent_debug['vehicle'])
    u = R.dot(u[:2])
    u = u * 4

    _dot(u[0], u[1], [255, 255, 255])

    def _stick_together(a, b):
        h = min(a.shape[0], b.shape[0])

        r1 = h / a.shape[0]
        r2 = h / b.shape[0]

        a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
        b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

        return np.concatenate([a, b], 1)

    full = _stick_together(canvas, birdview)

    bu.show_image('full', full)



class NoisyAgent(RoamingAgentMine):
    """
    Each parameter is in units of frames.
    State can be "drive" or "noise".
    """
    def __init__(self, env, noise=None):
        super().__init__(env._player, resolution=1, threshold_before=7.5, threshold_after=5.)

        # self.params = {'drive': (100, 'noise'), 'noise': (10, 'drive')}
        self.params = {'drive': (100, 'drive')}

        self.steps = 0
        self.state = 'drive'
        self.noise_steer = 0
        self.last_throttle = 0
        self.noise_func = noise if noise else lambda: np.random.uniform(-0.25, 0.25)

        self.speed_control = PIDController(K_P=0.5, K_I=0.5/20, K_D=0.1)
        self.turn_control = PIDController(K_P=0.75, K_I=1.0/20, K_D=0.0)

    def run_step(self, observations):
        self.steps += 1

        last_status = self.state
        num_steps, next_state = self.params[self.state]
        real_control = super().run_step(observations)
        real_control.throttle *= max((1.0 - abs(real_control.steer)), 0.25)

        control = carla.VehicleControl()
        control.manual_gear_shift = False

        if self.state == 'noise':
            control.steer = self.noise_steer
            control.throttle = self.last_throttle
        else:
            control.steer = real_control.steer
            control.throttle = real_control.throttle
            control.brake = real_control.brake

        if self.steps == num_steps:
            self.steps = 0
            self.state = next_state
            self.noise_steer = self.noise_func()
            self.last_throttle = control.throttle

        self.debug = {
                'waypoint': (self.waypoint.x, self.waypoint.y, self.waypoint.z),
                'vehicle': (self.vehicle.x, self.vehicle.y, self.vehicle.z)
                }

        return control, self.road_option, last_status, real_control


def get_episode(env, params):
    data = list()
    progress = tqdm.tqdm(range(params.frames_per_episode), desc='Frame')
    start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
    env_params = {
            'weather': np.random.choice(list(cu.TRAIN_WEATHERS.keys())),
            'start': start,
            'target': target,
            'n_pedestrians': params.n_pedestrians,
            'n_vehicles': params.n_vehicles,
            }

    env.init(**env_params)
    env.success_dist = 5.0

    agent = NoisyAgent(env)
    agent.set_route(env._start_pose.location, env._target_pose.location)

    # Real loop.
    while len(data) < params.frames_per_episode and not env.is_success() and not env.collided:
        for _ in range(params.frame_skip):
            env.tick()

            observations = env.get_observations()
            control, command, last_status, real_control = agent.run_step(observations)
            agent_debug = agent.debug
            env.apply_control(control)

            observations['command'] = command
            observations['control'] = control
            observations['real_control'] = real_control

            if not params.nodisplay:
                _debug(observations, agent_debug)

        observations['control'] = real_control
        processed = cu.process(observations)

        data.append(processed)

        progress.update(1)

    progress.close()

    if (not env.is_success() and not env.collided) or len(data) < 500:
        return None

    return data


def main(params):

    save_dir = Path(params.dataset_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    for i in tqdm.tqdm(range(params.n_episodes), desc='Episode'):
        with make_suite('FullTown01-v1', port=params.port, planner=params.planner) as env:
            filepath = save_dir.joinpath('%03d' % i)

            if filepath.exists():
                continue

            data = None

            while data is None:
                data = get_episode(env, params)

            lmdb_env = lmdb.open(str(filepath), map_size=int(1e10))
            n = len(data)

            with lmdb_env.begin(write=True) as txn:
                txn.put('len'.encode(), str(n).encode())

                for i, x in enumerate(data):
                    txn.put(
                            ('rgb_%04d' % i).encode(),
                            np.ascontiguousarray(x['rgb']).astype(np.uint8))
                    txn.put(
                            ('birdview_%04d' % i).encode(),
                            np.ascontiguousarray(x['birdview']).astype(np.uint8))
                    txn.put(
                            ('measurements_%04d' % i).encode(),
                            np.ascontiguousarray(x['measurements']).astype(np.float32))
                    txn.put(
                            ('control_%04d' % i).encode(),
                            np.ascontiguousarray(x['control']).astype(np.float32))

            total += len(data)

    print('Total frames: %d' % total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner', type=str, choices=['old', 'new'], default='new')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--n_vehicles', type=int, default=100)
    parser.add_argument('--n_pedestrians', type=int, default=250)
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--frames_per_episode', type=int, default=4000)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--nodisplay', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=2000)

    params = parser.parse_args()

    main(params)
