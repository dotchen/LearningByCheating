import numpy as np
import tqdm
import carla

from agents.navigation.roaming_agent import RoamingAgent

from bird_view.utils import carla_utils as cu
from bird_view.utils import bz_utils as bzu


TOWN = 'Town01'
TRAIN = [(25,29), (28,24), (99,103), (144,148), (151,147)]
VAL = [(57,39), (50,48), (36,53), (136,79), (22,76)]
PORT = 3000


def world_loop(opts_dict):
    params = {
            'spawn': 15,
            'weather': 'clear_noon',
            'n_vehicles': 0
            }

    with cu.CarlaWrapper(TOWN, cu.VEHICLE_NAME, PORT) as env:
        env.init(**params)
        agent = RoamingAgent(env._player, False, opts_dict)

        # Hack: fill up controller experience.
        for _ in range(30):
            env.tick()
            env.apply_control(agent.run_step()[0])

        for _ in tqdm.tqdm(range(125)):
            env.tick()

            observations = env.get_observations()
            inputs = cu.get_inputs(observations)

            debug = dict()
            control, command = agent.run_step(inputs, debug_info=debug)
            env.apply_control(control)

            observations.update({'control': control, 'command': command})

            processed = cu.process(observations)

            yield debug

            bzu.show_image('rgb', processed['rgb'])
            bzu.show_image('birdview', cu.visualize_birdview(processed['birdview']))


def main():
    import matplotlib.pyplot as plt; plt.ion()

    np.random.seed(0)

    for _ in tqdm.tqdm(range(10000), desc='Trials'):
        desired = list()
        current = list()
        output = list()
        e = list()

        K_P = np.random.uniform(0.5, 2.0)
        K_I = np.random.uniform(0.0, 2.0)
        K_D = np.random.uniform(0.0, 0.05)

        # Best so far.
        # K_P = 1.0
        # K_I = 0.5
        # K_D = 0.0

        opts_dict = {
                'lateral_control_dict': {
                    'K_P': K_P,
                    'K_I': K_I,
                    'K_D': K_D,
                    'dt': 0.1
                    }
                }

        for debug in world_loop(opts_dict):
            for x in [desired, current, output]:
                if len(x) > 500:
                    x.pop(0)

            desired.append(debug['desired'])
            current.append(debug['current'])
            output.append(debug['output'])
            e.append(debug['e'] ** 2)

        name = '%.1f_%.3f_%.3f_%.3f' % (sum(e), K_P, K_I, K_D)

        plt.cla()
        plt.plot(list(range(len(desired))), desired, 'b-')
        plt.plot(list(range(len(current))), current, 'r-')
        plt.plot(list(range(len(output))), output, 'c-')
        plt.savefig('/home/bradyzho/hd_data/images/%s.png' % name)


if __name__ == '__main__':
    main()
