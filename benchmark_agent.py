import argparse
import time

from pathlib import Path

from benchmark import make_suite, get_suites, ALL_SUITES
from benchmark.run_benchmark import run_benchmark

import bird_view.utils.bz_utils as bzu


def _agent_factory_hack(model_path, config, autopilot):
    """
    These imports before carla.Client() cause seg faults...
    """
    from bird_view.models.roaming import RoamingAgentMine

    if autopilot:
        return RoamingAgentMine

    import torch

    from bird_view.models import baseline
    from bird_view.models import birdview
    from bird_view.models import image

    model_args = config['model_args']
    model_name = model_args['model']
    model_to_class = {
            'birdview_dian': (birdview.BirdViewPolicyModelSS, birdview.BirdViewAgent),
            'image_ss': (image.ImagePolicyModelSS, image.ImageAgent),
            }

    model_class, agent_class = model_to_class[model_name]

    model = model_class(**config['model_args'])
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()

    agent_args = config.get('agent_args', dict())
    agent_args['model'] = model

    return lambda: agent_class(**agent_args)


def run(model_path, port, suite, big_cam, seed, autopilot, resume, max_run=10, show=False):
    log_dir = model_path.parent
    config = bzu.load_json(str(log_dir / 'config.json'))

    total_time = 0.0

    for suite_name in get_suites(suite):
        tick = time.time()

        benchmark_dir = log_dir / 'benchmark' / model_path.stem / ('%s_seed%d' % (suite_name, seed))
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        with make_suite(suite_name, port=port, big_cam=big_cam) as env:
            agent_maker = _agent_factory_hack(model_path, config, autopilot)

            run_benchmark(agent_maker, env, benchmark_dir, seed, autopilot, resume, max_run=max_run, show=show)

        elapsed = time.time() - tick
        total_time += elapsed

        print('%s: %.3f hours.' % (suite_name, elapsed / 3600))

    print('Total time: %.3f hours.' % (total_time / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--suite', choices=ALL_SUITES, default='town1')
    parser.add_argument('--big_cam', action='store_true')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--autopilot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-run', type=int, default=3)

    args = parser.parse_args()

    run(Path(args.model_path), args.port, args.suite, args.big_cam, args.seed, args.autopilot, args.resume, max_run=args.max_run, show=args.show)
