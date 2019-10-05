from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
import time

import bird_view.utils.bz_utils as bzu
import bird_view.utils.carla_utils as cu

from bird_view.models.common import crop_birdview


def _paint(observations, control, diagnostic, debug, env, show=False):
    import cv2
    import numpy as np
        

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    birdview = cu.visualize_birdview(observations['birdview'])
    birdview = crop_birdview(birdview)

    if 'big_cam' in observations:
        canvas = np.uint8(observations['big_cam']).copy()
        rgb = np.uint8(observations['rgb']).copy()
    else:
        canvas = np.uint8(observations['rgb']).copy()

    def _stick_together(a, b, axis=1):

        if axis == 1:
            h = min(a.shape[0], b.shape[0])

            r1 = h / a.shape[0]
            r2 = h / b.shape[0]
        
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
    
            return np.concatenate([a, b], 1)
            
        else:
            h = min(a.shape[1], b.shape[1])
            
            r1 = h / a.shape[1]
            r2 = h / b.shape[1]
        
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
    
            return np.concatenate([a, b], 0)

    def _write(text, i, j, canvas=canvas, fontsize=0.4):
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9+1)]
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, WHITE, 1)
                
    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(observations['command'], '???')
            
    if 'big_cam' in observations:
        fontsize = 0.8
    else:
        fontsize = 0.4

    _write('Command: ' + _command, 1, 0, fontsize=fontsize)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0, fontsize=fontsize)

    _write('Steer: %.2f' % control.steer, 4, 0, fontsize=fontsize)
    _write('Throttle: %.2f' % control.throttle, 5, 0, fontsize=fontsize)
    _write('Brake: %.1f' % control.brake, 6, 0, fontsize=fontsize)

    _write('Collided: %s' % diagnostic['collided'], 1, 6, fontsize=fontsize)
    _write('Invaded: %s' % diagnostic['invaded'], 2, 6, fontsize=fontsize)
    _write('Lights Ran: %d/%d' % (env.traffic_tracker.total_lights_ran, env.traffic_tracker.total_lights), 3, 6, fontsize=fontsize)
    _write('Goal: %.1f' % diagnostic['distance_to_goal'], 4, 6, fontsize=fontsize)

    _write('Time: %d' % env._tick, 5, 6, fontsize=fontsize)
    _write('FPS: %.2f' % (env._tick / (diagnostic['wall'])), 6, 6, fontsize=fontsize)

    for x, y in debug.get('locations', []):
        x = int(X - x / 2.0 * CROP_SIZE)
        y = int(Y + y / 2.0 * CROP_SIZE)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_world', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED
    
    for x, y in debug.get('locations_birdview', []):
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED       
 
    for x, y in debug.get('locations_pixel', []):
        S = R // 2
        if 'big_cam' in observations:
            rgb[y-S:y+S+1,x-S:x+S+1] = RED
        else:
            canvas[y-S:y+S+1,x-S:x+S+1] = RED
        
    for x, y in debug.get('curve', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        try:
            birdview[x,y] = [155, 0, 155]
        except:
            pass

    if 'target' in debug:
        x, y = debug['target'][:2]
        x = int(X - x * 4)
        y = int(Y + y * 4)
        birdview[x-R:x+R+1,y-R:y+R+1] = [0, 155, 155]

    ox, oy = observations['orientation']
    rot = np.array([
        [ox, oy],
        [-oy, ox]])
    u = observations['node'] - observations['position'][:2]
    v = observations['next'] - observations['position'][:2]
    u = rot.dot(u)
    x, y = u
    x = int(X - x * 4)
    y = int(Y + y * 4)
    v = rot.dot(v)
    x, y = v
    x = int(X - x * 4)
    y = int(Y + y * 4)

    if 'big_cam' in observations:
        _write('Network input/output', 1, 0, canvas=rgb)
        _write('Projected output', 1, 0, canvas=birdview)
        full = _stick_together(rgb, birdview)
    else:
        full = _stick_together(canvas, birdview)

    if 'image' in debug:
        full = _stick_together(full, cu.visualize_predicted_birdview(debug['image'], 0.01))
        
    if 'big_cam' in observations:
        full = _stick_together(canvas, full, axis=0)
    
    if show:
        bzu.show_image('canvas', full)
    bzu.add_to_video(full)


def run_single(env, weather, start, target, agent_maker, seed, autopilot, show=False):
    # HACK: deterministic vehicle spawns.
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])

    if not autopilot:
        agent = agent_maker()
    else:
        agent = agent_maker(env._player, resolution=1, threshold=7.5)
        agent.set_route(env._start_pose.location, env._target_pose.location)

    diagnostics = list()
    result = {
            'weather': weather,
            'start': start, 'target': target,
            'success': None, 't': None,
            'total_lights_ran': None,
            'total_lights': None,
            'collided': None,
            }

    while env.tick():
        observations = env.get_observations()
        control = agent.run_step(observations)
        diagnostic = env.apply_control(control)

        _paint(observations, control, diagnostic, agent.debug, env, show=show)

        diagnostic.pop('viz_img')
        diagnostics.append(diagnostic)

        if env.is_failure() or env.is_success():
            result['success'] = env.is_success()
            result['total_lights_ran'] = env.traffic_tracker.total_lights_ran
            result['total_lights'] = env.traffic_tracker.total_lights
            result['collided'] = env.collided
            result['t'] = env._tick
            break

    return result, diagnostics


def run_benchmark(agent_maker, env, benchmark_dir, seed, autopilot, resume, max_run=5, show=False):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / 'summary.csv'
    diagnostics_dir = benchmark_dir / 'diagnostics'
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    for weather, (start, target), run_name in tqdm.tqdm(env.all_tasks, total=total):
        if resume and len(summary) > 0 and ((summary['start'] == start) \
                       & (summary['target'] == target) \
                       & (summary['weather'] == weather)).any():
            print (weather, start, target)
            continue


        diagnostics_csv = str(diagnostics_dir / ('%s.csv' % run_name))

        bzu.init_video(save_dir=str(benchmark_dir / 'videos'), save_path=run_name)

        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed, autopilot, show=show)

        summary = summary.append(result, ignore_index=True)

        # Do this every timestep just in case.
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)

        num_run += 1

        if num_run >= max_run:
            break
