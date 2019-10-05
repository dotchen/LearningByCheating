"""
usage -

python3 parse.py --run_dir . --threshold 1.0 --town 1

or

python3 parse.py --run_dir . --threshold 1.0 --debug
"""
import argparse

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm


class Vector2(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __truediv__(self, c):
        return Vector2(self.x / c, self.y / c)

    def __add__(self, v):
        return Vector2(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return Vector2(self.x - v.x, self.y - v.y)

    def dot(self, v):
        return self.x * v.x + self.y * v.y

    def cross(self, v):
        return self.x * v.y - self.y * v.x

    def norm(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self):
        return self / (self.norm() + 1e-8)


def get_collision(p1, p2, lines):
    """
    line 1: p + t r
    line 2: q + u s
    """
    p = p1
    r = p2 - p1

    for a, b in lines:
        q = a
        s = b - a

        r_cross_s = r.cross(s)
        q_minus_p = q - p

        if abs(r_cross_s) < 1e-3:
            continue

        t = q_minus_p.cross(s) / r_cross_s
        u = q_minus_p.cross(r) / r_cross_s

        if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
            return True

    return False


def parse(df, lights):
    n = len(df)
    t = np.array(list(range(n)))
    traveled = 0.0

    broken_t = list()
    broken = list()

    for i in range(1, n):
        a = Vector2(df['x'][i-1], df['y'][i-1])
        b = Vector2(df['x'][i], df['y'][i])
        traveled += (a - b).norm()

        if not df['is_light_red'][i-1]:
            continue

        if get_collision(a, b, lights):
            broken_t.append(i)
            broken.append(df['is_light_red'][i-1])

    if args.debug:
        plt.plot(t, is_light_red)
        plt.plot(t, speed)
        plt.plot(broken_t, broken, 'r.')
        plt.show()

    return broken, traveled


def get_town(town):
    lights = Path('light_town%s.txt' % town).read_text().strip().split('\n')
    lights = [tuple(map(float, x.split())) for x in lights]
    lights = np.array(lights)

    alpha = 10.0
    lines = list()

    for x, y in lights:
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx = x + alpha * dx
            ny = y + alpha * dy

            lines.append((Vector2(x, y), Vector2(nx, ny)))

    return lines


def main(run_dir):
    result = list()

    for path in sorted(run_dir.glob('*/summary.csv')):
        summary_csv = pd.read_csv(str(path))
        total = len(summary_csv)
        lights = get_town(1 if 'Town01' in str(path) else 2)

        n_infractions = list()
        distances = list()

        for _, row in summary_csv.iterrows():
            weather = row['weather']
            start = row['start']
            target = row['target']
            run_csv = 'w%s_s%s_t%s.csv' % (weather, start, target)

            diag = pd.read_csv(str(path.parent / 'diagnostics' / run_csv))

            crosses, dist = parse(diag, lights)

            n_infractions.append(sum(crosses))
            distances.append(dist)

        print(path, total)
        print('%s infractions total.' % np.sum(n_infractions))
        print('%s total dist' % np.sum(distances))

        result.append({
            'suite': path.parent.stem,
            'infractions': sum(n_infractions),
            'per_10km': sum(n_infractions) / (np.sum(distances) / 10000),
            'distances': sum(distances)})

        pd.DataFrame(result).to_csv('%s/lights.csv' % path.parent.parent, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, type=str)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    main(Path(args.run_dir))
