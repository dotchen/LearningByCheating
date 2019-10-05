import sys

from pathlib import Path

import pandas as pd


log_dir = sys.argv[1]

for model_name in Path(log_dir).glob('*'):
    print(model_name.stem)

    for run_path in sorted(model_name.glob('*/*.csv')):
        run_name = run_path.parent.stem
        csv = pd.read_csv(run_path)

        print(run_name, '%.4f' % csv['success'].mean(), len(csv))

    print()
