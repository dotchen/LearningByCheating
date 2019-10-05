import re
import numpy as np
import pandas as pd
from terminaltables import DoubleTable
from pathlib import Path


def main(path_name):

	performance = dict()

	path = Path(path_name)
	for summary_path in path.glob('*/summary.csv'):
		name = summary_path.parent.name
		match = re.search('^(?P<suite_name>.*Town.*-v[0-9]+.*)_seed(?P<seed>[0-9]+)', name)
		suite_name = match.group('suite_name')
		seed = match.group('seed')

		summary = pd.read_csv(summary_path)

		if suite_name not in performance:
			performance[suite_name] = dict()

		performance[suite_name][seed] = (summary['success'].sum(), len(summary))

	table_data = []
	for suite_name, seeds in performance.items():

		successes, totals = np.array(list(zip(*seeds.values())))
		rates = successes / totals * 100

		if len(seeds) > 1:
			table_data.append([suite_name, "%.1f ± %.1f"%(np.mean(rates), np.std(rates, ddof=1)), "%d/%d"%(sum(successes),sum(totals)), ','.join(sorted(seeds.keys()))])
		else:
			table_data.append([suite_name, "%d"%np.mean(rates), "%d/%d"%(sum(successes),sum(totals)), ','.join(sorted(seeds.keys()))])

	table_data = sorted(table_data, key=lambda row: row[0])
	table_data = [('Suite Name', 'Success Rate', 'Total', 'Seeds')] + table_data
	table = DoubleTable(table_data, "Performance of %s"%path.name)
	print(table.table)



if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('path', help='path of benchmark folder')

	args = parser.parse_args()
	main(args.path)