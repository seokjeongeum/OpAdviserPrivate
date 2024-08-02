import json

from matplotlib import pyplot as plt

if __name__ == '__main__':
    with open('repo/history_twitter_smac_3.json') as f:
        data = json.load(f)['data']
        xs = [d['configuration']['innodb_io_capacity_max'] for d in data]
        ys = [d['configuration']['innodb_log_file_size'] for d in data]
        zs = [d['configuration']['innodb_thread_concurrency'] for d in data]
        c = [d['external_metrics'].get('tps') for d in data]
        fig, ax = plt.subplots(subplot_kw={
            'projection': '3d',
        })
        ax.scatter(xs, ys, zs, c=c)
        plt.show()
