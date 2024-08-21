import json

import numpy as np
import pandas as pd

if __name__ == '__main__':
    with open('repo/history_twitter_smac.json') as f:
        with open('scripts/twitter_smac_spaces')as s:
            j = json.load(f)
            a = np.array(j['data'])
            df = pd.DataFrame(j['data'])
            em = pd.DataFrame(df['external_metrics'].tolist())
            am = a[em['tps'].argmax()]
            pass
