import numpy as np
import requests
import json
import time

start = time.time()
src = '/lustrefs/disk/project/lt900038-ai23tn/frb_data/train/B0531+21_2020-05-31-11_36_46_0004095.npy'
npy = np.load(src)
num_steps = np.ceil(npy.shape[0] / 256) #1024

for idx in range(int(num_steps)):
    assert time.time() - start < 600. , 'time out'
    data = {'arr': json.dumps(npy[idx * 256: (idx+1) * 256].T.tolist())}
    response = requests.post(
        'http://127.0.0.1:8911/eval',
        json=data
    )
    cls = response.text
    assert len(cls) == 3, 'The output should be 3 binray integers like 000, 101, 001'
    assert cls.isnumeric(), 'The output should be 3 binray integers like 000, 101, 001'
    cls = np.array(list(cls)).astype(int)
    assert not (cls > 1).any(), 'The output should not be higher than 1'
    assert not (cls < 0).any(), 'The output should not be lower than 0'
    print(response.text)
    # break

print(time.time() - start , "seconds")

