import gzip
f = gzip.open('mnist/train-labels-idx1-ubyte.gz','r')

import numpy as np
#f.read(8)
buf = f.read()
data = np.frombuffer(buf, dtype=np.uint8)#.astype(np.float32)

image_size = 28
#data = data.reshape(-1, image_size, image_size)

print(data.shape, data[0:20])