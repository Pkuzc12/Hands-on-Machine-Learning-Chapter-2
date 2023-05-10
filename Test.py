import numpy as np
from zlib import crc32

indice = np.random.permutation(10)

print(crc32(np.int64(3)) & 0xffffffff < 0.1 * 2**32)