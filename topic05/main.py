# %%

import matplotlib.pyplot as plt

data = [[0, 128, 255], [64, 192, 32], [255, 128, 0]]
plt.imshow(data, cmap='gray')
plt.show()

# %%
#multi colored
import numpy as np

data = [
        [[255, 0], [128, 64]], 
        [[128, 255], [64, 0]], 
        [[0, 128], [255, 192]]
        ]
data = np.moveaxis(data, 0, -1)
plt.imshow(data)
plt.show()
