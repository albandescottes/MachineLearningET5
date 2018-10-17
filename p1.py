import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
       
train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

image_idx = 0
print('Label:', train_data['y'][image_idx])
plt.imshow(train_data['X'][:, :, :, image_idx])
plt.show()