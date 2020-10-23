
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import sklearn.datasets as dsets


matplotlib.use('TkAgg')
matplotlib.rc('font', size=24)

D, labels = dsets.make_circles(n_samples=1000, noise=0.1)

X = D[:,0]
Y = D[:,1]
Z = 1 - np.cos((X + Y)*np.pi/2)

X /= np.sqrt(2)
Z /= np.sqrt(2)

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax2 = fig.add_subplot(gs[0, 1])

ax1.scatter(X, Y, Z, color=[('r' if x == 0 else 'b') for x in labels])
ax1.set_title("Manifold")

ax2.scatter(X, Y, color=[('r' if x == 0 else 'b') for x in labels])
ax2.set_title("Original 2D Dataset")

#plt.show()
plt.savefig("manifold.png", bbox_inches='tight', dpi=300)
