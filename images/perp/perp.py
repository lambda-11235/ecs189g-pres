
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

plt.rc('font', size=32)
fig, axs = plt.subplots(2, 2, figsize=(16,16))
axs = axs.flatten()

npr.seed(1096)

xi = 5
xs = npr.normal(xi, 10, 20)
X = np.linspace(min(xs), max(xs), 1000)

def p(x, si):
    num = np.exp(-(xi - x)**2/si**2)
    den = np.sum(np.exp(-(xi - xs)**2/si**2))
    return num/den

def perp(xs, si):
    pxs = p(xs, si)
    return 2**(-np.sum(pxs*np.maximum(-1000, np.log2(pxs))))


for i, perpTarget in enumerate([3, 5, 10, 15]):
    low = 1.0e-2
    high = 1.0e3
    si = (high + low)/2
    while abs(perp(xs, si) - perpTarget) > 1.0e-2:
        if perp(xs, si) > perpTarget:
            high = si
            si = (si + low)/2
        else:
            low = si
            si = (si + high)/2

    axs[i].plot(X, p(X, si), 'b-', label="Gaussian Curve")
    axs[i].scatter(xs, p(xs, si), edgecolors='r', facecolors='r', label="$x_j$")
    axs[i].scatter(xi, p(xi, si), edgecolors='r', facecolors='none')
    axs[i].plot(xi, 0, 'ro')
    axs[i].plot([xi, xi], [0, p(xi, si)], 'm--', label="$x_i$")

    axs[i].set_title("$\sigma_i = {:.1f}$, Perp($P_i$) = {:.1f}".format(si, perp(xs, si)))
    axs[i].set_xlabel("$x_j$")
    axs[i].set_ylabel("$p_{j|i}$", rotation=0)
    #axs[i].legend()

fig.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig("perp.png", bbox_inches='tight', dpi=150)
#plt.show()
