
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

plt.rc('font', size=16)
fig, ax = plt.subplots(1, 1, figsize=(8,8))

npr.seed(1096)

xi = 5
xs = npr.normal(xi, 2, 10)
X = np.linspace(min(xs), max(xs), 1000)
perpTarget = 5

def p(x, si):
    num = np.exp(-(xi - x)**2/si**2)
    den = np.sum(np.exp(-(xi - xs)**2/si**2))
    return num/den

def perp(xs, si):
    pxs = p(xs, si)
    return 2**(-np.sum(pxs*np.maximum(-1000, np.log2(pxs))))


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

ax.plot(X, p(X, si), 'b-', label="Gaussian Curve")
ax.scatter(xs, p(xs, si), edgecolors='r', facecolors='r', label="$x_j$")
ax.scatter(xi, p(xi, si), edgecolors='r', facecolors='none')
ax.plot(xi, 0, 'ro')
ax.plot([xi, xi], [0, p(xi, si)], 'm--', label="$x_i$")

ax.set_title("$p_{{j|i}}$ Gaussian Curve on 1D Data")
ax.set_xlabel("$x_j$")
ax.set_ylabel("$p_{j|i}$", rotation=0)
#ax.legend()

plt.savefig("gauss.png", bbox_inches='tight', dpi=150)
#plt.show()
