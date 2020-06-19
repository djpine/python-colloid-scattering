import numpy as np
import matplotlib.pylab as plt
import pycolscat.hardsphere as hs
from matplotlib import rcParams

# parameters for plots
params = {'font.size': 9,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'lines.markersize': 3}
rcParams.update(params)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

phis = [0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.48]
# Plot hard-sphere Percus-Yevick g(r)
r = np.linspace(0, 5, 200)
for phi in phis:
    ax1.plot(r, hs.g_PY(r, phi), label=str(phi))
ax1.axhline(y=1, color='gray', lw=0.5, zorder=-1)
ax1.set_xlabel('$r/d$')
ax1.set_ylabel('$g(r)$')
ax1.legend(loc='upper right', title='$\qquad\phi$')

# Plot hard-sphere Percus-Yevick S(q)
qd = np.linspace(0, 40, 400)
for phi in phis:
    ax2.plot(qd, hs.s_PY(qd, phi), label=str(phi))
ax2.axhline(y=1, color='gray', lw=0.5, zorder=-1)
ax2.axhline(y=2.9, color='gray', lw=0.5, dashes=(4, 3), zorder=-1)
ax2.set_xlabel('$qd$')
ax2.set_ylabel('$S(q)$')
ax2.legend(loc='upper right', title='$\qquad\phi$')

fig.suptitle("Percus-Yevick structure for hard spheres")
plt.savefig('./plots/hardsphere_test.pdf')
plt.show()
