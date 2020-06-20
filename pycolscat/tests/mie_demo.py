import numpy as np
import matplotlib.pylab as plt
from pycolscat.mie import Mie
from matplotlib import rcParams

# parameters for plots
params = {'font.size': 9,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'lines.markersize': 3}
rcParams.update(params)

# Compare to Bohren & Huffman page 115
n_p = 1.33
n_m = 1.00
wavelength = 0.55e-6  # [m] all units SI
a_p = 1.5 * wavelength / np.pi

p1 = Mie(a_p, n_p, n_m, wavelength)

print('nmax = {}'.format(p1.nmax))
print('x = {}'.format(p1.x))
print('mc = {}'.format(p1.mc))

Qext = p1.qext()
Qscat = p1.qscat()
Qtrans = p1.qtrans()

n_angles = 181
theta_deg = np.linspace(0., float(n_angles - 1), n_angles)
theta_rad = np.deg2rad(theta_deg)
f1 = np.zeros(n_angles, dtype=float)
f2 = np.zeros(n_angles, dtype=float)
for i in range(n_angles):
    f1[i], f2[i] = p1.form_12(theta_rad[i])

fig, ax = plt.subplots()
ax.semilogy(theta_deg, f1, label=r'$|S_1|^2$')
ax.semilogy(theta_deg, f2, label=r'$|S_2|^2$')
ax.grid()
ax.set_xticks(range(0, 181, 30))
ax.set_xlim(0., 180.)
ax.set_xlabel('scattering angle [degrees]')
ax.legend()
ax.text(0.98, 0.73, '$Q_{{ext}} = {0:0.3f}$\n$Q_{{scat}} = {1:0.3f}$\n$Q_{{trans}} = {2:0.3f}$'
        .format(Qext, Qscat, Qtrans), ha='right', va='top', transform=ax.transAxes)
ax.text(0.98, 0.57, '$x = {0:0.3f}$\nRe$(m) = {1:0.3f}$\nIm$(m) = {2:0.3f}$'.format(p1.x, p1.mc.real, p1.mc.imag),
        ha='right', va='top', transform=ax.transAxes)
ax.set_title('Compare to Bohren & Huffman Fig 4.9, page 115')

plt.savefig('./plots/mie_demo.pdf')
plt.show()
