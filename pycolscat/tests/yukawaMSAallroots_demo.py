import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as kB
import pycolscat.yukawaMSAallroots as ykwaAll

# Assign values to parameters for input to yukawaMSA.msaHP
sigma = 5.e-9  # [m] hard-core diameter
phi = 0.3  # volume fraction
zeff = 20.  # effective charge
debye_length = 2.5e-9  # [m] Debye screening length
epsilon = 78.3  # dielectric constant of solvent (water)
tempC = 25.  # [°C] temperature in degrees Celcius

# instantiate yukawaMSA.msaHP class
p0 = ykwaAll.msaHP(phi, sigma, zeff, debye_length, epsilon, tempC)
i_root = p0.choose_root()
print("Physically relavant root = {0:d}".format(i_root))

# create independent variable arrays for yukawaMSA.msaHP class methods
r = np.linspace(0.01, 4., 400)
r_pot = np.linspace(1.01, 4., 100)  # dimensionless distances for plot of potential
K = np.linspace(0.01, 20., 200)

# report class attributes
print('\ntemperature = {}°C'.format(p0.tempC))
print('temperature = {} K'.format(p0.tempK))
print('phi = {}'.format(p0.phi))
print('kappa sigma = {}'.format(p0.kapsig))
print('zeff = {0:0.1f}'.format(p0.zeff))
print('epsilon = {}'.format(p0.epsilon))
print('psi0 = {0:0.3g} millivolts'.format(1000 * p0.psi0))
print('contact potential = {0:0.3g} kT'.format(p0.contact_potential))

# print out values of A, B, C, & F for all real roots
print('\nA = {}'.format(p0.A))
print('B = {}'.format(p0.B))
print('C = {}'.format(p0.C))
print('F = {}'.format(p0.F))

lineW = [0.5] * p0.F.size
lineW[i_root] = 2.  # Make traces thicker for physically relevant root than for other traces
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].plot(r_pot, p0.potential(r_pot * p0.sigma) / (kB * p0.tempK))
ax[0, 0].set_xlim(left=0.)
ax[0, 0].set_ylim(0., 10.)
ax[0, 0].axvline(x=1, color='gray', lw=0.5, dashes=(10, 4), zorder=-1)
ax[0, 0].set_xlabel(r'$r/\sigma$')
ax[0, 0].set_ylabel(r'$U(r)/k_BT$')
for i in range(p0.F.size):
    ax[1, 0].plot(r, p0.c(r, p0.A[i], p0.B[i], p0.C[i], p0.F[i]), lw=lineW[i], label=str(i))
    ax[1, 1].plot(K, p0.s(K, p0.A[i], p0.B[i], p0.C[i], p0.F[i]), lw=lineW[i], label=str(i))
    ax[0, 1].plot(r, p0.g(r, p0.A[i], p0.B[i], p0.C[i], p0.F[i]), lw=lineW[i], label=str(i))
ax[1, 0].set_xlabel(r'$r/\sigma$')
ax[1, 0].set_ylabel(r'$c(x)$, direct correlation function')
ax[1, 0].set_xlim(left=0.)
ax[1, 0].axvline(x=1, color='gray', lw=0.5, dashes=(10, 4), zorder=-1)

ax[1, 1].set_xlabel(r'$q\sigma$')
ax[1, 1].set_ylabel(r'$S(q)$, static structure factor')
ax[1, 1].axhline(y=0, color='gray', lw=0.5, zorder=-1)
ax[1, 1].axhline(y=1, color='gray', lw=0.5, zorder=-1, dashes=(10, 4))
ax[1, 1].axvline(x=2 * np.pi, color='gray', lw=0.5, dashes=(10, 4), zorder=-1)
info = '$\\phi = {0:0.2f}$\n$\\sigma = {1:0.0f}$ nm\n$\\kappa\\sigma = {2:0.2f}$\n$z_{{eff}} = {3:0.0f}$'
info += '\n$\epsilon = {4:0.3g}$\n$\psi_0  = {5:0.2f}$ mV\n$U(\sigma) = {6:0.3g}~k_BT$'
ax[0, 0].text(0.98, 0.98, info.format(p0.phi, 1.e9 * p0.sigma, p0.kapsig, p0.zeff, p0.epsilon,
                                      1000. * p0.psi0, p0.contact_potential),
              ha='right', va='top', transform=ax[0, 0].transAxes)

ax[0, 1].set_xlabel(r'$r/\sigma$')
ax[0, 1].set_ylabel(r'$g(r)$, radial distribution function')
ax[0, 1].axhline(y=0, color='gray', lw=0.5, zorder=-1)
ax[0, 1].axhline(y=1, color='gray', lw=0.5, zorder=-1, dashes=(10, 4))
ax[0, 1].axvline(x=1, color='gray', lw=0.5, zorder=-1, dashes=(10, 4))
ax[0, 1].text(0.98, 0.02, 'Root {0:d} is physically relevant'.format(i_root),
              ha='right', va='bottom', transform=ax[0, 1].transAxes)
ax[0, 1].legend(loc='upper right', title='        root')

plt.subplots_adjust(wspace=0.24)

plt.savefig('./plots/yukawaMSAallroots_demo.pdf')
plt.show()
