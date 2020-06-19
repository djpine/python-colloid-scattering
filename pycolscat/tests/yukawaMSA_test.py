# -*- coding: utf-8 -*-
# Copyright 2019-2020, David J. Pine
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as kB
import pycolscat.yukawa as ykwa

# Assign values to parameters for input to yukawaMSA.msaHP
sigma = 5.e-9  # [m] hard-core diameter
phi = 0.3  # volume fraction
zeff = 20.  # effective charge
debye_length = 2.5e-9  # [m] Debye screening length
epsilon = 78.3  # dielectric constant of solvent (water)
tempC = 25.  # [°C] temperature in degrees Celcius

# instantiate yukawaMSA.msaHP class
p0 = ykwa.msaHP(phi, sigma, zeff, debye_length, epsilon, tempC)

# create independent variable arrays for yukawaMSA.msaHP class methods
r = np.linspace(0.01, 4., 400)
r_pot = np.linspace(1.01, 4., 100)  # dimensionless distances for plot of potential
K = np.linspace(0.01, 20., 200)

# calculate interaction potential, direct correlation function, structure factor
# and pair correlation function using yukawaMSA.msaHP class methods
interaction_potential = p0.potential(r_pot * p0.sigma) / (kB * p0.tempK)
direct_correlation_func = p0.c(r)
structure_factor = p0.s(K)
pair_correlation_func = p0.g(r)

# report class attributes
print('temperature = {}°C'.format(p0.tempC))
print('temperature = {} K'.format(p0.tempK))
print('phi = {}'.format(p0.phi))
print('kappa sigma = {}'.format(p0.kapsig))
print('zeff = {0:0.1f}'.format(p0.zeff))
print('epsilon = {}'.format(p0.epsilon))
print('psi0 = {0:0.3g} millivolts'.format(1000 * p0.psi0))
print('contact potential = {0:0.3g} kT'.format(p0.contact_potential))

# plot outputs
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].plot(r_pot, interaction_potential)
ax[0, 0].set_xlim(left=0.)
ax[0, 0].set_ylim(0., 10.)
ax[0, 0].axvline(x=1, color='gray', lw=0.5, dashes=(10, 4), zorder=-1)
ax[0, 0].set_xlabel(r'$r/\sigma$')
ax[0, 0].set_ylabel(r'$U(r)/k_BT$')
txt = r'$\phi = {0:0.2f}$'.format(p0.phi)
txt += '\n' + r'$\sigma = {0:0.0f}$ nm'.format(1.e9 * p0.sigma)
txt += '\n' + r'$\kappa\sigma = {0:0.2f}$'.format(p0.kapsig)
txt += '\n' + r'$z_{{eff}} = {0:0.0f}$'.format(p0.kapsig)
txt += '\n' + r'$\epsilon = {0:0.3g}$'.format(p0.epsilon)
txt += '\n' + r'$\psi_0  = {0:0.2f}$ mV'.format(1000. * p0.psi0)
txt += '\n' + r'$U(\sigma) = {0:0.3g}~k_BT$'.format(p0.contact_potential)
ax[0, 0].text(0.98, 0.98, txt, ha='right', va='top', transform=ax[0, 0].transAxes)

ax[1, 0].plot(r, direct_correlation_func)  # class call
ax[1, 1].plot(K, structure_factor)  # class call
ax[0, 1].plot(r, pair_correlation_func)  # class call
ax[1, 0].set_xlabel(r'$r/\sigma$')
ax[1, 0].set_ylabel(r'$c(x)$, direct correlation function')
ax[1, 0].set_xlim(left=0.)
ax[1, 0].axvline(x=1, color='gray', lw=0.5, dashes=(10, 4), zorder=-1)

ax[1, 1].set_xlabel(r'$q\sigma$')
ax[1, 1].set_ylabel(r'$S(q)$, static structure factor')
ax[1, 1].axhline(y=0, color='gray', lw=0.5, zorder=-1)
ax[1, 1].axhline(y=1, color='gray', lw=0.5, zorder=-1, dashes=(10, 4))
ax[1, 1].axvline(x=2 * np.pi, color='gray', lw=0.5, dashes=(10, 4), zorder=-1)

ax[0, 1].set_xlabel(r'$r/\sigma$')
ax[0, 1].set_ylabel(r'$g(r)$, radial distribution function')
ax[0, 1].axhline(y=0, color='gray', lw=0.5, zorder=-1)
ax[0, 1].axhline(y=1, color='gray', lw=0.5, zorder=-1, dashes=(10, 4))
ax[0, 1].axvline(x=1, color='gray', lw=0.5, zorder=-1, dashes=(10, 4))
fig.suptitle('Compare with Fig. 1 in Hayter & Penfold, Mol. Phys. vol. 42, pp. 109-118, (1981)')

plt.subplots_adjust(wspace=0.24)
plt.savefig('./plots/yukawaMSA_test.pdf')
plt.show()
