# -*- coding: utf-8 -*-
# Copyright 2019-2020, David J. Pine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# sys.path.append('../transport')
import pycolscat.hardsphere as hs
from pycolscat.mie import Mie

from pycolscat.dws import qeff, mfp, ref_index_BG, diff_eff
from pycolscat.diffusion import dstokesEin

# Parameters for plots
params = {'font.size': 9,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'lines.markersize': 3}
rcParams.update(params)

# Calculation of effective diffusion coefficient (time consuming)
# True: do calculation; False: do not do calculation
hydrocalc = True

# Set parameters (SI for all units)
radius = 0.20e-6  # [m]
n_p = 1.59
n_m = 1.33
wavelength = 632.8e-9  # [m]
phi_range = np.arange(0.025, 0.5, 0.025)
ka0 = 2. * np.pi * radius / wavelength  # in vacuum

tempC = 25.
viscosity = 8.90e-4  # [Pa-s]
D0 = dstokesEin(tempC, viscosity, radius)

# Instantiate Mie, which is needed to calculate mean free paths
# Here we use the actual refractive index of the background
# medium (solvent).  If we used an index averaged over the particles
# as the backgound index, then the Mie scattering would depend on
# volume fraction.
p1 = Mie(radius, n_p, n_m, wavelength)

# Calculate efficiency factors, mean free paths for each phi
# if hydrocalc is True, also calculate effective diffusion
# coefficient and DWS decay rate.
qsca = np.zeros(phi_range.size)
qtra = np.zeros(phi_range.size)
smfp = np.zeros(phi_range.size)
tmfp = np.zeros(phi_range.size)
deff = np.zeros(phi_range.size)
decayrate = np.zeros(phi_range.size)
n_em = np.zeros(phi_range.size)

for i, phi in enumerate(phi_range):
    # Sets index for S(q) -> S(theta)
    n_em[i] = ref_index_BG(n_m, n_p, phi)
    kam = ka0 * n_em[i]
    # Calculate efficiency factors and mean free paths
    qsca[i], qtra[i] = qeff(p1, hs.s_PY, kam, (phi,))
    smfp[i], tmfp[i] = mfp(p1, phi, hs.s_PY, kam, (phi,))
    # Calculate effective diffusion coefficient
    if hydrocalc is True:
        # the arguments for hs.s_PY and G_PY as passed as tuples, not as iterables
        deff[i], decayrate[i] = diff_eff(p1, phi, hs.s_PY, kam, (phi,), hs.g_PY, (phi,))

# Plot results
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(phi_range, qsca, 's', label=r'$Q_{sca}$')
ax[0, 0].plot(phi_range, qtra, 'o', label=r'$Q_{tra}$')
ax[0, 0].set_xlim(0., 0.5)
ax[0, 0].set_ylim(bottom=0.)
ax[0, 0].set_xlabel('volume fraction')
ax[0, 0].set_ylabel('scattering efficiency factors')
txt = r'diameter = {0:0.3g} $\mu$m'.format(2. * radius * 1.e6)
txt += '\n' + r'$\lambda$ = {0:0.3g} nm'.format(2. * radius * 1.e9)
txt += '\n' + r'$n_p$ = {0:0.3g}'.format(n_p)
txt += '\n' + r'$n_n$ = {0:0.3g}'.format(n_m)
ax[0, 0].text(0.98, 0.8, txt, ha='right', va='top', transform=ax[0, 0].transAxes)
ax[0, 0].legend()

ax[0, 1].semilogy(phi_range, smfp, 's', label='mean free path')
ax[0, 1].semilogy(phi_range, tmfp, 'o', label='transport mean free path')
ax[0, 1].set_xlim(0., 0.5)
ax[0, 1].set_xlabel('volume fraction')
ax[0, 1].set_ylabel('mean free paths [m]')
ax[0, 1].legend()

ax[1, 0].plot(phi_range, n_em, '^')
ax[1, 0].set_xlim(0., 0.5)
ax[1, 0].set_xlabel('volume fraction')
ax[1, 0].set_ylabel('effective refractive index')

if hydrocalc is True:
    ax[1, 1].plot(phi_range, deff, '<')
    ax[1, 1].set_xlim(0., 0.5)
    ax[1, 1].set_xlabel('volume fraction')
    txt = r'$D_\mathrm{eff}/D_0 = \langle D \rangle/D_0$, effective diffusion coefficient'
    ax[1, 1].set_ylabel(txt)
    axright = ax[1, 1].twinx()
    axright.plot(phi_range, D0 * decayrate / radius ** 2, 'o', color='C3')
    txt = r'$D_\mathrm{eff}k_\mathrm{em}^2$, decay rate [$s^{-1}$]'
    axright.set_ylabel(txt, color='C3')
    axright.tick_params('y', colors='C3')
    txt = r'$D_0 = {0:0.3g}~\mu$m$^2$/s'.format(D0 * 1.e12)
    ax[1, 1].text(0.98, 0.98, txt, ha='right', va='top', transform=ax[1, 1].transAxes)
    plt.savefig('./plots/dws_test.pdf')

plt.show()
