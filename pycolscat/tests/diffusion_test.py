# -*- coding: utf-8 -*-
# Copyright 2019-2020, David J. Pine
"""
Demonstration of hydro, dcoop, and dstokesEin functions to calculate H(q)
and D(q)/D_0 for several different volume fractions. This code takes about 15
seconds for each calculation on a 2018-vintage MacBookPro.  It uses the radial
distribution function g(r) for hard spheres, which is imported from the
structure module available in this package. Because the routine  calculates H(q)
at a single q (and phi) only, the code below uses list-map-lambda to generate an
array of H(q) values for each volume fraction phi. It produces a plot for direct
comparison to Figs. 6 & 8 in Snook et al.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pycolscat.hardsphere as hs
from pycolscat.diffusion import hydro, dcoop, dstokesEin
import time

# parameters for plots
params = {'font.size': 9,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'lines.markersize': 3,
          'legend.borderpad': 0.25,
          'legend.handletextpad': 0.2,
          'legend.handlelength': 1.5}
rcParams.update(params)

# input parameters and test diff_stokesEin function
tempC = 25.
viscosity = 8.90e-4  # [Pa-s]
radius = 0.50e-6  # [m]
D0 = dstokesEin(tempC, viscosity, radius)
txt = 'Stokes-Einstein diffusivity of 1-micron-diameter '
txt += 'sphere at 25°C = {0:0.3g} microns-squared/second'.format(D0 * 1.e12)
print(txt)

phi_range = np.arange(0.05, 0.5, 0.1)
qd = np.linspace(0.1, 20., 50)  # 500 points for a nice smooth curve, but it takes awhile
hq = np.zeros((phi_range.size, qd.size))
hq0 = np.zeros(phi_range.size)
hqinf = np.zeros(phi_range.size)
start = time.time()
for i, phi in enumerate(phi_range):
    hq[i, :] = np.array(list(map(lambda q: hydro(q, phi, hs.g_PY, *(phi,))[0], qd)))
    hq0[i] = hydro(0., phi, hs.g_PY, *(phi,))[0]
    hqinf[i] = hydro(np.inf, phi, hs.g_PY, *(phi,))[0]
end = time.time()
print("Executed in {0:0.2f} seconds".format(end - start))

start = time.time()
dcq = np.zeros((phi_range.size, qd.size))
dcq0 = np.zeros(phi_range.size)
dcqinf = np.zeros(phi_range.size)
for i, phi in enumerate(phi_range):
    dcq[i, :] = np.array(list(map(lambda q: dcoop(q, phi, hs.g_PY, (phi,), hs.s_PY, (phi,))[0], qd)))
    dcq0[i] = dcoop(0, phi, hs.g_PY, (phi,), hs.s_PY, (phi,))[0]
    dcqinf[i] = dcoop(np.inf, phi, hs.g_PY, (phi,), hs.s_PY, *(phi,))[0]
end = time.time()
print("Executed in {0:0.2f} seconds".format(end - start))

plt.close('all')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for i, phi in enumerate(phi_range):
    ax1.plot(qd, hq[i, :])
    ax1.text(16., hq[i, -1] + 0.02, "$\phi = {0:0.2f}$".format(phi))
ax1.set_prop_cycle(None)
for i in range(phi_range.size):
    ax1.plot(0., hq0[i], 'o')
ax1.set_prop_cycle(None)
for i in range(phi_range.size):
    ax1.plot(qd[-1] + 0.5, hqinf[i], 'o')
ax1.set_ylim(0., 1.)
ax1.set_ylabel("$H(q)$")

for i, phi in enumerate(phi_range):
    ax2.plot(qd, 1. / dcq[i, :])
    ax2.text(16., 1. / dcq[i, -1] + 0.03, "$\phi = {0:0.2f}$".format(phi))
ax2.set_prop_cycle(None)
for i in range(phi_range.size):
    ax2.plot(0, 1. / dcq0[i], 'o')
ax2.set_prop_cycle(None)
for i in range(phi_range.size):
    ax2.plot(qd[-1] + 0.5, 1. / dcqinf[i], 'o')

ax2.set_ylim(0., 10.5)
ax2.set_ylabel("$D_0/D(q) = S(q)/H(q)$")
for ax in [ax1, ax2]:
    ax.set_xlim(0., 20.6)
    ax.set_xlabel("$qd$")

plt.savefig('./plots/diffusion_test.pdf')
plt.show()
