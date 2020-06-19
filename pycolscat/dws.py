# -*- coding: utf-8 -*-
# Copyright 2019-2020, David J. Pine
#
# This package is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This package is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this package. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy.integrate as spint
from pycolscat.mie import Mie
from pycolscat.diffusion import hydro

__all__ = ['ref_index_MG', 'ref_index_BG', 'qeff', 'mfp', 'diff_eff']


def _intgrnd_FS(theta, mie_inst, s, kam, *sparams):
    """
    Integrand for calculating the mean free path using
    Percus-Yevick hard-sphere structure factor
    qd : wavevector times the particle diameter
    phi : volume fraction of spheres
    mie_inst : instance of Mie class from mie_uniform_sphere
    """
    qd = 4. * kam * np.sin(0.5 * theta)
    return np.sin(theta) * mie_inst.form(theta) * s(qd, *sparams)


def _intgrnd_FStrans(theta, mie_inst, s, kam, *sparams):
    """
    Integrand for calculating the transport mean free path
    using Percus-Yevick hard-sphere structure factor
    qd : wavevector times the particle diameter
    phi : volume fraction of spheres
    mie_inst : instance of Mie class from mie_uniform_sphere
    """
    qd = 4. * kam * np.sin(0.5 * theta)
    return np.sin(theta) * (1. - np.cos(theta)) * mie_inst.form(theta) * s(qd, *sparams)


def _intgrnd_FH1cos(theta, mie_inst, g, phi, kam, *gparams):
    """
    Integrand used for calculating effective collective diffusion
    coefficient using Percus-Yevick hard-sphere structure factor
    theta : theta array in radians
    phi : volume fraction of spheres
    ka : wavevector in effective medium times particle radius
    mie_inst : instance of Mie class from mie_uniform_sphere
    """
    qd = 4. * kam * np.sin(0.5 * theta)
    return np.sin(theta) * (1. - np.cos(theta)) * mie_inst.form(theta) * hydro(qd, phi, g, *gparams)[0]


def ref_index_MG(n_m, n_p, phi):
    """
    Returns the Maxwell-Garnet effective refractive index of two-component material
    """
    e_m = n_m * n_m
    e_p = n_p * n_p
    e_d = e_p - e_m
    e_MG = e_m * (3. * e_m + (1. + 2. * phi) * e_d) / (3. * e_m + (1. - phi) * e_d)
    return np.sqrt(e_MG)


def ref_index_BG(n_m, n_p, phi):
    """
    Returns the Bruggeman effective refractive index of two-component material
    """
    e_m = n_m * n_m
    e_p = n_p * n_p
    b = (2. - 3. * phi) * e_m + (3. * phi - 1.) * e_p
    e_BG = 0.25 * (b + np.sqrt(8. * e_m * e_p + b * b))
    return np.sqrt(e_BG)


def qeff(mie_inst, s, kam, sparams):
    """Returns scattering efficiency factors for Mie + structure"""
    intgrl1 = spint.quad(_intgrnd_FS, 0., np.pi, args=(mie_inst, s, kam, *sparams))
    intgrl2 = spint.quad(_intgrnd_FStrans, 0., np.pi, args=(mie_inst, s, kam, *sparams))
    ka2 = mie_inst.x * mie_inst.x
    qsca = intgrl1[0] / ka2
    qtra = intgrl2[0] / ka2
    return qsca, qtra


def mfp(mie_inst, phi, s, kam, sparams):
    """Returns mean free paths for Mie + structure"""
    qsca, qtra = qeff(mie_inst, s, kam, sparams)
    smfp = 4. * mie_inst.a_p / (3. * phi * qsca)
    tmfp = 4. * mie_inst.a_p / (3. * phi * qtra)
    return smfp, tmfp


def diff_eff(mie_inst, phi, s, kam, sparams, g, gparams):
    """Returns effective diffusion coefficient"""
    intgrl2 = spint.quad(_intgrnd_FStrans, 0., np.pi, args=(mie_inst, s, kam, *sparams))
    intgrl3 = spint.quad(_intgrnd_FH1cos, 0., np.pi, args=(mie_inst, g, phi, kam, *gparams))
    diff = intgrl3[0] / intgrl2[0]
    decayrate = kam * kam * diff
    return diff, decayrate
