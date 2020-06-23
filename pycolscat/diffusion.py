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
"""
Functions for determining the diffusion of spherical colloids in liquid
suspensions measured using dynamic light scattering (DLS) and diffusing-wave
spectroscopy (DWS):

H(q) : Hydrodynamic function coefficient as a function of wavevector q
D(q) : Collective diffusion coefficient as a function of wavevector q
D0 : Stokes-Einstein diffusion coefficient measured at infinite dilution

Notes
-----
1. The function hydro, which returns a value of H(q), involves numerical
   integration. It may be possible to speed this up considerably using
   Numba but modifications of hydro are needed to make it compatible with
   Numba's capabilities.

2. The numerical integrator scipy.integrate.quad that is used in this
   routine issues a bunch of warnings about possible problems with the
   convergence of some of the integrals.  Nevertheless, the output is
   in good agreement with the resutss of Snook et al. (see below). It
   may be possible to rewrite the integrals to get faster convergence
   of the numerical integrals.
"""
import numpy as np
import scipy.integrate
from scipy.constants import Boltzmann as kB


def hydro(q, phi, g, *gparams):
    # TODO: refactor and speed up with numba
    """
    Parameters
    ----------
    q : float
        q-vector times particle diameter
    phi : float
        volume fraction
    g : string
        function name of radial distribution function g(r)
        This should be supplied by an external module (such as the hardsphere
        module available in this package)
    *gparams : tuple (iterable)
        extra arguments of function g(r) [such as phi, screening length, etc, as needed]

    Returns
    -------
    H(q): 2-tuple of floats
        Hydrodynamic function H(q), estimated absolute error based on numerical integrations

    Notes
    -----
    In addition to calculating H(q) for any non-negative q, the routine also
    calculates H(infininity) if the the q argument is set to np.inf.

    Reference:
    ----------
    .. [1] I Snook, W van Megan, R J A Tough (SvMT), Diffusion in concentrated
           hard sphere dispersions: Effective two particle mobility tensors,
           J. Chem. Phys. 78, 5825-5836 (1983).
    """
    kappa1 = 0.25 * phi  # Section III.B. p. 5830
    kappa2 = phi * (8.5 + 400. * phi ** 6)  # Section III.B. p. 5830

    def f(r):  # After Eq (33)
        """
        Parameters:
        ----------
        r : numpy.ndarray
            distance from sphere center: same for h(r), intgrnd0, intgrnd1, intgrnd2

        Returns
        -------
        f(r): function f(r) that appears in Eq. 33 of SvMT
        """
        k2r = kappa2 * r
        qr = q * r
        f1a = np.exp(-k2r) / qr
        f1a *= (1. + 1. / k2r) / kappa2 - (1. + 3. / k2r * (1. + 1. / k2r)) / (q * qr)
        return f1a + 1. / (q * k2r * k2r) * (3. / (qr * qr) - 1.)

    def h(r):
        """
        Returns
        -------
        h(r): function h(r) that appears in Eq. 33 of SvMT
        """
        k2r = kappa2 * r
        qqr = q * q * r
        return np.exp(-k2r) * (1. + 3. / k2r * (1. + 1. / k2r)) / qqr - 3. / (qqr * k2r * k2r)

    def intgrnd0(r):  # Eq (34) integrand of first integral
        return r * np.exp(-kappa2 * r) * (g(r, *gparams) - 1.)

    def intgrnd1(r):  # Eq (33) integrand of 1st integral
        qr = q * r
        return (f(r) * np.sin(qr) + h(r) * np.cos(qr)) * (g(r, *gparams) - 1.)

    def intgrnd2(r):  # Eq (33) integrand of 2nd integral
        return np.exp(-kappa1 * r) * g(r, *gparams) / (r * r)

    if q == 0:  # Eq (34)
        int_0 = scipy.integrate.quad(intgrnd0, 0., np.inf)
        int_2 = scipy.integrate.quad(intgrnd2, 0., np.inf)
        hyd = 1. + 12. * phi * int_0[0] - 1.875 * phi * int_2[0]
        hyd_err = 12. * phi * int_0[1] - 1.875 * phi * int_2[1]
    elif q == np.inf:  # Eq (35)
        int_2 = scipy.integrate.quad(intgrnd2, 0., np.inf)
        hyd = 1. - 1.875 * phi * int_2[0]
        hyd_err = 1.875 * phi * int_2[1]
    else:  # Eq (33)
        int_1 = scipy.integrate.quad(intgrnd1, 0., np.inf)
        int_2 = scipy.integrate.quad(intgrnd2, 0., np.inf)
        hyd = 1. - 36. * phi * int_1[0] - 1.875 * phi * int_2[0]
        hyd_err = 36. * phi * int_1[1] - 1.875 * phi * int_2[1]
    return hyd, hyd_err


def H(qd, phi, g, *gparams):
    """
    Inputs
    ______
    qd : numpy.ndarray
        wavevector times the particle diameter
    phi : float
        volume fraction of spheres

    Returns
    -------
    H : numpy.ndarray
        H(q) hydrodynamic function appearing in the collective
        diffusion coefficient D(q) of colloidal spheres
    """
    return np.array(list(map(lambda qq: hydro(qq, phi, g, *gparams)[0], qd)))


def dcoop(q, phi, g, gparams, s, sparams):
    """
    Parameters
    ----------
    q : float [NOT an array]
        q-vector times particle diameter
    phi : float
        volume fraction
    g : function name of radial distribution function g(r)
        This should be supplied by an external module (such as the hard_sphere_structure
        module available in David Pine's GitHub repository)
    gparams : tuple
        extra arguments of function g(r) [such as phi, screening length, etc, as needed]
    s : func
        function name of static structure factor S(q)
        This should be supplied by an external module (such as the hard_sphere_structure
        module available in David Pine's GitHub repository)
    sparams : tuple
        extra arguments of function s(qd) [such as phi, screening length, etc, as needed]

    Returns
    -------
    D(q): Collective diffusion coefficient as a function of q.

    Notes
    -----
    In addition to calculating H(q) for any non-negative q, the routine also
    calculates H(infinity) if the the q argument is set to np.inf.

    Reference:
    ----------
    I Snook, W van Megan, R J A Tough, Diffusion in concentrated
    hard sphere dispersions: Effective two particle mobility tensors,
    J. Chem. Phys. 78, 5825-5836 (1983).
    """
    if q != np.inf:
        return hydro(q, phi, g, *gparams) / s(q, *sparams)
    else:
        return hydro(q, phi, g, *gparams)


def dstokesEin(tempC, viscosity, radius):
    """
    Return Stokes-Einstein diffusion coefficient for a sphere
    """
    return kB * (tempC + 273.15) / (6. * np.pi * viscosity * radius)
