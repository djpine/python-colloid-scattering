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


def _intgrnd_FS(theta, mie_inst, s, kam, *sparams):
    """
    Integrand for calculating the mean free path using
    a suitable structure factor to describe the correlations
    betwwn particles
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


def qefc(mie_inst, s, kam, sparams):
    """
    Calculates the scattering and transport efficiency factors for
    light scattering from spherical colloids with interparticle
    correlations characterized by a static structure factor S(q).

    Parameters
    ----------
    mie_inst : instance of the Mie class included in this package
        particle size and refractive index contrast for Mie scattering
        from individual particles are specified at instantiation
    s : function (not a NumPy ufunc)
        static structure factor S(q) for a single value of q.  The
        first argument of s must be qd, the wavevector times the
        particle diameter.
    kam : value of ka, wavevector [in the effective medium for S(q)]
        times the particle radius.  To determine the efficiency
        factors, this routine performs a numerical integration
        over the scattering angles theta. The value of kam is used
        to convert the scattering angles to the wavevectors.  The
        refractive index of the effective medium should be used to
        determine kam.
    sparams : tuple (not an iterable like *params would be)
        Extra arguments of s [after the first argument, which must be
        qd].  Arguments must be written in the same order that they
        appear in s.  Note that if there is only one extra argument
        of s besides qd, say phi, the tuple is written as (phi,) with
        a comma.

    Returns
    -------
    qsca, qtra : floats
        The dimensionless scattering and transport efficiency factors
        for particles at finite volume fractions phi where there are
        significant spatial correlations in the particle positions.
    """
    intgrl1 = spint.quad(_intgrnd_FS, 0., np.pi, args=(mie_inst, s, kam, *sparams))
    intgrl2 = spint.quad(_intgrnd_FStrans, 0., np.pi, args=(mie_inst, s, kam, *sparams))
    ka2 = mie_inst.x * mie_inst.x
    qsca = intgrl1[0] / ka2
    qtra = intgrl2[0] / ka2
    return qsca, qtra


def mfp(mie_inst, phi, s, kam, sparams):
    """
    Calculates the scattering and transport mean free paths for
    light scattering from spherical colloids with interparticle
    correlations characterized by a static structure factor S(q).

    Parameters
    ----------
    mie_inst : instance of the Mie class included in this package
        particle size and refractive index contrast for Mie scattering
        from individual particles are specified at instantiation.
    phi : float
        volume fraction of particles.  Note that this is used only
        to calculate the number density of particles needed to
        determine the mean free paths.  If one of the arguments of
        s is also the volume fraction, that is specified separately
        in the tuple sparams.
    s : function (not a NumPy ufunc)
        static structure factor S(q) for a single value of q.  The
        first argument of s must be qd, the wavevector times the
        particle diameter.
    kam : value of ka, wavevector [in the effective medium for S(q)]
        times the particle radius.  To determine the efficiency
        factors, this routine performs a numerical integration
        over the scattering angles theta. The value of kam is used
        to convert the scattering angles to the wavevectors.  The
        refractive index of the effective medium should be used to
        determine kam.
    sparams : tuple (not an iterable like *params would be)
        Extra arguments of s [after the first argument, which must be
        qd].  Arguments must be written in the same order that they
        appear in s.  Note that if there is only one extra argument
        of s besides qd, say phi, the tuple is written as (phi,) with
        a comma.

    Returns
    -------
    smfp, tmfp : floats [m]
        The scattering and transport mean free paths in meters for
        particles at finite volume fractions phi where there are
        significant spatial correlations in the particle positions.
    """

    qsca, qtra = qefc(mie_inst, s, kam, sparams)
    smfp = 4. * mie_inst.a_p / (3. * phi * qsca)
    tmfp = 4. * mie_inst.a_p / (3. * phi * qtra)
    return smfp, tmfp


def diff_eff(mie_inst, phi, s, kam, sparams, g, gparams):
    """
    Calculates the dimensionless effective diffusion coefficient
    Deff/D0, where D0 is the Stokes-Einstein diffusion coefficient,
    and the dimensionless decay rate (Deff/D0) kam^2.

    Parameters
    ----------
    mie_inst : instance of the Mie class included in this package
        particle size and refractive index contrast for Mie scattering
        from individual particles are specified at instantiation.
    phi : float
        volume fraction of particles.  Note that this is used only
        to calculate the number density of particles needed to
        determine the mean free paths.  If one of the arguments of
        s is also the volume fraction, that is specified separately
        in the tuple sparams.
    s : function (not a NumPy ufunc)
        static structure factor S(q) for a single value of q.  The
        first argument of s must be qd, the wavevector times the
        particle diameter.
    kam : value of ka, wavevector [in the effective medium for S(q)]
        times the particle radius.  To determine the efficiency
        factors, this routine performs a numerical integration
        over the scattering angles theta. The value of kam is used
        to convert the scattering angles to the wavevectors.  The
        refractive index of the effective medium should be used to
        determine kam.
    sparams : tuple (not an iterable like *sparams would be)
        Extra arguments of s [after the first argument, which must be
        qd].  Arguments must be written in the same order that they
        appear in s.  Note that if there is only one extra argument
        of s besides qd, say phi, the tuple is written as (phi,) with
        a comma.
    g : function (not a NumPy ufunc)
        radial distribution function for a single value of q.  The
        first argument of g must be r/sigma, the radial distance from
        a particle center, divided by the particle diameter sigman.
        g should correspond to the same correlations as the static
        structure factor s.
    gparams : tuple (not an iterable like *gparams would be)
        Extra arguments of g [after the first argument, which must be
        qd].  Arguments must be written in the same order that they
        appear in s.  Note that if there is only one extra argument
        of s besides qd, say phi, the tuple is written as (phi,) with
        a comma.
    """
    intgrl2 = spint.quad(_intgrnd_FStrans, 0., np.pi, args=(mie_inst, s, kam, *sparams))
    intgrl3 = spint.quad(_intgrnd_FH1cos, 0., np.pi, args=(mie_inst, g, phi, kam, *gparams))
    diff = intgrl3[0] / intgrl2[0]
    decayrate = kam * kam * diff
    return diff, decayrate
