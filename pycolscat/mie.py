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

"""
Calculates Mie coefficients, cross sections, and form factors for spheres.

REFERENCES:
1. H. C. van de Hulst,
   Light Scattering by Small Particles,
   (Dover, New York 1981).
2. W. J. Wiscombe, "Improved Mie scattering algorithms,"
   Appl. Opt. 19, 1505-1509 (1980).
3. C. F. Bohren and D. R. Huffman
   Absorption and Scattering of Light by Small Particles,
   (Wiley, New York 1983).

HISTORY
This code was adapted by D. J. Pine (2020) from Python code by 
Christine Middleton (2012), who adapted it from C code written 
by D. J. Pine (1990)
Copyright (c) 2020 David J. Pine
"""


def mie_coefs_up(x, m, nmax):
    """
    Returns the Mie scattering coefficients for a sphere
    Page numbers refer to Ref. 1 by van de Hulst.
    Warning: Uses upward recurrence to calculate a and b
    coefficients, which is unstable for large size
    parameters x.  Generally works well for particles up
    to several microns in size.
    ...
    Arguments
    ---------
    x : float
        size parameter on page 123
    m : float or complex
        refractive of particle / refractive index of matrix
    nmax: int
        number of coefficients needed in Mie sum
    Returns
    -------
    a : numpy.ndarray
        Mie a coefficients
    b : numpy.ndarray
        Mie b coefficients
    """
    # Complex refractive index of particles / real index of medium
    mc = m + 0.j  # Relative refractive index
    y = mc * x

    psix = np.zeros(nmax + 1, dtype=np.float64)
    dpsix = np.zeros(nmax + 1, dtype=np.float64)
    chix = np.zeros(nmax + 1, dtype=np.float64)
    dchix = np.zeros(nmax + 1, dtype=np.float64)
    psiy = np.zeros(nmax + 1, dtype=np.complex128)
    dpsiy = np.zeros(nmax + 1, dtype=np.complex128)
    a = np.zeros(nmax + 1, dtype=np.complex128)
    b = np.zeros(nmax + 1, dtype=np.complex128)

    snx = np.sin(x)
    csx = np.cos(x)
    sny = np.sin(y)
    csy = np.cos(y)

    for n in range(1, nmax + 1):
        if n == 1:
            psix[0] = snx
            dpsix[0] = csx
            psix[1] = snx / x - csx
            dpsix[1] = psix[0] - psix[1] / x
            chix[0] = csx
            dchix[0] = -snx
            chix[1] = csx / x + snx
            dchix[1] = chix[0] - chix[1] / x
            psiy[0] = sny
            dpsiy[0] = csy
            psiy[1] = sny / y - csy
            dpsiy[1] = psiy[0] - psiy[1] / y
        else:
            amp = float(2 * n - 1)
            psix[n] = amp / x * psix[n - 1] - psix[n - 2]
            dpsix[n] = psix[n - 1] - n / x * psix[n]
            chix[n] = amp / x * chix[n - 1] - chix[n - 2]
            dchix[n] = chix[n - 1] - n / x * chix[n]
            psiy[n] = amp / y * psiy[n - 1] - psiy[n - 2]
            dpsiy[n] = psiy[n - 1] - n / y * psiy[n]
        tp1 = psix[n] * dpsiy[n]
        tp2 = psiy[n] * dpsix[n]
        bt1 = complex(psix[n], chix[n]) * dpsiy[n]
        bt2 = complex(dpsix[n], dchix[n]) * psiy[n]
        a[n] = (tp1 - mc * tp2) / (bt1 - mc * bt2)
        b[n] = (mc * tp1 - tp2) / (mc * bt1 - bt2)
    return a, b


class Mie:
    """
    A class that computes the Mie scattering functions for spheres

    ...

    Inputs (also Attributes)
    ______
    a_p : float
        sphere radius [m]
    n_p : float (real or complex)
        sphere refractive index
    n_m : float (real)
        medium (solvent) refractive index
    wavelength : float (real)
        vacuum wavelength of light [m]

    Attributes
    __________
    x : float
        size parameter, wavevector in medium (solvent) times particle radius
    y : float or complex
        wavevector in sphere times particle radius
    nmax : int
        number of terms needed in Mie sums from Ref. 2 by Wiscombe, page 1508
    mc : complex
        refractive index of particle / refractive index of matrix
    a, b : (numpy.ndarray, numpy.ndarray) (np.complex128, np.complex128)
        Mie a & b coefficients (from 1 to nmax, a[0] = b[0] = 0.+0.j)

    Methods
    _______
    qext : float
        Returns Qext, extinction cross section divided by geometrical cross
        section using formula on page 127 vdH
    qscat : float
        Returns Qsca, scattering cross section divided by geometrical cross
        section using formula on page 128 vdH
    qtrans : float
        Returns Qtrans = <cos θ> Qsca, transport cross section divided by
        geometrical cross section using formula on page 128 vdH
    s1s2(costheta) : (complex, complex)
        Returns S1 and S2 Mie scattering functions using formulas on page
        125 vdH
    form(theta) : float
        Returns form function i1 + i2 on page 126 vdH
    form_12(theta) : float
        Returns form functions i1 and i2 on page 126 vdH
    """

    def __init__(self,
                 a_p,  # sphere radius [m]
                 n_p,  # sphere real or complex refractive index
                 n_m,  # matrix real refractive index
                 wavelength,  # vacuum wavelength of light [m]
                 mie_coefs=mie_coefs_up  # function to calculate scatsngl coefficients
                 ):
        ka = 2. * np.pi * a_p / wavelength  # wavevector in vacuum times particle radius
        x = n_m * ka  # size parameter (wavevector in matrix times particle radius)
        # Determine the number of terms needed in Mie sums from Wiscombe (1980)
        if x <= 8.:
            nmax = int(x + 4. * x ** (1. / 3.) + 1.)
        elif x <= 4200.:
            nmax = int(x + 4.05 * x ** (1. / 3.) + 2.)
        else:
            nmax = int(x + 4. * x ** (1. / 3.) + 2.)
        self.nmax = nmax
        self.a_p = a_p
        self.n_p = n_p + 0.j
        self.n_m = n_m
        self.mc = self.n_p / self.n_m
        self.wavelength = wavelength
        self.y = self.n_p * ka  # wavevector in particle times particle radius
        self.x = x  # size parameter: wavevector in medium times particle radius
        self.a, self.b = mie_coefs(self.x, self.mc, self.nmax)

    def _pi_tau(self, costheta):
        """Returns pi and tau functions, page 124, vdH"""
        pi = np.zeros(self.nmax + 1)
        tau = np.zeros(self.nmax + 1)
        pi[1] = 1.
        pi[2] = 3. * costheta
        tau[1] = costheta
        tau[2] = 3. * (2. * costheta * costheta - 1.)
        if self.nmax > 2:
            for n in range(3, self.nmax + 1):
                pi[n] = (float(2 * n - 1) * costheta * pi[n - 1] - float(n) * pi[n - 2]) / float(n - 1)
                tau[n] = float(n) * costheta * pi[n] - float(n + 1) * pi[n - 1]
        return pi, tau

    def qext(self):
        """Returns Qext as defined on page 127"""
        sum = 0.0
        n = 1
        for n in range(1, self.nmax + 1):
            sum = sum + float(2 * n + 1) * (self.a[n].real + self.b[n].real)
        return 2. * sum / (self.x * self.x)

    def qscat(self):
        """Returns Qsca as defined on page 127"""
        sum = 0.
        for n in range(1, self.nmax + 1):
            sum = sum + float(2 * n + 1) * ((self.a[n] * self.a[n].conjugate()).real +
                                            (self.b[n] * self.b[n].conjugate()).real)
        return 2. * sum / (self.x * self.x)

    def qtrans(self):
        """Returns (1 - <cos(theta)>) * Qsca as defined on page 128"""
        sum = 0.0
        for n in range(1, self.nmax):
            p = self.a[n] * self.a[n + 1].conjugate() + self.b[n] * self.b[n + 1].conjugate()
            q = self.a[n] * self.b[n].conjugate()
            sum += (float(n * (n + 2.)) * p.real + (float(2 * n + 1) / float(n)) * q.real) / float(n + 1)
        cosqt = 4.0 * sum / (self.x * self.x)
        return self.qscat() - cosqt

    def s1s2(self, costheta):
        """Returns S1 and S2 functions on page 125"""
        pi, tau = self._pi_tau(costheta)
        s1 = s2 = 0. + 0.j
        for n in range(1, self.nmax + 1):
            s1 += float(2 * n + 1) * (self.a[n] * pi[n] + self.b[n] * tau[n]) / float(n * (n + 1))
            s2 += float(2 * n + 1) * (self.b[n] * pi[n] + self.a[n] * tau[n]) / float(n * (n + 1))
        return s1, s2

    def form(self, theta):
        """Returns the form function F(θ) = i1 + i2 on page 126-7"""
        s1, s2 = self.s1s2(np.cos(theta))
        f1 = (s1 * s1.conjugate()).real
        f2 = (s2 * s2.conjugate()).real
        return f1 + f2

    def form_12(self, theta):
        """Returns the form functions i1 & i2 on page 126"""
        s1, s2 = self.s1s2(np.cos(theta))
        f1 = (s1 * s1.conjugate()).real
        f2 = (s2 * s2.conjugate()).real
        return f1, f2


def F(theta, mie_inst):
    """
    Inputs
    ______
    theta : numpy.ndarray
        scattering angles [radians]
    mie_inst : instance of Mie class from this module

    Returns
    -------
    F : numpy.ndarray
        unpolarized Mie form function, |S1|^2 + |S2|^2
    """
    form = np.zeros(theta.size, dtype=float)
    for i in range(theta.size):
        form[i] = mie_inst.form(theta[i])
    return form
