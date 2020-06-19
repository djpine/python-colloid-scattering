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
import scipy.constants as const
import scipy.optimize as op
import scipy.integrate

pi = np.pi


class msaHP:
    """
    This routine is for diagnostic purposes only.  It calculates the MSA functions
    for all real roots of the quartic that is solved in the Hayter-Penfold solution,
    including the physically irrelevant as well as the physically relevant root.
    Use the routine YukawaMSA for calculations.  Use this one only to diagnose any
    problems you might have with YukawaMSA.

    Calculates the static structure factor S(q), pair correlation function g(r), the
    direct correlation function c(r), and ancillary functions for hard-sphere Yukawa
    fluid calculated using the mean spherical approximation of Hayter and Penfold
    (see [1]_).

    Attributes
    ----------
    phi : float
        volume fraction of spheres
    sigma : float
        hard-core diameter of spheres
    zeff : float or integer
        effective charge of spheres in units of |e|
    debye_length : float
        Debye screening length in meters
    epsilon : float
        dielectric constant of solvent (pure number = 1 for vacuum)
    tempC : float
        temperature of system in degrees Celcius
    kappa : float
        inverse Debye length (inverse meters)
    kapsig : float
        kappa sigma
    psi0 : float
        surface potential [volts]
    contact_potential :
        contact potential [kT]
    A, B, C, F : numpy.ndarray
        coefficients from Eqs. 7 & 8 of [1]_
        returns only the real roots (usually 2 of 4 total)

    Methods
    -------
    s(K) : numpy.ndarray
        static structure factor with K = q sigma
    g(x) : numpy.ndarray
        pair correlation (radial distribution) function with x = r / sigma
    c(x) : numpy.ndarray
        direct correlation function with x = r / sigma
    potential(r) : numpy.ndarray
        screened Coulomb (Yukawa) potential in SI units

    Notes
    -----
    The coefficient F used in calculating the coefficients A, B, & C, as well as s(K),
    g(x), and c(x) is obtained by solving for the roots a fourth-order polynomial. Only
    one of the four roots gives the physically reasonable result with g(x)=0 for x<1.
    Typically, there are two complex roots, which are discarded [in _polyroots method].
    The methods will return s(K), g(x), and c(x) for all of the real roots.

    References
    ----------
    ..  [1] JB Hayter, J Penfold, "An analytic structure factor for macroion solutions",
        Molecular Physics vol. 42, pp. 109-118, 1981.
    """

    def __init__(self, phi, sigma, zeff, debye_length, epsilon=78.3, tempC=25.):
        """
        Constructor for the SqYukawaMSA class

        Parameters
        ----------
        """
        self.phi = phi
        self.sigma = sigma
        self.zeff = float(zeff)
        self.debye_length = debye_length
        self.epsilon = epsilon
        self.tempC = tempC
        self.tempK = tempC + 273.15

        eps = epsilon * const.epsilon_0
        self.kappa = 1. / debye_length
        self.kapsig = sigma / debye_length
        self.psi0 = zeff * const.e / (pi * eps * sigma * (2. + self.kapsig))
        self.contact_potential = pi * eps * sigma * self.psi0 * self.psi0 / (const.k * self.tempK)

        # Calculate A, B, C, F coefficients from Eqs. 7 & 8 of [1]_
        A, B, C, F = self._coefficients()
        self.A = A
        self.B = B
        self.C = C
        self.F = F

    def _coefficients(self):
        k = self.kapsig
        eta = self.phi
        ge = self.contact_potential
        k2 = k * k
        k4 = k2 * k2
        shk = np.sinh(k)
        chk = np.cosh(k)

        delta = 1. - eta
        delta2 = delta * delta
        delta4 = delta2 * delta2

        alpha1 = -(2. * eta + 1.) * delta / k
        alpha2 = (-1. + eta * (-4. + 14. * eta)) / k2
        alpha3 = 36. * eta * eta / k4

        beta1 = -(1. + eta * (7. + eta)) * delta / k
        beta2 = 9. * eta * (-2. + eta * (4. + eta)) / k2
        beta3 = 12. * eta * (-1. + eta * (8. + 2. * eta)) / k4

        nu1 = -(5. + eta * (45. + eta * (3. + eta))) * delta / k
        nu2 = (-20. + eta * (42. + eta * (3. + 2. * eta))) / k2
        nu3 = (-5. + eta * (30. + 2. * eta * eta)) / k4
        nu4 = nu1 + 24. * eta * k * nu3
        nu5 = 6. * eta * (nu2 + 4. * nu3)

        phi1 = 6. * eta / k
        phi2 = delta - 12. * eta / k2

        tau1 = (eta + 5.) / (5. * k)
        tau2 = (eta + 2.) / k2
        tau3 = -12. * eta * ge * (tau1 + tau2)
        tau4 = 3. * eta * k2 * (tau1 * tau1 - tau2 * tau2)
        tau5 = 0.3 * eta * (eta + 8.) - 2. * ((2. * eta + 1.) / k) ** 2

        a1 = (24. * eta * ge * (alpha1 + alpha2 + (1. + k) * alpha3) - (2. * eta + 1.) ** 2) / delta4
        a2 = 24. * eta * (alpha3 * (shk - k * chk) + alpha2 * shk - alpha1 * chk) / delta4
        a3 = 24. * eta * (((2. * eta + 1.) / k) ** 2 - 0.5 * delta2 + alpha3 * (
                chk - 1. - k * shk) - alpha1 * shk + alpha2 * chk) / delta4

        b1 = eta * (1.5 * (eta + 2.) ** 2 - 12. * ge * (beta1 + beta2 + (1. + k) * beta3)) / delta4
        b2 = 12. * eta * (beta3 * (k * chk - shk) - beta2 * shk + beta1 * chk) / delta4
        b3 = 12. * eta * (0.5 * delta2 * (eta + 2.) - 3. * eta * ((eta + 2.) / k) ** 2 - beta3 * (
                chk - 1. - k * shk) + beta1 * shk - beta2 * chk) / delta4

        v1 = (0.25 * (2. * eta + 1.0) * (10. + eta * (-2. + eta)) - ge * (nu4 + nu5)) / (5. * delta4)
        v2 = (nu4 * chk - nu5 * shk) / (5. * delta4)
        v3 = ((5. + eta * eta * (-6. + eta)) * delta - 6. * eta * (
                10. + eta * (18. + eta * (-3. + eta * 2.))) / k2 + 24. * eta * nu3 + nu4 * shk - nu5 * chk) / (
                     5. * delta4)

        p1 = (ge * (phi1 - phi2) ** 2 - 0.5 * (eta + 2.0)) / delta2
        p2 = ((phi1 * phi1 + phi2 * phi2) * shk + 2. * phi1 * phi2 * chk) / delta2
        p3 = ((phi1 * phi1 + phi2 * phi2) * chk + 2. * phi1 * phi2 * shk + phi1 * phi1 - phi2 * phi2) / delta2

        t1 = tau3 + tau4 * a1 + tau5 * b1
        t2 = tau4 * a2 + tau5 * b2 + 12. * eta * (tau1 * chk - tau2 * shk)
        t3 = tau4 * a3 + tau5 * b3 + 12. * eta * (tau1 * shk - tau2 * (chk - 1.)) - 0.4 * eta * (eta + 10.) - 1.

        mu1 = t2 * a2 - 12. * eta * v2 * v2
        mu2 = t1 * a2 + t2 * a1 - 24. * eta * v1 * v2
        mu3 = t2 * a3 + t3 * a2 - 24. * eta * v2 * v3
        mu4 = t1 * a1 - 12. * eta * v1 * v1
        mu5 = t1 * a3 + t3 * a1 - 24. * eta * v1 * v3
        mu6 = t3 * a3 - 12. * eta * v3 * v3

        lambda1 = 12. * eta * p2 * p2
        lambda2 = 24. * eta * p1 * p2 - 2. * b2
        lambda3 = 24. * eta * p2 * p3
        lambda4 = 12. * eta * p1 * p1 - 2. * b1
        lambda5 = 24. * eta * p1 * p3 - 2. * b3 - k2
        lambda6 = 12. * eta * p3 * p3

        omega16 = mu1 * lambda6 - mu6 * lambda1
        omega13 = mu1 * lambda3 - mu3 * lambda1
        omega36 = mu3 * lambda6 - mu6 * lambda3
        omega15 = mu1 * lambda5 - mu5 * lambda1
        omega35 = mu3 * lambda5 - mu5 * lambda3
        omega26 = mu2 * lambda6 - mu6 * lambda2
        omega12 = mu1 * lambda2 - mu2 * lambda1
        omega14 = mu1 * lambda4 - mu4 * lambda1
        omega34 = mu3 * lambda4 - mu4 * lambda3
        omega25 = mu2 * lambda5 - mu5 * lambda2
        omega24 = mu2 * lambda4 - mu4 * lambda2

        w4 = omega16 * omega16 - omega13 * omega36
        w3 = 2. * omega16 * omega15 - omega13 * (omega35 + omega26) - omega12 * omega36
        w2 = omega15 * omega15 + 2. * omega16 * omega14 - omega13 * (omega34 + omega25) - omega12 * (omega35 + omega26)
        w1 = 2. * omega15 * omega14 - omega13 * omega24 - omega12 * (omega34 + omega25)
        w0 = omega14 * omega14 - omega12 * omega24

        F = self._polyroots([w0, w1, w2, w3, w4])
        A = np.zeros(F.size)
        B = np.zeros(F.size)
        C = np.zeros(F.size)
        for i in range(F.size):
            C[i] = -(omega14 + F[i] * (omega15 + F[i] * omega16)) / (omega12 + F[i] * omega13)
            B[i] = b1 + b2 * C[i] + b3 * F[i]
            A[i] = a1 + a2 * C[i] + a3 * F[i]
        return A, B, C, F

    def _polyroots(self, c, polish=True, print_roots=True):
        """
        Calculates the roots of the polynomial given by Eq. 8 in Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)

        Parameters
        ----------
        c : list or array
            w coefficients of the polynomial given by Eq. 8 in Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)
        polish : boolean (default True)
            refines roots after they are found using numpy polyroots routine (usually not necessary)
        print_roots : boolean (default True)
            prints out the complex and real roots of Eq. 8 in Hayter & Penfold

        Returns
        -------
        F : 1D array
            real roots of the polynomial given by Eq. 8 in Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)
        """
        roots = np.polynomial.polynomial.polyroots(c)
        roots_real = roots[np.abs(roots.imag / roots.real) < 1e-10].real  # discard complex roots
        if print_roots is True:
            np.set_printoptions(formatter={'float_kind': '{:9.6g}'.format, 'complex_kind': '{:15.4g}'.format})
            print('roots = {}'.format(roots))
            print('real roots = {}'.format(roots_real))
            np.set_printoptions()
        if polish is True:
            # polish roots with Newton-Raphson (generally not necessary)
            f = lambda x: np.polynomial.polynomial.polyval(x, c)
            cp = np.polynomial.polynomial.polyder(c)
            fprime = lambda x: np.polynomial.polynomial.polyval(x, cp)
            for i in range(roots_real.size):
                roots_real[i] = op.newton(f, roots_real[i], fprime, tol=1e-8)
        return roots_real

    def _a(self, K, A, B, C, F):
        """
         Calculates the function a(K) given by Eq. 14 in Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)

         Parameters
         ----------
         K : 1D numpy array
             Q sigma, the q scattering vector times the hard sphere diameter sigma
         A, B, C, F : 1D numpy arrays
            coefficients from Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)

         Returns
         -------
         a(K) : 1D array
             function a(K) given by Eq. 14 in Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)
         """
        k = self.kapsig
        shk, chk = np.sinh(k), np.cosh(k)
        sK, cK = np.sin(K), np.cos(K)
        K2 = K * K
        K3 = K2 * K
        aa = (sK - K * cK) / K3 + self.phi * (0.5 / K3) * (
                24. / K3 + 4. * (1. - 6. / K2) * sK - (K - 12. / K + 24. / K3) * cK)
        bb = ((2. / K - K) * cK + 2. * sK - 2. / K) / K3
        kKdenom = K * (K2 + k * k)
        cc = (k * chk * sK - K * shk * cK) / kKdenom
        ff = (k * shk * sK - K * (chk * cK - 1.)) / kKdenom + (cK - 1.) / K2
        last = self.contact_potential * (k * sK + K * cK) / kKdenom
        return A * aa + B * bb + C * cc + F * ff - last

    def c(self, x, A, B, C, F):
        """
        Calculates the direct correlation function c(x) given by Eqs. 5a and 6 in Hayter & Penfold,
        Mol. Phys. 42, 109-118 (1981)

        Parameters
        ----------
        x : 1D numpy array
            r/sigma where r is the center-to-center distance between spheres and sigma is the hard core diameter
        A, B, C, F : 1D numpy arrays
            coefficients from Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)

        Returns
        -------
        c(x) : 1D numpy array of length x.size
            direct correlation function c(x).  If there is more than one set of A, B, C, F coefficients,
            the routine will return c(x) for each set.  The physically relevant one is the one for which
            g(x)=0 for x<1.
        """
        k = self.kapsig
        kx = self.kapsig * x
        return np.where(x < 1.,
                        A * (1. + 0.5 * self.phi * x ** 3) + B * x + (C * np.sinh(kx) + F * (np.cosh(kx) - 1.)) / x,
                        - self.contact_potential * np.exp(k - kx) / x)

    def s(self, K, A, B, C, F):
        """
        Calculates the static structure factor for a Yukawa potential using the mean spherical approximation from
        Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)

        Parameters
        ----------
        K : 1D numpy array
            qd where q is the wavevector and d is the hard-sphere diameter
        A, B, C, F : 1D numpy arrays
            coefficients from Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)

        Returns
        -------
        1D numpy array:
            The Yukawa MSA static structure factor as a function of qd.  If there is more than one set of
            A, B, C, F coefficients, the routine will return s(K) for each set.  The physically relevant one
            is the one for which g(x)=0 for x<1.
        """
        return 1. / (1. - 24. * self.phi * self._a(K, A, B, C, F))

    def g(self, x, A, B, C, F, Kmax=1024, nK=8 * 1024):
        """
        Calculates the pair correlation (radial distribution) function for a Yukawa potential using the mean
        spherical approximation from Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)

        Parameters
        ----------
        x : 1D numpy array
            r/sigma where r is the center-to-center distance between spheres and sigma is the hard core diameter
        A, B, C, F : 1D numpy arrays
            coefficients from Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)
        Kmax : integer
            maximum dimensionless Q vectors used to calculate s(K).  A large number is needed since g(x) is calculated
            by integrating over s(K), as specified by Eq. 12 of Hayter & Penfold, Mol. Phys. 42, 109-118 (1981)
        nK : integer
            the number of points where s(K) is calculated.  nK = 8*Kmax is needed to get g(x)=0 for x<1, particularly
            at higher values of phi.

        Returns
        -------
        1D numpy array:
            The Yukawa MSA pair correlation (radial distribution) function as a function of x = r/sigma.
            If there is more than one set of A, B, C, F coefficients, the routine will return g(x) for each set.
            The physically relevant one is the one for which g(x)=0 for x<1.

        Notes
        -----
        The negative undershoot of g(r) just above x=0 at lower volume fractions is a known problem of the
        Hayter-Penfold MSA scheme.
        """
        K = np.linspace(0.01, Kmax, nK)
        ss = self.s(K, A, B, C, F)
        h = np.zeros(x.size)
        for i in range(x.size):
            h[i] = scipy.integrate.simps((ss - 1.) * K * np.sin(K * x[i]), x=K, dx=K[1] - K[0])
        return 1. + h / (12. * pi * self.phi * x)

    def choose_root(self):
        """
        Determines which of the real roots of the polynomial given by Eq. 8 in Hayter & Penfold,
        Mol. Phys. 42, 109-118 (1981) is the physically relevant one [the one that gives g(r)=0
        (to within numerical precision)].
        """
        gsum = np.inf
        x = np.linspace(0.02, 0.98, 20)
        for i in range(self.F.size):
            gg = self.g(x, self.A[i], self.B[i], self.C[i], self.F[i])
            gsqr = np.sum(gg * gg)
            if gsqr < gsum:
                gsum = gsqr
                i_root = i
        return i_root

    def potential(self, r):
        """
        Returns potential in SI units

        Parameters
        ----------
        r: sphere center-to-center distance in meters
        """
        amp = pi * self.epsilon * const.epsilon_0 * (self.sigma * self.psi0) ** 2
        return amp * np.exp(-self.kappa * (r - self.sigma)) / r
