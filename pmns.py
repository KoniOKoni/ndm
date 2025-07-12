import numpy as np
from numpy import exp, log, sqrt, pi
from cobaya.likelihood import Likelihood

class NeutrinoYukawaLikelihood(Likelihood):
    def initialize(self):
        # Constants
        self.mu0 = 1e25  # GUT scale
        self.vH = 250e9
        self.betak = 35/6
        # Observed values (normal hierarchy)
        self.obs_msq21 = 7.5e-5
        self.obs_msq31 = 2.55e-3
        self.obs_s12sq = 3.18e-1
        self.obs_s23sq = 5.74e-1
        self.obs_s13sq = 2.2e-2
        # Uncertainties
        self.sigma_msq21 = 0.21e-5
        self.sigma_msq31 = 0.025e-3
        self.sigma_s12sq = 0.16e-1
        self.sigma_s23sq = 0.14e-1
        self.sigma_s13sq = 0.065e-2
        # Charged-lepton Yukawas
        self.ye = 0.51099895e6 * sqrt(2) / (250e9)
        self.ymu = 105.6583755e6 * sqrt(2) / (250e9)
        self.ytau = 1776.93e6 * sqrt(2) / (250e9)
    
    def logp(self, **params_values):
        # Convert log10 VEVs
        log10_ve = params_values['log10_ve']
        log10_vmu = params_values['log10_vmu']
        log10_vtau = params_values['log10_vtau']
        ve = 10**log10_ve
        vmu = 10**log10_vmu
        vtau = 10**log10_vtau
        g0 = params_values['g0']
        # Prior cuts
        if not ((log10_ve > log10_vmu) and (log10_vmu > log10_vtau)):
            return -np.inf
        if log10_ve < 12 or log10_vmu < 12 or log10_vtau < 12:
            return -np.inf
        if g0 > 0.6 or g0 <= 0:
            return -np.inf

        # Neutrino Yukawa matrix after EWSB
        Y = (self.vH / sqrt(2)) * (self.Mnu(g0, ve, vmu, vtau) + self.Inu(g0, ve, vmu, vtau))

        # Diagonalize
        eigenvals, U = np.linalg.eig(Y)
        masses = np.abs(eigenvals)

        # Mass-squared differences
        dmsq21 = masses[1]**2 - masses[0]**2
        dmsq31 = masses[2]**2 - masses[0]**2

        # Mixing angles
        s13sq = U[0, 2]**2
        if s13sq <= 0 or s13sq >= 1:
            return -np.inf
        s12sq = (U[0, 1] / sqrt(1 - s13sq))**2
        s23sq = (U[1, 2] / sqrt(1 - s13sq))**2

        # Chi-square
        chi2 = ((dmsq21 - self.obs_msq21) / self.sigma_msq21)**2
        chi2 += ((dmsq31 - self.obs_msq31) / self.sigma_msq31)**2
        chi2 += ((s12sq - self.obs_s12sq) / self.sigma_s12sq)**2
        chi2 += ((s23sq - self.obs_s23sq) / self.sigma_s23sq)**2
        chi2 += ((s13sq - self.obs_s13sq) / self.sigma_s13sq)**2

        return -0.5 * chi2

    def g(self, g0, v):
        return (g0**(-2) + (self.betak / (8 * pi**2)) * log(v / self.mu0))**(-0.5)

    def Mnu(self, g0, ve, vmu, vtau):
        ge = self.g(g0, ve)
        gmu = self.g(g0, vmu)
        gtau = self.g(g0, vtau)
        return np.diag([exp(-8 * pi**2 / ge**2) * self.ye,
                        exp(-8 * pi**2 / gmu**2) * self.ymu,
                        exp(-8 * pi**2 / gtau**2) * self.ytau])

    def Inu(self, g0, ve, vmu, vtau):
        ge = self.g(g0, ve)
        gmu = self.g(g0, vmu)
        gtau = self.g(g0, vtau)
        inste = exp(-8 * pi**2 / ge**2)
        instmu = exp(-8 * pi**2 / gmu**2)
        insttau = exp(-8 * pi**2 / gtau**2)
        remu = ve / vmu
        retau = ve / vtau
        rmutau = vmu / vtau
        mat = np.array([
            [inste * self.ye, inste * self.ye / remu, inste * self.ye / retau],
            [instmu * self.ymu * remu, instmu * self.ymu, instmu * self.ymu / rmutau],
            [insttau * self.ytau * retau, insttau * self.ytau * rmutau, insttau * self.ytau]
        ])
        return 0.01 * mat