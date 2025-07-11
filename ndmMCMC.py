import numpy as np
from numpy import sin, cos, exp, pi, log, sqrt
import matplotlib.pyplot as plt
import emcee #MCMC

mu0 = 1e25 #GUT scale
vH = 250*1e9
betak = 35/6

#Normal hierarchy values
obs_msq21 = 7.5*1e-5
obs_msq31 = 2.55*1e-3
obs_s12sq = 3.18*1e-1
obs_s23sq = 5.74*1e-1
obs_s13sq = 2.2*1e-2
#obs_CP = 1.08*pi

sigma_msq21 = 0.21*1e-5
sigma_msq31 = 0.025*1e-3
sigma_s12sq = 0.16*1e-1
sigma_s23sq = 0.14*1e-1
sigma_s13sq = 0.065*1e-2
#sigma_CP = 0.125*pi

#CL Yukawa coefficients
ye = 0.51099895000*1e6*sqrt(2)/(250*1e9)
ymu = 105.6583755*1e6*sqrt(2)/(250*1e9)
ytau = 1776.93*1e6*sqrt(2)/(250*1e9)

#1-loop beta function solution
def g(g0, k, v):
    return (g0**(-2) + (k/(8*pi**2))*log(v/mu0))**(-1/2)

#Mass
def Mnu(k, g0, ve, vmu, vtau):
    ge = g(g0, k, ve)
    gmu = g(g0, k, vmu)
    gtau = g(g0, k, vtau)
    return np.array([[np.exp(-8*np.pi**2/(ge**2))*ye, 0, 0],
                     [0, np.exp(-8*np.pi**2/(gmu**2))*ymu, 0],
                     [0,0,np.exp(-8*np.pi**2/(gtau**2))*ytau]])

#Mixing
def Inu(k, g0, ve, vmu, vtau):
    ge = g(g0, k, ve)
    gmu = g(g0, k, vmu)
    gtau = g(g0, k, vtau)
    inste = np.exp(-8*np.pi**2/(ge**2))
    instmu = np.exp(-8*np.pi**2/(gmu**2))
    insttau = np.exp(-8*np.pi**2/(gtau**2))
    remu = ve/vmu
    retau = ve/vtau
    rmutau = vmu/vtau
    return 0.01*np.array([[inste*ye, inste*ye/remu, inste*ye/retau],
                         [instmu*ymu*remu, instmu*ymu, instmu*ymu/rmutau],
                         [insttau*ytau*retau, insttau*ytau*rmutau, insttau*ytau]])

def log_prob(theta):
    g0 = theta[0]
    ve, vmu, vtau = 10**theta[1:]
    if (np.any(theta[1:] < 12)):
        return -np.inf

    Y = (vH/sqrt(2))*(Mnu(betak, g0, ve, vmu, vtau) + Inu(betak, g0, ve, vmu, vtau)) #Neutrino Yukawa after EWSB

    eigenvals, U = np.linalg.eig(Y)
    masses = np.abs(eigenvals)

    dmsq21 = masses[1]**2 - masses[0]**2
    dmsq31 = masses[2]**2 - masses[0]**2

    s13sq = U[0,2]**2
    s12sq = (U[0,1]/sqrt(1-s13sq))**2
    s23sq = (U[1,2]/sqrt(1-s13sq))**2

    chi2 = ((dmsq21 - obs_msq21)/sigma_msq21)**2
    chi2 += ((dmsq31 - obs_msq31)/sigma_msq31)**2
    chi2 += ((s12sq - obs_s12sq)/sigma_s12sq)**2
    chi2 += ((s23sq - obs_s23sq)/sigma_s23sq)**2
    chi2 += ((s13sq - obs_s13sq)/sigma_s13sq)**2

    return -0.5*chi2

ndim = 4
nwalkers = 50
p0 = np.zeros((nwalkers, ndim))
p0[:,0] = np.random.uniform(0.1, 0.7, size=nwalkers)
p0[:, 1:] = np.random.uniform(13, 23, size=(nwalkers, ndim-1))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(p0, 1000)

samples = sampler.get_chain(flat=True)
print(samples)

#v = np.logspace(13, 23, 1000)
#plt.plot(v, g(1, betak, v))
#plt.xscale('log')
#plt.show()    
    