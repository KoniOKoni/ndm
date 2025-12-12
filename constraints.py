import numpy as np
import cmath

mmu = 100e6
Gmu = 2.9e-10
e = 0.3
p = 50e6

m21sq_obs = 7.49e-5
m32sq_obs = 2.513e-3
m21sq_sigma = 0.19e-5
m32sq_sigma = 0.020e-3

s12sq_obs = 0.308
s12sq_sigma = 0.011
s23sq_obs = 0.470
s23sq_sigma = 0.015
s13sq_obs = 0.02215
s13sq_sigma = 0.00057
CP_obs = 212
CP_sigma = 30

def CLFV_constraints(results):
    nu_mass = results["nu_mass"]
    m_psi = results["m_psi"]
    U_Npsi = results["U_Npsi"]
    ynue = results["ynue"]
    ynumu = results["ynumu"]
    PMNS = results["PMNS_dagger"]

    U = np.transpose(PMNS, axes=(0,2,1)).conjugate()
    s13, expCP = cmath.polar(U[:,0,2])
    chi_s13sq = (s13**2 - s13sq_obs)/s13sq_sigma
    c13 = np.sqrt(1 - s13**2)
    s12 = abs(U[:,0,1])/c13
    chi_s12sq = (s12**2 - s12sq_obs)/s12sq_sigma
    s23 = abs(U[:,1,2])/c13
    chi_s23sq = (s23**2 - s23sq_obs)/s23sq_sigma



    #Masssum constraint
    cond_masssum = np.sum(nu_mass, axis=1) < 1

    m1 = nu_mass[:,2]
    m2 = nu_mass[:,1]
    m3 = nu_mass[:,0]

    chi_m21sq = (m2**2 - m1**2 - m21sq_obs)/m21sq_sigma
    chi_m32sq = (m3**2 - m2**2 - m32sq_obs)/m32sq_sigma

    cond_nu_positive = np.all(nu_mass > 0, axis=1)

    Ieg = U_Npsi[:, 0,0] * U_Npsi[:, 0, 1].conjugate() * m_psi[:, 0]**3 + U_Npsi[:, 0,1] * U_Npsi[:, 1, 1].conjugate() * m_psi[:, 1]**3 + U_Npsi[:, 0,2] * U_Npsi[:, 2, 1].conjugate() * m_psi[:, 2]**3
    Ieee = U_Npsi[:, 0,0] * U_Npsi[:, 0, 1].conjugate() * m_psi[:, 0]**4 + U_Npsi[:, 0,1] * U_Npsi[:, 1, 1].conjugate() * m_psi[:, 1]**4 + U_Npsi[:, 0,2] * U_Npsi[:, 2, 1].conjugate() * m_psi[:, 2]**4
    Iegg = U_Npsi[:, 0,0] * U_Npsi[:, 0, 1].conjugate() * m_psi[:, 0] + U_Npsi[:, 0,1] * U_Npsi[:, 1, 1].conjugate() * m_psi[:, 1] + U_Npsi[:, 0,2] * U_Npsi[:, 2, 1].conjugate() * m_psi[:, 2]

    Ieg *= e * ynue * ynumu * p / (mmu**4)
    Ieee *= e**2 * ynue * ynumu / (p**6)
    Iegg *= e**2 * ynue * ynumu / (p**2)

    Gamma_mueg = Ieg**2 * mmu
    Gamma_mueee = Ieee**2 * mmu**5
    Gamma_muegg = Iegg**2 * mmu**3

    Br_eg = Gamma_mueg / Gmu
    Br_eee = Gamma_mueee / Gmu
    Br_egg = Gamma_muegg / Gmu

    cond_CLFV = (Br_eg < 3.1e-13) & (Br_eee < 1e-12) & (Br_egg < 7.2e-11)

    mask = cond_masssum & cond_nu_positive & cond_CLFV

    return mask, -0.5 * (chi_m21sq**2 + chi_m32sq**2 + chi_s12sq**2 + chi_s13sq**2 + chi_s23sq**2)



