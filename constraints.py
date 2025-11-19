import numpy as np

mmu = 100e6
Gmu = 2.9e-10
e = 0.3
p = 50e6

def CLFV_constraints(results):
    nu_mass = results["nu_mass"]
    m_psi = results["m_psi"]
    U_Npsi = results["U_Npsi"]
    ynue = results["ynue"]
    ynumu = results["ynumu"]

    #Masssum constraint
    cond_masssum = np.sum(nu_mass, axis=1) < 1

    m1 = nu_mass[:,2]
    m2 = nu_mass[:,1]
    m3 = nu_mass[:,0]

    cond_m21sq = ((m2**2 - m1**2) < 7.69e-5) & ((m2**2 - m1**2) > 7.31e-5)
    cond_m32sq = ((m3**2 - m2**2) < 2.477e-3) & ((m3**2 - m2**2) > 2.425e-3)

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

    mask = cond_masssum & cond_nu_positive & cond_CLFV & cond_m21sq & cond_m32sq

    return mask



