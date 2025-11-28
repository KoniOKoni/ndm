import numpy as np
import cmath

ye = 2.94e-6
ymu = 6.07e-4
ytau = 1.02e-2
vH = 250e9

def g2inv(Lambda, g0, v):
    return 1.0/(g0**2) + (35.0 / 6.0) / (8.0 * np.pi**2) * np.log(v/Lambda)

def model_NDM(params, idx):
    Lambda = 10**params[:, idx['LogLambda']]
    g0 = params[:, idx['g0']]
    ve = 10**params[:, idx['Logve']]
    vmu = 10**params[:, idx['Logvmu']]
    vtau = 10**params[:, idx['Logvtau']]
    
    vs = [ve, vmu, vtau]
    ys = [ye, ymu, ytau]

    gamma = np.zeros((params.shape[0], 3, 3), dtype=complex)
    M_inst = np.zeros((params.shape[0], 3, 3), dtype=complex)
    G = np.zeros((params.shape[0], 3, 3), dtype=complex)
    Y = np.zeros((params.shape[0], 3, 3), dtype=complex)
    gamma_Npsi = np.zeros((params.shape[0], 3, 3), dtype=complex)

    for i in range(3):
        for j in range(i+1):
            key_re = f"Logg{i+1}{j+1}Re"
            key_im = f"Logg{i+1}{j+1}Im"
            gamma[:, i, j] = 10**params[:, idx[key_re]] + (1j) * 10**params[:, idx[key_im]]

    for i in range(3):
        for j in range(3):
            gamma_Npsi[:, i, j] = vs[i] * gamma[:, i, j]

    inst_e = np.exp(-8 * np.pi**2 * g2inv(Lambda, g0, ve))
    inst_mu = np.exp(-8 * np.pi**2 * g2inv(Lambda, g0, vmu))
    inst_tau = np.exp(-8 * np.pi**2 * g2inv(Lambda, g0, vtau))

    M_inst[:, 0, 0] = ye * inst_e
    M_inst[:, 1, 1] = ymu * inst_mu
    M_inst[:, 2, 2] = ytau * inst_tau

    for i in range(3):
        for j in range(3):
            for k in range(3):
                G[:, i, j] += np.exp(-8 * np.pi**2 * g2inv(Lambda, g0, vs[i])) * gamma[:, i, k] * gamma[:, j, k].conjugate() * ys[i] * vs[i] * vs[j] / vs[i]**2

    Y = (M_inst + G)*vH/np.sqrt(2)

    U, nu_mass, PMNS_dagger = np.linalg.svd(Y)

    U_Npsi, m_psi, Vh_Npsi = np.linalg.svd(gamma_Npsi)

    return {"nu_mass" : nu_mass, "m_psi" : m_psi, "U_Npsi" : U_Npsi, "ynue" : ye*inst_e, "ynumu" : ymu*inst_mu}

    






