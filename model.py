import numpy as np

def _g2inv(Lambda, g0, v):
    return 1.0/(g0**2) + (35.0 / 6.0) / (8.0 * np.pi**2) * np.log(v/Lambda)

def model(params, idx):
    Lambda = 10**params[:, idx['LogLambda']]
    g0 = params[:, idx['g0']]
    ve = 10**params[:, idx['Logve']]
    vmu = 10**params[:, idx['Logvmu']]
    vtau = 10**params[:, idx['Logvtau']]

    ge2inv = _g2inv(Lambda, g0, ve)
    gmu2inv = _g2inv(Lambda, g0, vmu)
    gtau2inv = _g2inv(Lambda, g0, vtau)

    