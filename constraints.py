import numpy as np




def Br(M, U):


def CLFV_constraints(results):
    masssum = results["masssum"]
    m_psi = results["m_psi"]
    U_Npsi = results["U_Npsi"]

    #Masssum constraint
    cond_masssum = masssum < 0.12

    UUdagger = U_Npsi @ U_Npsi.conj().transpose(0,2,1)

