import numpy as np
import scipy as sc
from scipy.special import gamma as Gamma
"""
Plots the model of Gupta 2009
"""


def Tau(t,PVT):
    """
    This function calculates the non-dimensionalized time of the cough model of Gupta 2009

    param t: Time (s)
    param PVT: Peak velocity time (s)
    """

    tau = t/ PVT
    return tau

def M_exp(Q,CPFR):  
    """
    This function calculates the non-dimensionalized flowrate of the cough model of Gupta 2009 from experimental data
    param Q: Flowrate of the cough (L/s)
    param CPFR: Cough peak flow rate (L/s)
    """

    m = Q/CPFR
    return m

def M_initial(Tau,a1,b1,c1):
    """
    This function calculates the flowrate for Tau<1.2, and the first part of the equation of the model when Tau>1.2
    according to Gupta 2009
    See M_model for more details
    """

    return a1* Tau**(b1-1) * np.exp(-Tau/c1) / (Gamma(b1) * c1**(b1)) 

def M_model(Tau,PVT,CPFR,CEV):
    """
    This function calculates the non-dimensionalized flowrate of the cough model of Gupta 2009
    param Tau: Non-dimensionalized time usually goes from 0 to about 10
    param PVT: Peak velocity time (s)
    param CPFR: Cough peak flow rate (L/s)
    param CEV: Cough exit velocity (m/s)
    The coefficients are taken from the article
    """
    #constants
    a1 = 1.680
    b1 = 3.338
    c1 = 0.428
    a2 = CEV/ (PVT*CPFR) -a1
    b2 = -2.158 *CEV/ (PVT*CPFR) + 10.457
    c2 = 1.8 / (b2-1)

    m = np.zeros(len(Tau))

    mask = Tau < 1.2

    m[mask] = M_initial(Tau[mask],a1,b1,c1)
    m[~mask] = M_initial(Tau[~mask],a1,b1,c1) + a2 * (Tau[~mask] -1.2)**(b2-1) * np.exp(-(Tau[~mask]-1.2)/c2)/ (Gamma(b2)*c2**b2)

    return m


def estimator(gender,weight,height):
    """
    This function estimates the PVT (s), CPFR (L/s), CEV (L) by taking the gender (Male or Female), weigth( kg), length(m)
    \\
    Returns: 
    PVT, CPFR, CEV
    """

    if gender.lower() == "Male".lower():
        CPFR = -8.890 + 6.3952 * height + 0.0346 * weight
        CEV = 0.138 * CPFR + 0.2983
        PVT = (1.360 * CPFR + 65.860)*1E-3
    elif gender.lower() == "Female".lower():
        CPFR = -3.9702 + 4.6265 * height
        CEV = 0.0204 * CPFR - 0.043
        PVT = (3.152 * CPFR + 64.631)*1E-3
    else:
        print("Choose either Male or Female")
        exit()
    return PVT, CPFR, CEV