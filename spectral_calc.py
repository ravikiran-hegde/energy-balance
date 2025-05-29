from scipy.constants import h, c, k
import numpy as np

g = 9.81 # m s-2

#########################

def planck_nu(nu_cm, T = 260.):
    """
    Planck function B(nu, T) in terms of wavenumber nu [cm^-1]
    Returns spectral radiance [W/m^2/sr/(cm^-1)]
    """
    
    nu_m = nu_cm * 100.0  # convert input from cm^-1 to m^-1

    # standard Planck formula (per m^-1)
    B = np.pi* (2 * h * c**2 * nu_m**3) / (np.exp(h * c * nu_m / (k * T)) - 1)
    return B

#########################

def column_mass_h2o(T, RH = 1.):
    """
    Returns the total mass of water vapour in a column with specified surface temperature and relative humidity
    Reference: Jeevanjee 2023, Stevens and Kluft 2023
    """

    L = 2.5e6 # J kg−1; Latent heat of vaporisation
    R_H2O = 461.5 # J kg−1 K−1; specific gas constant of water vapor
    

    LAPSE_RATE = 6.5 / 1000 # K m-1

    P_NU_REF = 2.5e11 # Pa
    # typical values; not really sensitive

    TS = 290 # K 
    TSTR = 210 # K

    m_ref = RH * P_NU_REF * (TS + TSTR)/ ( 2 * LAPSE_RATE * L )
    
    return  m_ref * np.exp( - L / (R_H2O * T)) # kg m-2

def column_mass_co2(q_co2):
    """
    Returns the total mass of CO2 in a column with specified concentration
    Reference: Jeevanjee 2023
    """
    PS = 1e5 # Pa
    P_REF = 0.5e5  # Pa ; this accounts for the pressure broadening 
    return q_co2 * PS**2 / (P_REF * g) # kg m-2

def column_mass_o3():
    """
    Returns the total mass of O3; typically constant in our case
    """
    
    return 6.5 # kg m-2


#########################

def h2o_abs_coef_nu_(nu):
    """
    Returns the mass absorption coefficients of water vapour

    Reference: Jeevanjee 2023; valid only for the atmospheric window region
    """

    K1 = 130. # m2 kg-1
    K2 = 8. # m2 kg-1
    NU1 = 150. # cm-1
    NU2 = 1500. # cm-1
    L1 = 56. # cm-1
    L2 = 40. # cm-1

    if 10 <= nu <= 1500:
        return np.max((K1 * np.exp( - abs( nu - NU1) / L1), 
                   K2 * np.exp( - abs( nu - NU2) / L2)))
    else:
        return 0.

def co2_abs_coef_nu_(nu):
    """
    Returns the mass absorption coefficients of CO2

    Reference: Jeevanjee 2023; valid only for the bad centred at ~675 cm-1
    """
        
    K0 = 240. # m2 kg-1
    L0 = 10.5 # cm-1
    NU0 = 667. # cm-1

    if 500 <= nu <= 850:
        return K0 * np.exp( - np.abs( nu - NU0) / L0)
    else:
        return 0.


def o3_abs_coef_nu_(nu):
    """
    Returns the mass absorption coefficients of ozone

    valid only for the bad centred at ~1040 cm-1
    """
    K0 = 45. # m2 kg-1
    L0 = 10.5 # cm-1
    NU0 = 1040. # cm-1

    if 880 <= nu <= 1200:
        return K0 * np.exp( - np.abs( nu - NU0) / L0)
    else:
        return 0.

# vectorising; so that numpy arrays can be passed directly
h2o_abs_coef_nu = np.vectorize(h2o_abs_coef_nu_)
co2_abs_coef_nu = np.vectorize(co2_abs_coef_nu_)
o3_abs_coef_nu = np.vectorize(o3_abs_coef_nu_)
