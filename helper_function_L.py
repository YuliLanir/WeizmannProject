##################################################################################################
# in this helper functions we change zm for it to account for distance from mirror to the qubit
#################################################################################################

# imports
import matplotlib.pyplot as plt
from os import path
import numpy as np
import scipy
import scipy.signal
from qutip import *


#######################################################################################################
# in these next functions, we change the zm in order for it to consider driving from the mirror instead
# of from the qubit
#####do we need to change the green function as well???##############################################
# the only thing that changes is z0 - but if all qubits shift, does it matter?
#####################################################################################################

def construct_wg_driving_operator_L_plus(N, k, r, L, d, dim=2):
    # function constructing operator for the de-excitation of two-excitation states through the waveguide
    # input params: 'N' = number of qubits in model,
    # 'K' = the wave number, *not* depednent on d! (phi = k*d)
    # 'r' = mirror/cavity reflection coefficient,
    # 'dim' = number of transmon level considered in the sim
    # 'L' = length of waveguide
    d_r = ( L - (N-1) * d ) / 2 #distance between mirror to first\last qubit
    H_wg_exc = tensor([Qobj(0)])  # initialize an empty tensor.
    j = complex(0 + 1j)
    for m in range(N):
        wgexc_plus = [qeye(dim)] * N
        # distance from mirror to first qubit + distance of qubit, normalized to the middle ##is correct???
        zm = (d_r + (m + 1) * d) - L/2
        wgexc_plus[m] = (np.exp(j * k * zm) + r * np.exp(j * k * L) * np.exp(-j * k * zm)) * destroy(dim)
        H_wg_exc = H_wg_exc + tensor(wgexc_plus)
    return H_wg_exc


def construct_wg_driving_operator_L_minus(N, k, r, L, d, dim=2):
    # function constructing operator for the de-excitation of two-excitation states through the waveguide
    # input params: 'N' = number of qubits in model, 'Phi' = phase gained between two neighboring qubits,
    # 'r' = mirror/cavity reflection coefficient,
    # 'dim' = number of transmon level considered in the sim
    d_r = (L - (N - 1) * d) / 2
    H_wg_exc = tensor([Qobj(0)])  # initialize an empty tensor.
    j = complex(0 + 1j)
    for m in range(N):
        wgexc_minus = [qeye(dim)] * N
        zm = (d_r + (m + 1) * d)- L/2
        wgexc_minus[m] = (np.exp(-j * k * zm) + r * np.exp(j * k * L) * np.exp(j * k * zm)) * create(dim)
        H_wg_exc = H_wg_exc + tensor(wgexc_minus)
    return H_wg_exc


# parameters
N = 6 # number of transmons in the model
c_light = 3E-1 #  0.3 m/ns
d = 3E-4 # 0.3 mm = 0.0003 m
eps = 6.5 # for silicon
L = 0.015 #15 mm

omega0 = 6 * (2*np.pi) # omega0 ~ 6 GHz
gamma0 = 0.03 * (2*np.pi) # gamma_1D ~ 30 MHz

phi0 = np.round(omega0*d*np.sqrt(eps)/c_light,5) # phi ~ 0.1
k = np.round(omega0*np.sqrt(eps)/c_light,5) # phi = k*d


alpha = -0.25 * (2*np.pi) # 200 MHz
q_dim = 2 # Use only first two transmon levels
r_cavity = -0.99 # cavity/mirror reflection coefficient


new_d = (c_light*0.1)/(omega0*np.sqrt(eps))
print(new_d)
