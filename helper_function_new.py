# imports
import matplotlib.pyplot as plt
from os import path
import numpy as np
import scipy
import scipy.signal
from qutip import *

# Definitions and parrameters
N = 6  # number of qubits in chain
c_light = 3E-1  # 0.3 m/ns
d = 0.005  # distance between qubits = 5 mm
d_mirror = 0.005  # distance between cavity/mirror and edge qubits (n=1,n=N)
L_mirror = 2 * d_mirror + (N - 1) * d  # Total length of mirror/cavity


def err_val_omega(err, omega0, gamma0, N, d, c_light=3E-1):
    # function that creates random omega,phi and gammas according to err size
    # 'err' = the maximal error percentage between transmons,
    # 'omega0' = the mean frequency of a qubit,
    # 'N' = number of qubits
    omega_val = [np.random.uniform((omega0 * (1 - err)), (omega0 * (1 + err))) for _ in
                 range(N)]  # list of freq     with random error
    # phi_val = [np.round(omega * d / c_light / (2 * np.pi), 5) for omega in omega_val]  # ## GUY and SASHA : keep     the phi the same for all the qubits
    gamma_val = [gamma0 for _ in range(N)]  # fixed gamma
    return omega_val, gamma_val


def err_val_gamma(err, omega0, gamma0, N, d, c_light=3E-1):
    # function that creates random omega,phi and gammas according to err size
    # 'err' = the maximal error percentage between transmons,
    # 'omega0' = the mean frequency of a qubit,
    # 'N' = number of qubits
    omega_val = [omega0 for _ in range(N)]  # fixed omega
    gamma_val = [np.random.uniform((gamma0 * (1 - err)), (gamma0 * (1 + err))) for _ in
                 range(N)]  # list of freq with random error
    # phi_val = [(np.round(omega0) * d / c_light / (2 * np.pi), 5) for _ in range(N)]  # fixed phi

    return omega_val, gamma_val


def GreenFcavity(m, n, phi, r):
    # function that creates the Green function of a photon in the system of waveguide+cavity
    # imput params: 'm,n' = qubit indices,
    # 'phi' = phase gained between two neighboring qubits m and n
    # 'r' = mirror/cavity reflection coefficient;
    j = complex(0 + 1j)
    zm = m + 1 - (N + 1) / 2
    zn = n + 1 - (N + 1) / 2

    G0 = np.exp(j * phi * abs(zm - zn))

    rB = r * np.exp(-j * phi * (N + 1))  # phi = w*d/c ==> phi*(N+1) = w*L/c = kL
    return G0 + 2 * rB / (1 - rB ** 2) * (np.cos(phi * (zm + zn)) + rB * np.cos(phi * (zm - zn)))


##YULI: didn't change function since it's not in use..
def construct_H0(N, phi, r=0, dim=2, alpha=0):
    # function that returns a N-qubit Hamiltonian for the system
    # (without driving).
    # Here we use parametrize the system (frequency,anharmonicity)
    # relative to the decay rate to the waveguide gamma.
    # The Hamiltonian is a sum of terms consisting of:
    # raising operator in index 'm'
    # lowering operator in index 'n'
    # identity (i.e. does not act) on all other indices
    # input params: 'N' = number of qubits in model,
    # 'phi' = list of phases gained between two neighboring qubits,
    # 'r' = mirror/cavity reflection coefficient,
    # 'dim' = number of transmon levels considered in the sim.
    # (i.e. dim = 2 is an ideal two level system, which is
    # equivalent to infinite anharmonicity)
    # 'alpha' = anharmonicity;

    j = complex(0 + 1j)
    H0 = tensor([Qobj(0)])  # initialize an empty tensor.
    for m in range(N):
        for n in range(N):
            H = [qeye(dim)] * N
            H[m] = H[m] * create(dim)
            H[n] = H[n] * destroy(dim)
            H0 = H0 + tensor(H) * (-j) * GreenFcavity(m, n, phi, r)

    for n in range(N):
        H = [qeye(dim)] * N
        H[n] = H[n] * create(dim) * create(dim) * destroy(dim) * destroy(dim) * alpha / 2
        H0 = H0 + tensor(H)

    return H0


def construct_full_H(N, omega0, d0, gamma0, phi0, r=0, dim=2, alpha=0):
    # function that returns a N-qubit Hamiltonian for the system (without driving).
    # Here we use 'realistic' parameters rather than consider the parameters (frequency,anharmonicity) relative to
    # the decay rate to the waveguide gamma.
    # The Hamiltonian is a sum of terms consisting of:
    # raising operator in index 'm'
    # lowering operator in index 'n'

    # identity (i.e. does not act) on all other indices
    # input params: 'N' = number of qubits in model,
    # 'omega0' = list of frequencies of each transmon,
    # 'd0' = spacing between transmons,
    # 'gamma0' = list of decay rates of the transmons into the waveguide,
    # 'phi0' = fixed value of phi
    # 'r' = mirror/cavity reflection coefficient,
    # 'dim' = number of transmon levels considered in the sim,
    # 'alpha' = anharmonicity;

    j = complex(0 + 1j)

    H1 = tensor([Qobj(0)])  # initialize an empty tensor.
    for m in range(N):
        for n in range(N):
            ## GUY and SASHA : keep the phi the same for all the qubits
            gamma = np.sqrt(gamma0[m] * gamma0[
                n])  ## we normalize the gamma do it's index symmetrical according to the two qubit gammas
            # phi0 = np.round(omega0[m] * d0 / c_light / (2 * np.pi), 5)  # the phi is determined by the photon freq
            H = [qeye(dim)] * N
            H[m] = H[m] * create(dim)
            H[n] = H[n] * destroy(dim)
            H1 = H1 + tensor(H) * (-j) * gamma / (2 * np.pi) * GreenFcavity(m, n, phi0, r)

    for n in range(N):
        H = [qeye(dim)] * N
        # YULI: is true?
        H[n] = H[n] * (omega0[n] / (2 * np.pi) * create(dim) * destroy(dim) + (
                create(dim) * create(dim) * destroy(dim) * destroy(dim)) * alpha / 2)  # *alpha/2
        H1 = H1 + tensor(H)

    return H1


def construct_NumN(N, dim=2):
    # function that constructs number operator in an N-dimensional Hilbert space (N being number of qubits in the model)
    # input params: 'N' = number of qubits in model, 'dim' = number of transmon level considered in the sim
    Num0 = tensor([Qobj(0)])  # initialize an empty tensor.
    for n in range(N):
        Num = [qeye(dim)] * N
        Num[n] = Num[n] * (qutip.num(dim))
        Num0 = Num0 + tensor(Num)

    return Num0


def get_2excitations_states(N, H0, dim=2):
    # function that returns the eigenvectors and eigenvalues of states with two excitations (two photons in the qubits/waveguide).
    # This is implemented by numberically diagonalizing the Hamiltonian H0, iterating over all 2^N eigenstates and selecting only the states where expecation value of number operator equals to 2
    # input params: 'N' = number of qubits in model, 'H0' = Hamiltonian of the system without driving, 'dim' = number of transmon level considered in the sim

    e_val_2exc = []  # list to contain two-excitations eigenvalues
    e_vec_2exc = []  # list to contain two-excitations eigenvectors
    e_val, e_vec = H0.eigenstates()  # find all eigenvalues and eigenvectors of H0
    NumN = construct_NumN(N, dim)
    for i in range(len(e_val)):
        n_i = expect(NumN, e_vec[i])
        if abs(n_i - 2) < 0.1:
            e_val_2exc.append(e_val[i] / 2)
            e_vec_2exc.append(e_vec[i])

    return e_val_2exc, e_vec_2exc


def get_amplitudes(eigvec, N, dim=2):
    # function that calculates for a given two excitation eigenstate, the wavefunction amplitudes (for each combination of indicis m,n) and inverse participation ratios (iprs).
    # Amplitudes are calculated by taking the inner product of the eigenstate with each of the double excited states.
    # input params: 'eigvec' = the eigenvector of the Hamiltonian,
    # 'N' = number of qubits in model,
    # 'dim' = number of transmon level considered in the sim

    Psi0 = tensor([basis(dim, 0)] * N)  # initial state of 6 transmons in ground state
    Psis_val = np.zeros((N, N))  # initialize a N X N array to store the amplitudes
    ipr = 0  # inverse participation ratio for this eigenvector
    for m in range(N):
        for n in range(N):
            excite_mn = [qeye(dim)] * N  # initialize quantum obj of size N.
            excite_mn[m] = excite_mn[m] * create(dim)
            excite_mn[n] = excite_mn[n] * create(dim)
            Psi_mn = tensor(excite_mn) * Psi0 / np.sqrt(2)  # 2-excitation state at indices m,n
            Psi1 = Psi_mn.dag() * eigvec  # calculate inner product with desired H eigenvector
            Psis_val[m, n] = np.array(np.real(Psi1) ** 2 + np.imag(Psi1) ** 2)[
                0, 0]  # multiplication of two Qobj results in array of dim[[1],[1]].
            ipr = ipr + Psis_val[m, n] ** 2
    denom = sum(np.reshape(Psis_val, N ** 2)) ** 2
    ipr = ipr / denom
    return Psis_val, ipr


## construct a creation operator for a specific two-excitation eigenstate (eigvec).
## The operator form is $sum_{m,n} psi_{m,n} a_m^dagger a_n dagger
def construct_2excitation_creation_operator(eigvec, N, dim=2):
    # function constructing the creation operator of the two excitation states in the N-dimensional Hilbert space.
    # The operator consists of a sum of term with a raising operator in indices m,n and identity in all other indices.
    # Each term is multiplied by the corresponding amplitude for the given indicies
    # input params: 'eigvec' = the eigenvector of the Hamiltonian,
    # 'N' = number of qubits in model,
    # 'dim' = number of transmon level considered in the sim

    psi_2exc = tensor([Qobj(0)])  # initialize an empty tensor.
    psi0 = tensor([basis(dim, 0)] * N)
    for m in range(N):
        for n in range(N):
            psi = [qeye(dim)] * N
            psi[m] = psi[m] * create(dim)
            psi[n] = psi[n] * create(dim)
            # psi_mn = tensor(psi)*psi0/np.sqrt(2) # 2-excitation state at indices m,n
            psi_mn = tensor(psi) * psi0  # 2-excitation state at indices m,n
            Amp_mn = psi_mn.dag() * eigvec
            psi_2exc = psi_2exc + tensor(psi) * Amp_mn
    return psi_2exc


def exc_f_plus(t, args):
    # Construct the function describing transmon driving/excitation (exponents and/or cosine).
    # Since the "forward" and "backwards" oscillating term (e^+iwt and e^-iwt terms)
    # are proportional to different operators (a and a^dagger) we define them as different functions

    E = args['E']
    eps0 = args['eps0']
    Delta0 = args['Delta0']

    j = complex(0 + 1j)
    return E * np.exp(j * (eps0 + Delta0 / 2) * t) + E * np.exp(j * (eps0 - Delta0 / 2) * t)


def exc_f_minus(t, args):
    # Construct the function describing transmon driving/excitation (exponents and/or cosine).
    # Since the "forward" and "backwards" oscillating term (e^+iwt and e^-iwt terms) are proportional to different operators (a and a^dagger) we define them as different functions

    E = args['E']
    eps0 = args['eps0']
    Delta0 = args['Delta0']

    j = complex(0 + 1j)
    return E * np.exp(-j * (eps0 + Delta0 / 2) * t) + E * np.exp(-j * (eps0 - Delta0 / 2) * t)


def exc_f(t, args):
    # Construct the function describing transmon driving/excitation (exponents and/or cosine).

    E = args['E']
    eps0 = args['eps0']
    Delta0 = args['Delta0']

    j = complex(0 + 1j)
    return E * np.exp(-j * (eps0) * t) * np.cos(Delta0 * t)


# note: this driving assumes that the mirrors are distance d from qubits - this does not match current chip design

def construct_wg_driving_operator_plus(N, Phi, r, dim=2):
    # function constructing operator for the de-excitation of two-excitation states through the waveguide
    # input params: 'N' = number of qubits in model,
    # 'Phi' = phase gained between two neighboring qubits,
    # 'r' = mirror/cavity reflection coefficient,
    # 'dim' = number of transmon level considered in the sim

    H_wg_exc = tensor([Qobj(0)])  # initialize an empty tensor.
    j = complex(0 + 1j)
    for m in range(N):
        wgexc_plus = [qeye(dim)] * N
        zm = m + 1 - (N + 1) / 2
        # wgexc_plus[m] = (np.exp(j*Phi*zm)+r*np.exp(-j*Phi*(N+1))*np.exp(-j*Phi*zm))*destroy(dim)
        wgexc_plus[m] = (np.exp(j * Phi * zm) + r * np.exp(j * Phi * (N + 1)) * np.exp(-j * Phi * zm)) * destroy(dim)
        H_wg_exc = H_wg_exc + tensor(wgexc_plus)
    return H_wg_exc


def construct_wg_driving_operator_minus(N, Phi, r, dim=2):
    # function constructing operator for the de-excitation of two-excitation states through the waveguide
    # input params: 'N' = number of qubits in model, 'Phi' = phase gained between two neighboring qubits,
    # 'r' = mirror/cavity reflection coefficient,
    # 'dim' = number of transmon level considered in the sim

    H_wg_exc = tensor([Qobj(0)])  # initialize an empty tensor.
    j = complex(0 + 1j)
    for m in range(N):
        wgexc_minus = [qeye(dim)] * N
        zm = m + 1 - (N + 1) / 2
        wgexc_minus[m] = (np.exp(-j * Phi * zm) + r * np.exp(j * Phi * (N + 1)) * np.exp(j * Phi * zm)) * create(dim)
        H_wg_exc = H_wg_exc + tensor(wgexc_minus)
    return H_wg_exc

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
