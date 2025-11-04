from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.circuit.library import UCC
import numpy as np
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import CG, L_BFGS_B, SLSQP
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator import generate_fermionic_excitations
from qiskit.primitives import Estimator
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit.circuit.library import TwoLocal
from pyscf import gto, scf, lo
from pyscf import ao2mo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
import argparse as ap
#from adapt_vqe.new_adapt import AdaptVQE


#parser = ap.ArgumentParser()
#parser.add_argument('--minrange', type=str, default='290')


def get_geom():
    xyz = ''' C                  1.02483238    0.00000000    0.00000000;
              C                  0.00000000   -1.02483238    0.00000000;
              C                 -1.02483238    0.00000000    0.00000000;
              C                  0.00000000    1.02483238    0.00000000;
              H                  2.10052774    0.00000000    0.00000000;
              H                  0.00000000   -2.10052774    0.00000000;
              H                 -2.10052774    0.00000000    0.00000000;
              H                  0.00000000    2.10052774    0.00000000'''
    return xyz



#dist_array = np.linspace(0.5, 3.5, num=int((3.5-0.5)/0.25)+1)
#dist = 2.0

numpy_solver = NumPyMinimumEigensolver()
numpy_energy_list = []
nuc_rep_list = []


xyz = get_geom()
mol = gto.M(atom = xyz, basis = '6-31G', symmetry=False)

# RHF calculation
rhf = scf.RHF(mol)
rhf.kernel()

# orbital localization
#nao_localized = lo.orth_ao(rhf, 'nao')
#boys_loc = lo.boys.Boys(mol, mo_coeff=rhf.mo_coeff[:,[4,5,6,7,10,11,12,13]]).kernel()
from pyscf.tools.molden import load
#mo_loc = load("cbd_d4h_casscf44.molden")
loc_MO_pipek = rhf.mo_coeff#mo_loc[2]

#print(nao_localized)



# 1- and 2-electron integral
hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
hcore_lmo = np.einsum('pi,pq,qj->ij', loc_MO_pipek, hcore_ao, loc_MO_pipek)
eri_4fold_ao = mol.intor('int2e_sph')
eri_4fold_lmo = ao2mo.incore.full(eri_4fold_ao, loc_MO_pipek)
print('integrals done')
pyscf_hamiltonian_lmo = ElectronicEnergy.from_raw_integrals(hcore_lmo, eri_4fold_lmo)
print('hamiltonian done')

pyscf_hamiltonian_lmo.nuclear_repulsion_energy = rhf.energy_nuc()
print(pyscf_hamiltonian_lmo.nuclear_repulsion_energy)
mapper = JordanWignerMapper()
'''second_q_op_lmo = pyscf_hamiltonian_lmo.second_q_op()  # Hamiltonian in second-quantized form
qubit_op_lmo = mapper.map(second_q_op_lmo)  # Hamiltonian in pauli string form
'''
#print(qubit_op_lmo)
print('qubit hamiltonian done')
print()

#qubit_op_lmo.num_qubits
original_as_transformer = ActiveSpaceTransformer(4, 4, active_orbitals=[12,13,14,16])

total_elec = sum(mol.nelec)
total_orb = rhf.mo_coeff.shape[1] 
occupation_list = list(rhf.mo_occ/2)
original_as_transformer.prepare_active_space(total_elec, total_orb, occupation_alpha=occupation_list, occupation_beta=occupation_list)

original_as_hamiltonian = original_as_transformer.transform_hamiltonian(pyscf_hamiltonian_lmo)
original_as_shift = original_as_hamiltonian.constants['ActiveSpaceTransformer']
print('original_as_shift',original_as_shift)
original_as_second_q_op_lmo = original_as_hamiltonian.second_q_op()
original_as_qubit_op_lmo = mapper.map(original_as_second_q_op_lmo)

#as_num_particles = (1,1)
#as_num_spatial_orbitals = n_active_orb
#as_num_spin_orbitals = int(2 * as_num_spatial_orbitals)

# ED on active space
original_as_exact_result = numpy_solver.compute_minimum_eigenvalue(operator=original_as_qubit_op_lmo)
print('original_as_exact_result.eigenvalue', original_as_exact_result.eigenvalue, flush=True)

optimizer = L_BFGS_B(maxiter=10000)

num_spatial_orbitals = 4
num_spin_orbitals = int(2 * num_spatial_orbitals)


num_particles = (2,2)
singles_excitations_list = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, generalized=True)

doubles_excitations_list = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, generalized=True)




all_excitations = doubles_excitations_list + singles_excitations_list

def custom_excitation(num_particles, num_spatial_orbitals):
    return all_excitations    


hf_state = HartreeFock(
    num_spatial_orbitals,
    num_particles,
    qubit_mapper=mapper
)


ansatz = UCC(
    num_spatial_orbitals,
    num_particles,
    qubit_mapper=mapper,
    initial_state=hf_state,
    excitations=custom_excitation,
    generalized=True
)



fermionic_pool = ansatz.operators
print(fermionic_pool)
print('fermionic_pool length =', len(fermionic_pool))
from qiskit.quantum_info import SparsePauliOp
#qubit_pool = [SparsePauliOp(op.primitive) for op in fermionic_pool]


new_qubit_pool = []
for i in range(len(fermionic_pool)):
    for j in range(len(fermionic_pool[i])):
        new_qubit_pool.append(fermionic_pool[i][j])

print(new_qubit_pool)
print('qubit_pool length =', len(new_qubit_pool))
nrg_list = []
def store_intermediate_result(eval_count, parameters, mean, std):
    nrg_list.append(mean)
    print(mean, flush=True)


noiseless_estimator = Estimator()
vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result)

from adapt_vqe.adapt_custom_pool import AdaptVQE
adapt_vqe = AdaptVQE(vqe, gradient_threshold=1e-08, eigenvalue_threshold=1e-08, original_excitations_list = new_qubit_pool)
#adapt_vqe = AdaptVQE(vqe, gradient_threshold=1e-08, eigenvalue_threshold=1e-08)
result = adapt_vqe.compute_minimum_eigenvalue(original_as_qubit_op_lmo)




print(f"===================ADAPT-VQE calculation ==============================")
print(result)
print("####################################")
optimal_circ = result.optimal_circuit
number_gates = optimal_circ.decompose().decompose().decompose().count_ops()
#excitation_list = optimal_circ._get_excitation_list()
#num_param_adapt = optimal_circ.num_parameters
eigenvalue = result.eigenvalue
print("vqe E_elec =", eigenvalue)
print("exact E_elec = ",original_as_exact_result.eigenvalue)

print("####################################")
print(number_gates)
#print("####################################")
#print(excitation_list)
#print("####################################")
#print(num_param_adapt)
#print("####################################")
#print(f"Exact electronic energy = {qiskit_result.eigenvalue} Ha")
#print(f"ADAPT-VQE electronic energy = {eigenvalue} Ha" )
print(f"ADAPT-VQE energy error = {eigenvalue - original_as_exact_result.eigenvalue} Ha")
