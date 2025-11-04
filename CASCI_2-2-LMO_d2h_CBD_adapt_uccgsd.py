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
    xyz = '''C 0.0000000000 -0.6769380253 -0.7827569236;
             C 0.0000000000 -0.6769380253 0.7827569236;
             C 0.0000000000 0.6769380253 0.7827569236;
             C 0.0000000000 0.6769380253 -0.7827569236;
             H 0.0000000000 -1.4379809006 -1.5441628360;
             H 0.0000000000 -1.4379809006 1.5441628360;
             H 0.0000000000 1.4379809006 1.5441628360;
             H 0.0000000000 1.4379809006 -1.5441628360'''
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
mo_loc = load('cbd_d2h_hf_loc.molden')#load("cbd_d2h_casscf44_loc.molden")
loc_MO_pipek = mo_loc[2]

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
original_as_transformer = ActiveSpaceTransformer(4, 4, active_orbitals=[12,13,14,15])

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
qc_full = QuantumCircuit(num_spin_orbitals)
as_spin_orb_list = []
as_shift_list = []
# =========== Choice of active space =============

n_active_elec = 2
n_active_orb = 2

active_space_list = [[12,15],[13,14]]    # index of canonical LMOs
#active_space_list = [[12,14],[13,15]]    # index of CASSCF LMOs
exact_as_tot_energy_list = []
vqe_as_tot_energy_list = []

cas_idx_fullsystem = [[0,3], [1,2]]   # index of canonical LMOs
#cas_idx_fullsystem = [[0,2], [1,3]]   # index of CASSCF LMOs
for icount, active_space in enumerate(active_space_list):
    # select the index of active spatial orbitals [zero indexing] --->
    active_spatial_orb_idx = active_space


    as_transformer = ActiveSpaceTransformer(n_active_elec, n_active_orb, active_orbitals=active_spatial_orb_idx)
    #as_problem = as_transformer.transform(problem)
    #as_hamiltonian = as_problem.hamiltonian
    #as_second_q_op = as_hamiltonian.second_q_op()  # Hamiltonian in second-quantized form
    #as_qubit_op = mapper.map(as_second_q_op)  # Hamiltonian in pauli string form

    #as_nuclear_repulsion_energy = as_hamiltonian.nuclear_repulsion_energy

    #as_problem = as_transformer.transform(problem)

    # assuming that total system size is 4 electrons in 4 orbitals:
    as_transformer.prepare_active_space(total_elec, total_orb, occupation_alpha=occupation_list, occupation_beta=occupation_list)

    # after preparation, we can now transform the full Hamiltonian to active space Hamiltonian
    as_hamiltonian = as_transformer.transform_hamiltonian(pyscf_hamiltonian_lmo)
    #as_hamiltonian.nuclear_repulsion_energy = rhf.energy_nuc()
    as_shift = as_hamiltonian.constants['ActiveSpaceTransformer']
    print('as_shift',as_shift, flush=True)
    as_shift_list.append(as_shift)
    #print(as_hamiltonian.nuclear_repulsion_energy)

    as_second_q_op_lmo = as_hamiltonian.second_q_op()
    as_qubit_op_lmo = mapper.map(as_second_q_op_lmo)

    as_num_particles = (1,1)
    as_num_spatial_orbitals = n_active_orb
    as_num_spin_orbitals = int(2 * as_num_spatial_orbitals)

    # ED on active space
    as_exact_result = numpy_solver.compute_minimum_eigenvalue(operator=as_qubit_op_lmo)
    as_exact_tot_energy = as_exact_result.eigenvalue #+ as_shift + as_hamiltonian.nuclear_repulsion_energy 
    print(as_exact_tot_energy)
    exact_as_tot_energy_list.append(as_exact_tot_energy)


    # start HEA-VQE 
    estimator = AerEstimator(run_options={"shots":None}, backend_options={"method": "statevector"}, approximation=True)

    # change the max_iter and optimizer for HEA optimizer --->
    hea_max_iter = 10000
    optimizer_hea = CG(maxiter=hea_max_iter) # Optimizer Used for HEA ansatz

    def store_intermediate_result(eval_count, parameters, mean, std):
        print(mean, flush=True)


    # change the layer repetition --->
    repeated_layers = 8
    hea_ansatz = TwoLocal(as_num_spin_orbitals, ['rx','ry','rz'], 'cx', 'linear', reps=repeated_layers, insert_barriers=False)


    vqe = VQE(estimator, hea_ansatz, optimizer=optimizer_hea, callback=store_intermediate_result)
    print('vqe done', flush=True)
    as_result = vqe.compute_minimum_eigenvalue(operator=as_qubit_op_lmo)
    as_elec_energy = as_result.eigenvalue.real

    params_opt_hea = list(as_result.optimal_point)
    vqe_as_tot_energy = as_elec_energy# + as_hamiltonian.nuclear_repulsion_energy + as_shift
    vqe_as_tot_energy_list.append(vqe_as_tot_energy)

    hea_active_space_state = hea_ansatz.assign_parameters(params_opt_hea) # store it for all (2,2) HEA
    #hea_active_space_state_list.append(hea_active_space_state)
    #as_result_list.append(as_result)

    alpha_cas = cas_idx_fullsystem[icount]
    beta_cas = [i+num_spatial_orbitals for i in alpha_cas]
    as_spin_orb = alpha_cas + beta_cas

    as_spin_orb_list.append(as_spin_orb)

    qc_full = qc_full.compose(hea_active_space_state, qubits=as_spin_orb) # this is the circuit of the full system once the HEA active space is embedded into it






num_particles = (2,2)
singles_excitations_list = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, generalized=True)

doubles_excitations_list = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, generalized=True)


new_singles_excitations_list = []
for i in range(len(singles_excitations_list)):
    n = tuple(singles_excitations_list[i][0] + singles_excitations_list[i][1])
    n_list = list(n)
    sorted_n_list = sorted(n_list)

    subset_list = []
    for as_spin_orb in as_spin_orb_list:
        #print(as_spin_orb)
        is_subset = set(sorted_n_list).issubset(as_spin_orb)
        #print(sorted_n_list)
        subset_list.append(is_subset)

    print(subset_list)

    if not any(subset_list)==True:
        new_singles_excitations_list.append(singles_excitations_list[i])


new_doubles_excitations_list = [] = []
for i in range(len(doubles_excitations_list)):
    n = tuple(doubles_excitations_list[i][0] + doubles_excitations_list[i][1])
    n_list = list(n)
    sorted_n_list = sorted(n_list)

    subset_list = []
    for as_spin_orb in as_spin_orb_list:
        #print(as_spin_orb)
        is_subset = set(sorted_n_list).issubset(as_spin_orb)
        #print(sorted_n_list)
        subset_list.append(is_subset)

    print(subset_list)

    if not any(subset_list)==True:
        new_doubles_excitations_list.append(doubles_excitations_list[i])



all_excitations = new_doubles_excitations_list + new_singles_excitations_list

def custom_excitation(num_particles, num_spatial_orbitals):
    return all_excitations    

ansatz = UCC(
    num_spatial_orbitals,
    num_particles,
    qubit_mapper=mapper,
    initial_state=qc_full,
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



exact_as_tot_energy_list = np.array(exact_as_tot_energy_list)
vqe_as_tot_energy_list = np.array(vqe_as_tot_energy_list)
print(f"Active space E_elec error {vqe_as_tot_energy_list-exact_as_tot_energy_list} Ha")


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
