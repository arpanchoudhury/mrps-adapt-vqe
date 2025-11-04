from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.circuit.library import UCC
import numpy as np
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import CG, L_BFGS_B
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator import generate_fermionic_excitations
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator




def get_geom(dist):

    xyz = '''H 0.0 0.0 0.0;
             H 0.0 0.0  {};
             H {}  0.0 0.0;
             H {}  0.0  {}'''.format(dist,dist,dist,dist)
    return xyz


noiseless_estimator = Estimator()

#dist_array = np.linspace(0.5, 3.5, num=int((3.5-0.5)/0.25)+1)
dist = 2.0

numpy_solver = NumPyMinimumEigensolver()
numpy_energy_list = []
nuc_rep_list = []


xyz = get_geom(dist)

#print(xyz)
driver = PySCFDriver(
atom=xyz,
basis="sto3g",
charge=0,
spin=0,
unit=DistanceUnit.ANGSTROM)

problem = driver.run()

mapper = JordanWignerMapper()
qiskit_hamiltonian = problem.hamiltonian
qiskit_second_q_op = qiskit_hamiltonian.second_q_op()  # Hamiltonian in second-quantized form
qiskit_qubit_op = mapper.map(qiskit_second_q_op)  # Hamiltonian in pauli string form

qiskit_nuclear_repulsion_energy = qiskit_hamiltonian.nuclear_repulsion_energy
nuc_rep_list.append(qiskit_nuclear_repulsion_energy)

num_particles = problem.num_particles
num_spatial_orbitals = problem.num_spatial_orbitals
num_spin_orbitals = int(2 * num_spatial_orbitals)

qiskit_result = numpy_solver.compute_minimum_eigenvalue(operator=qiskit_qubit_op)
numpy_energy_list.append(qiskit_result.eigenvalue)
print('exact energy =', qiskit_result.eigenvalue+qiskit_nuclear_repulsion_energy)



optimizer = L_BFGS_B(maxiter=10000)

hf_state = HartreeFock(
    num_spatial_orbitals,
    num_particles,
    qubit_mapper=mapper,

)


ansatz = UCC(
    num_spatial_orbitals,
    num_particles,
    qubit_mapper=mapper,
    initial_state=hf_state,
    excitations='sd',
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
#nrg_list = []
'''def store_intermediate_result(eval_count, parameters, mean, std):
    #nrg_list.append(mean+qiskit_nuclear_repulsion_energy)
    print(mean+qiskit_nuclear_repulsion_energy, flush=True)
'''

vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer)#, callback=store_intermediate_result)

from adapt_vqe.adapt_custom_pool import AdaptVQE
adapt_vqe = AdaptVQE(vqe, gradient_threshold=1e-08, eigenvalue_threshold=1e-08, original_excitations_list = new_qubit_pool)
result = adapt_vqe.compute_minimum_eigenvalue(qiskit_qubit_op)
#result = vqe.compute_minimum_eigenvalue(qiskit_qubit_op)

########################################################################



print(f"=================== VQE calculation: (Distance = {dist} ang) ==============================")
print(result)
print("####################################")
optimal_circ = result.optimal_circuit
number_gates = optimal_circ.decompose().decompose().decompose().count_ops()
excitation_list = optimal_circ._get_excitation_list()
num_param_adapt = optimal_circ.num_parameters
eigenvalue = result.eigenvalue
print(len(result.optimal_parameters))
print("####################################")
print(number_gates)
print("####################################")
print(excitation_list)
print("####################################")
print(num_param_adapt)
print("####################################")
print(f"Exact electronic energy = {qiskit_result.eigenvalue} Ha")
print(f"ADAPT-VQE electronic energy = {eigenvalue} Ha" )
print(f"ADAPT-VQE energy error = {eigenvalue - qiskit_result.eigenvalue} Ha")

print("####################################")
