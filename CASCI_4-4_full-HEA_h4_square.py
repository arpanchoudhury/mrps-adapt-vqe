from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.circuit.library import UCC
import numpy as np
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit_algorithms.optimizers import CG, L_BFGS_B
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator import generate_fermionic_excitations
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
from qiskit.circuit.library import TwoLocal



def get_geom(dist):

    xyz = '''H 0.0 0.0 0.0;
             H 0.0 0.0 1.0;
             H {}  0.0 0.0;
             H {}  0.0 1.0'''.format(dist,dist)
    return xyz


noiseless_estimator = Estimator()

#dist_array = np.linspace(0.5, 3.5, num=int((3.5-0.5)/0.25)+1)
dist_array = [1.0]

numpy_solver = NumPyMinimumEigensolver()
numpy_energy_list = []
nuc_rep_list = []


for dist in dist_array:
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



    
    optimizer = L_BFGS_B(maxiter=10000)



    ansatz = TwoLocal(num_spin_orbitals, ['rx','ry','rz'], 'cx', 'linear', reps=8, insert_barriers=True)


    nrg_list = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        nrg_list.append(mean+qiskit_nuclear_repulsion_energy)
        print(mean, flush=True)



    # MP2 amplitude calculation with PYSCF calculation
    """from pyscf import gto

    mol = gto.M(atom=xyz, basis="sto3g")
    mp = mol.MP2().run()
    t2 = mp.t2

    t2_threshold = 1e-10

    num_occ_ = t2.shape[0]
    amplitudes = np.zeros(len(ansatz.excitation_list), dtype=float)
    for index, excitation in enumerate(ansatz.excitation_list):
        if len(excitation[0]) == 2:
            # Get the amplitude of the double excitation.
            [[i, j], [a, b]] = np.asarray(excitation) % num_occ_
            amplitude = t2[i, j, a - num_occ_, b - num_occ_]
            amplitudes[index] = amplitude if abs(amplitude) > t2_threshold else 0.0

    initial_point = np.tile(amplitudes, ansatz.reps)

    #initial_point = [0]*ansatz.num_parameters"""
    vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result)
    #adapt_vqe = AdaptVQE(vqe, gradient_threshold=1e-08, eigenvalue_threshold=1e-08)
    result = vqe.compute_minimum_eigenvalue(qiskit_qubit_op)
    #aerestimator = AerEstimator(run_options={"shots":None}, backend_options={"method": "statevector"}, approximation=True)
    #vqe = VQE(aerestimator, ansatz, optimizer=optimizer, callback=store_intermediate_result)
    #result = vqe.compute_minimum_eigenvalue(qiskit_qubit_op)



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
    nrg_list = np.array(nrg_list)
    #np.savetxt('CASCI_4-4_h4_square_2.5A_uccgsd_LBFGSB_full-itetration.txt', nrg_list)
    #np.savetxt('CASCI_4-4_h4_square_2.5A_uccgsd_LBFGSB_itetration.txt', np.array(result.eigenvalue_history))

print("PES distances ", dist_array)
print("Nuclear repulsion energies ", nuc_rep_list)
