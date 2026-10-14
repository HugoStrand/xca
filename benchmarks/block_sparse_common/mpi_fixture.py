"""Small deterministic Solver fixture for MPI ownership measurements."""
import numpy as np
from triqs.operators import n, c, c_dag
from triqs_xca import Solver


def make_solver(args, backend):
    modes = args.modes
    number = [n('f', i) for i in range(modes)]
    total = sum(number)
    h = sum((.13*(i+1)+.007*i*i)*number[i] for i in range(modes))
    for i in range(modes):
        for j in range(i):
            h += (.17+.003*(i+j))*number[i]*number[j]
            if args.partition == 'occupation' or (args.partition == 'mixed' and (i < modes//2) != (j < modes//2)):
                continue
            z = .011*(1+i+j)+.009j*(i-j)
            h += z*c_dag('f', i)*c('f', j)+z.conjugate()*c_dag('f', j)*c('f', i)
    sym = {'occupation': 'automatic', 'number': [total],
           'mixed': [sum(number[:modes//2]), sum(number[modes//2:])]}[args.partition]
    solver = Solver(h, 2., 20., 1e-10, [['f', modes]], conserved_operators=sym,
                    verbose=False, block_sparse_backend=backend,
                    block_sparse_operator_backend=args.operator_backend,
                    block_sparse_special_backend=args.special_backend)
    coefficients = np.zeros((2, modes, modes), complex)
    for i in range(modes):
        for j in range(modes):
            compatible = i == j or args.partition == 'number' or (
                args.partition == 'mixed' and (i < modes//2) == (j < modes//2))
            if compatible:
                for p in range(2):
                    coefficients[p, i, j] = .3-.1*p if i == j else .025*np.exp(1j*(i-j))
    solver.set_hybridization_poles_and_coefficients(np.array([1.3, -.8]), coefficients)
    if args.dynamic:
        solver.has_dynamic_interactions = True
        solver.dynint_ops = [total, c_dag('f', 0)*c('f', 1)]
        solver.dynint_coeffs = np.array([[[.07, 0], [0, 0]], [[0, 0], [0, .03+.01j]]], complex)
    solver.init_diagram_evaluator()
    return solver
