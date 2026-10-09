r""" Analytic test of the diagrams for n degenerate orbitals coupled to a bath level at zero energy

The local Hamiltonian H_loc = 0 gives a degenerate atomic pseudo-particle propagator

.. math::
    G(\tau) = -e^{-\alpha \tau}, \quad \alpha = \ln(N) / \beta,

with N = 2^n the dimension of the local Hilbert space of the n orbitals, such that Z = 1.
The hybridization is a single bath pole at zero energy with unit weight in every orbital,

.. math::
    \Delta_{ab}(\tau) = -\frac{1}{2} \delta_{ab},

given directly as a pole, so no hybridization fit enters.

Since G is a single exponential it factorizes out of all self-energy and single-particle
Green's function diagrams, and the diagram integrals become polynomials in tau. E.g. the
OCA self-energy, with O_a running over the creation and annihilation operators,

.. math::
    \Sigma_{OCA}(\tau) = \sum_{abcd} \int_0^\tau d\tau_1 \int_0^{\tau_1} d\tau_2 \,
        \Delta_{ab}(\tau - \tau_2) \Delta_{cd}(\tau_1) \,
        O_a G(\tau - \tau_1) O_c G(\tau_1 - \tau_2) O_b G(\tau_2) O_d

simplifies to

.. math::
    \Sigma_{OCA}(\tau) = (-1/2)^2 G(\tau) A \int_0^\tau d\tau_1 \int_0^{\tau_1} d\tau_2
    = (-1/2)^2 G(\tau) \, n (n - 1) \, \tau^2 / 2,

where the sum over the operators, with the sign of the crossing diagram, gives the
combinatorial factor A = n (n - 1) times the identity.

The same reduction gives the self-energy and the single-particle Green's function up to
third order, which are compared on the atomic propagator with the dense evaluator and with the
block-sparse evaluator, using both the total density and the automatic partition. Third order
diagrams that carry three distinct orbitals exist only for n >= 3, beyond the analytic tests
api_one_fermion.py (n = 1) and api_two_fermions.py (n = 2).
The model is orbital and particle-hole symmetric, with a constant hybridization and every
reference a real function of tau times the identity, so the test checks the combinatorial
factors, signs and integration domains of the diagrams, but not the time arguments of the
hybridization lines, a transposed or conjugated index, a permutation of the orbitals or a
reflection tau -> beta - tau of g, which those tests and api.py check.
"""

import numpy as np

from triqs.operators import Operator, n as n_op

from triqs_xca import Solver
from triqs_xca.solver import pseudo_particle_block_gf_to_dense


def K(n):
    """ Combinatorial factor of the third order self-energy """
    return n * (4*n**2 - 9*n + 4)


def C_1(n):
    """ Combinatorial factor of the third order single-particle Green's function
    with two internal vertices on each side of tau """
    return (n - 2) * n + (n - 1) * (n - 2)


def C_2(n):
    """ Combinatorial factor of the third order single-particle Green's function
    with three internal vertices on one side of tau and one on the other """
    return (n - 1)**2


class AnalyticDiagrams:

    def __init__(self, n, tau, beta):
        """ Atomic propagator G0, self-energy Sigma (Fock basis) and single-particle Green's
        function g of n orbitals on the times tau. Sigma and g are dicts over the order,
        cumulative up to that order. """

        N = 2**n
        I_n = np.eye(n)[None, :, :]
        I_N = np.eye(N)[None, :, :]
        t = tau[:, None, None]

        alpha = np.log(N) / beta
        self.G0 = -np.exp(-alpha * t) * I_N

        dSigma = {
            1: -(-0.5) * n * self.G0,
            2: (-0.5)**2 * t**2 / 2 * n * (n - 1) * self.G0,
            3: (-0.5)**3 * t**4 / 24 * (-K(n)) * self.G0,
            }

        dg = {
            1: -0.5 * I_n + 0 * t,
            2: -0.5 * (beta - t) * t * (n - 1) / 2 * I_n,
            3: -(-0.5)**2 * (
                (beta - t)**2 * t**2 / 4 * C_1(n)
                + (beta - t)**3 * t / 6 * C_2(n)
                + (beta - t) * t**3 / 6 * C_2(n)
                ) / 2 * I_n,
            }

        self.Sigma = {order: sum(dSigma[o] for o in range(1, order + 1)) for order in dSigma}
        self.g = {order: sum(dg[o] for o in range(1, order + 1)) for order in dg}


def max_abs(x):
    return np.max(np.abs(x))


def plot_comparison(tau, ana, Sigma, g):
    """ Order by order contributions to the first diagonal element, numeric (crosses) and analytic (lines) """

    from triqs.plot.mpl_interface import plt
    if plt.get_backend().lower() == 'agg': return

    plt.figure(figsize=(9, 5))
    subp = [2, 3, 1]

    for name, num, ref in ((r'\Sigma', Sigma, ana.Sigma), ('g', g, ana.g)):
        for order in ref:
            d_num = num[order] - num.get(order - 1, 0)
            d_ref = ref[order] - ref.get(order - 1, 0)

            plt.subplot(*subp); subp[-1] += 1
            plt.plot(tau, d_num[:, 0, 0].real, 'x', label='numeric')
            plt.plot(tau, d_ref[:, 0, 0], '-', label='analytic')
            plt.xlabel(r'$\tau$')
            plt.ylabel(f'${name}^{{({order})}}(\\tau)$')
            plt.legend()

    plt.tight_layout()
    plt.show()


def check_analytic_diagrams(n, conserved_operators, beta, w_max, eps, verbose=False):
    """ Diagrams of n orbitals up to third order on the atomic propagator, against the closed forms. """

    print('='*72)
    print(f'n = {n}, conserved_operators = {conserved_operators}')
    print('='*72)

    # The pole hybridization and the atomic propagator are exact, the agreement is limited by the
    # DLR accuracy eps of the diagram integrals
    atol = 1000 * eps

    use_dense_solver = conserved_operators == 'none'

    # With H_loc = 0 every orbital density is conserved, so the automatic partition has only
    # one-state blocks, while for n > 1 the total density partition has multi-state blocks
    multi_state_blocks = conserved_operators == 'total_density' and n > 1

    N_op = sum(n_op('0', i) for i in range(n))

    conserved_operators = dict(
        none=[],
        total_density=[N_op],
        automatic='automatic',
        )[conserved_operators]

    S = Solver(
        0.0 * Operator(), beta, w_max, eps, gf_struct=[('0', n)],
        conserved_operators=conserved_operators, verbose=False,
        )

    assert S.use_dense_solver == use_dense_solver
    if use_dense_solver:
        assert S.atom_diag.n_subspaces == 1
    else:
        assert S.atom_diag.n_subspaces > 1, 'vacuous test, no block structure'

    if multi_state_blocks:
        assert max(S.atom_diag.get_subspace_dims()) > 1, 'vacuous test, no multi-state blocks'

    S.set_hybridization_poles_and_coefficients(np.array([0.0]), np.eye(n, dtype=complex)[None, :, :])
    S.init_diagram_evaluator()

    to_dense = lambda G: pseudo_particle_block_gf_to_dense(G, S.atom_diag).data

    tau = np.array([float(t) for t in S.mesh_tau])
    ana = AnalyticDiagrams(n, tau, beta)

    # The third order contributions are well above the tolerance
    assert max_abs(ana.Sigma[3] - ana.Sigma[2]) > 1e3 * atol, 'vacuous test, third order Sigma is not resolved'
    assert max_abs(ana.g[3] - ana.g[2]) > 1e3 * atol, 'vacuous test, third order g is not resolved'

    G0 = to_dense(S.G0)
    print(f'G0_diff = {max_abs(G0 - ana.G0):2.2E}')
    np.testing.assert_allclose(G0, ana.G0, rtol=0, atol=atol)

    Sigma, g = dict(), dict()

    for order in ana.Sigma:
        Sigma[order] = to_dense(S.eval_pseudo_particle_self_energy(S.G0, order))
        g[order] = S.eval_single_particle_greens_function(S.G0, order).data

        print(f'order = {order}: Sigma_diff = {max_abs(Sigma[order] - ana.Sigma[order]):2.2E}, '
              f'g_diff = {max_abs(g[order] - ana.g[order]):2.2E}')

        np.testing.assert_allclose(
            Sigma[order], ana.Sigma[order], rtol=0, atol=atol, err_msg=f'Sigma order {order}')
        np.testing.assert_allclose(
            g[order], ana.g[order], rtol=0, atol=atol, err_msg=f'g order {order}')

    if verbose:
        plot_comparison(tau, ana, Sigma, g)


def test_analytic_diagrams(beta=2.0, w_max=10.0, eps=1e-12, verbose=False):

    for n in (1, 2, 3):
        for conserved_operators in ('none', 'total_density', 'automatic'):
            check_analytic_diagrams(n, conserved_operators, beta, w_max, eps, verbose)


if __name__ == '__main__':

    test_analytic_diagrams(verbose=False)
