r""" Interacting spinful fermion coupled to one bath level, against the NESSi-ppsc code.

Local Hamiltonian and hybridization function, the same for both spins,

.. math::
    H = U n_\uparrow n_\downarrow + (\epsilon_1 - \mu) (n_\uparrow + n_\downarrow), \quad
    \Delta(i\omega_n) = \frac{V^2}{i\omega_n - \epsilon_2}.

The single-particle Green's function in the atomic limit (V = 0, order one), NCA (order one) and OCA
(order two) is compared pointwise with NESSi-ppsc reference data on a uniform imaginary time grid, for the
dense and the block-sparse evaluator. NESSi-ppsc is an independent implementation of the pseudo-particle
expansion. The hybridization function is compared as well, which checks that the two codes solve the same
problem. """

import numpy as np

from h5 import HDFArchive

from triqs.gfs import inverse, iOmega_n, make_gf_dlr_imfreq, make_gf_dlr_imtime, make_gf_imtime
from triqs.operators import n

from triqs_xca import Solver


def max_abs(x):
    return np.max(np.abs(x))


def load_reference(cases, filename='nessi_ppsc_one_spinful_fermion.ref.h5'):
    """ NESSi-ppsc imaginary times, Green's function and hybridization function of each case. """
    with HDFArchive(filename, 'r') as A:
        return {case: {key: np.array(A[case][key]) for key in ('tau', 'g_tau', 'delta_tau')} for case in cases}


def test_one_spinful_fermion(
        beta=10.0, U=1.0, mu=0.0, eps1=-0.1, eps2=0.1, V=0.1, w_max=40.0, eps=1e-12,
        ppsc_tol=1e-9, ppsc_maxiter=100, verbose=False):

    spin_names = ['up', 'do']
    gf_struct = [[spin, 1] for spin in spin_names]

    n_up, n_do = n('up', 0), n('do', 0)
    H = U * n_up * n_do + (eps1 - mu) * (n_up + n_do)

    # The self-consistency stops when the update of the pseudo-particle propagator is below ppsc_tol, so the
    # propagator, and the Green's function evaluated from it, deviate from the converged solution by a few ppsc_tol
    atol_g = 10 * ppsc_tol

    # The hybridization function enters through its DLR representation, accurate to eps
    atol_Delta = 100 * eps

    # Hybridization strength and expansion order of the atomic limit, NCA and OCA
    cases = dict(
        atom=dict(V_case=0.0, max_order=1),
        nca=dict(V_case=V, max_order=1),
        oca=dict(V_case=V, max_order=2),
        )

    def solve(conserved_operators, V_case, max_order):
        """ Self-consistent solution at hybridization strength V_case and expansion order max_order. """

        S = Solver(H, beta, w_max, eps, gf_struct, conserved_operators=conserved_operators, verbose=False)
        assert S.use_dense_solver == (conserved_operators == [])

        for _, Delta_tau in S.Delta_tau:
            Delta_w = make_gf_dlr_imfreq(Delta_tau)
            Delta_w << V_case**2 * inverse(iOmega_n - eps2)
            Delta_tau << make_gf_dlr_imtime(Delta_w)

        S.solve(max_order=max_order, tol=ppsc_tol, maxiter=ppsc_maxiter, verbose=False)
        assert S.diff_G < ppsc_tol, f'not converged after {S.n_ppsc_iter} iterations: diff_G = {S.diff_G:2.2E}'

        return S

    ref = load_reference(cases)

    # The comparison resolves the hybridization (atom and nca) and the expansion order (nca and oca)
    for a, b in (('atom', 'nca'), ('nca', 'oca')):
        assert max_abs(ref[a]['g_tau'] - ref[b]['g_tau']) > 100 * atol_g, f'vacuous test, {a} and {b} agree'

    for conserved_operators, tag in (([], 'dense'), ('automatic', 'block-sparse')):

        g = dict()
        for case, params in cases.items():

            S = solve(conserved_operators, **params)

            n_tau = len(ref[case]['tau'])
            G_tau = make_gf_imtime(S.G_tau, n_tau=n_tau)
            Delta_tau = make_gf_imtime(S.Delta_tau, n_tau=n_tau)

            # The uniform grid of the solver and the imaginary times of the reference coincide
            tau = np.array([float(t) for t in G_tau.mesh])
            np.testing.assert_allclose(tau, ref[case]['tau'], rtol=0, atol=1e-12)

            err_g = max(max_abs(G_tau[spin].data[:, 0, 0] - ref[case]['g_tau']) for spin in spin_names)
            # Input hybridization, not modified by solve: checks that both codes solve the same problem
            err_Delta = max(max_abs(Delta_tau[spin].data[:, 0, 0] - ref[case]['delta_tau']) for spin in spin_names)

            print(f'{tag}, {case}: max|g - g_nessi| = {err_g:2.2E}, max|Delta - Delta_nessi| = {err_Delta:2.2E}')

            assert err_g < atol_g, f'{tag}, {case}: max|g - g_nessi| = {err_g:2.2E} exceeds {atol_g:1.0E}'
            assert err_Delta < atol_Delta, \
                f'{tag}, {case}: max|Delta - Delta_nessi| = {err_Delta:2.2E} exceeds {atol_Delta:1.0E}'

            g[case] = G_tau[spin_names[0]].data[:, 0, 0]

        if verbose:
            plot_comparison(tau, ref, g, tag)


def plot_comparison(tau, ref, g, tag):
    """ Green's function of the solver (crosses) and NESSi-ppsc (lines) for each case, and their difference.
    Shown only with an interactive backend. """

    from triqs.plot.mpl_interface import plt
    if plt.get_backend().lower() == 'agg': return

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for k, (case, g_case) in enumerate(g.items()):
        axes[0].plot(tau, ref[case]['g_tau'], '-', color=f'C{k}', label=case)
        axes[0].plot(tau[::25], g_case[::25].real, 'x', color=f'C{k}')
        axes[1].plot(tau, g_case.real - ref[case]['g_tau'], '-', color=f'C{k}', label=case)

    axes[0].set_ylabel(r'$g(\tau)$')
    axes[1].set_ylabel(r'$g(\tau) - g_{\mathrm{NESSi}}(\tau)$')
    for ax in axes:
        ax.set_xlabel(r'$\tau$')
        ax.legend()

    fig.suptitle(tag)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    test_one_spinful_fermion(verbose=False)
