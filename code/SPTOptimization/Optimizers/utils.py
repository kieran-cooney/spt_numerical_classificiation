from itertools import accumulate, count

import numpy as np
import tenpy.linalg.np_conserved as npc

from SPTOptimization.utils import (
    get_transfer_matrix_from_tp_unitary_and_b_tensor,
    multiply_transfer_matrices,
    multiply_transfer_matrices_from_right,
    get_right_identity_environment_from_tp_tensor,
    get_left_identity_environment_from_tp_tensor
)

from SPTOptimization.gradients import (
    expectation_gradient_from_environments_and_b_tensor
)


def max_expectation(grad):
    """
    Given a tensor grad which represents an contraction of tensors yielding an
    expectation value except for one unitary which has not been contracted,
    return the largest possible value for the expectation over all such
    unitaries.

    Parameters
    ----------
    grad: tenpy.linalg.np_conserved.Array with legs 'p' and 'p*'
        A tensor which if contracted against an operator yields an expectation.
    Returns
    -------
    float
        The best possible expectation over all unitaries.
    """
    S = npc.svd(grad.complex_conj(), compute_uv=False)
    return np.sum(S)


def one_site_optimization(grad):
    """
    Given a tensor grad which represents an contraction of tensors yielding an
    expectation value except for one unitary which has not been contracted,
    return the largest possible value for the expectation over all such
    unitaries and the unitary which achieves this maximum.

    Parameters
    ----------
    grad: tenpy.linalg.np_conserved.Array with legs 'p' and 'p*'
        A tensor which if contracted against an operator yields an expectation.
    Returns
    -------
    pair of tenpy.linalg.np_conserved.Array with legs 'p' and 'p*' and float
        The best possible expectation over all unitaries and the unitary
        attaining this maximum.
    """
    U, S, VH = npc.svd(
        grad.conj(),
        compute_uv=True,
        inner_labels=['i', 'i*']
    )

    out_unitary = npc.tensordot(U, VH, [['i',], ['i*',]])

    return (np.sum(S), out_unitary)


def one_site_optimization_sweep_right(left_environment, mps_tensors,
    unitaries, mps_tensors_bra=None, right_environments=None):
    """
    Given an MPS psi with environment immediately to the left of it, and a
    set of one site unitaries, optimize each of the unitaries in turn and
    update in place. Output a list of the expectations found at each step.

    Parameters
    ----------
    left_environment: tenpy.linalg.np_conserved.Array
        Evnironment to the left of psi. Tensor with legs 'vR' and 'vR*'.
    mps_tensors: list of tenpy.linalg.np_conserved.Array
        The i-th element should correspond to the "B" tensor (or some other
        gauge) of some MPS psi. Assumed the tensors belong to adjacent sites
        in psi.
    unitaries: List of tenpy.npc.Array
        List of single site operators. Should be same length mps_tensors.
    mps_tensors_bra: list of tenpy.linalg.np_conserved.Array
        If provided, then will be used as tensors for the "bra" in the
        expectation value expression. If not provided, then mps_tensors are
        used. 
    right_environments: list of tenpy.linalg.np_conserved.Array
        A list of 2-leg tensors with legs 'vL', 'vL*'. The i-th element
        corresponds to the contribution to the expectation immediately to the
        right of site i, prior to any unitaries being updated.
        If not provided, will be calculated.
    Returns
    -------
    triple of (list of float, tenpy.linalg.np_conserved.Array, list of
    tenpy.linalg.np_conserved.Array)
        For the first element in the triple, the i-th element corresponds to
        the expectation after optimizing the unitary at the i-th site.
        For the second, last left environment found with legs 'vR' and 'vR*'.
        The third is a list of the calculated transfer matrices.
    """
    num_sites = len(mps_tensors)

    last_right_environment = get_right_identity_environment_from_tp_tensor(
        mps_tensors[-1]
    )

    if mps_tensors_bra is None:
        mps_tensors_bra = mps_tensors

    leg_label_pairs = [
        sorted(u.get_leg_labels(), key=lambda l: '*' in l)
        for u in unitaries
    ]
    
    leg_labels = [p[0] for p in leg_label_pairs]
    leg_labels_conj = [p[1] for p in leg_label_pairs]
    
    if right_environments is None:
        transfer_matrices = [
            get_transfer_matrix_from_tp_unitary_and_b_tensor(b1, u, b2, ll, ll_c)
            for b1, u, b2, ll, ll_c in zip(
                mps_tensors,
                unitaries,
                mps_tensors_bra,
                leg_labels,
                leg_labels_conj
            )
        ]

        # Slice so we avoid calculating the inner most right environment, which
        # is unused.
        right_environments = list(accumulate(
            transfer_matrices[:0:-1], 
            multiply_transfer_matrices_from_right,
            initial=last_right_environment
        ))[::-1]

    current_left_environment = left_environment
    expectations = list()

    hexes = zip(
        count(),
        mps_tensors,
        right_environments,
        mps_tensors_bra,
        leg_labels,
        leg_labels_conj
    )

    new_transfer_matrices = list()

    for i, b, right_environment, b_bra, ll, ll_c in hexes:
        # Compute gradient
        grad = expectation_gradient_from_environments_and_b_tensor(
            b, current_left_environment, right_environment, b_bra
        )

        # Optimize unitary, update in place
        new_expectation, new_unitary = one_site_optimization(grad)

        unitaries[i] = new_unitary
        expectations.append(new_expectation)

        # Create new left environment
        transfer_matrix = get_transfer_matrix_from_tp_unitary_and_b_tensor(
            b, new_unitary, b_bra, ll, ll_c
        )

        new_transfer_matrices.append(transfer_matrix)

        current_left_environment = multiply_transfer_matrices(
            current_left_environment,
            transfer_matrix
        )

    return (expectations, current_left_environment, new_transfer_matrices)


def one_site_optimization_sweep_left(right_environment, mps_tensors,
    unitaries, mps_tensors_bra=None, left_environments=None):
    """
    Given an MPS psi with environment immediately to the right of it, and a
    set of one site unitaries, optimize each of the unitaries in turn and
    update in place. Output a list of the expectations found at each step.

    Parameters
    ----------
    right_environment: tenpy.linalg.np_conserved.Array
        Evnironment to the left of psi. Tensor with legs 'vL' and 'vL*'.
    mps_tensors: list of tenpy.linalg.np_conserved.Array
        The i-th element should correspond to the "B" tensor (or some other
        gauge) of some MPS psi. Assumed the tensors belong to adjacent sites
        in psi.
    unitaries: List of tenpy.npc.Array
        List of single site operators. Should be same length mps_tensors.
    mps_tensors_bra: list of tenpy.linalg.np_conserved.Array
        If provided, then will be used as tensors for the "bra" in the
        expectation value expression. If not provided, then mps_tensors are
        used. 
    left_environments: list of tenpy.linalg.np_conserved.Array
        A list of 2-leg tensors with legs 'vR', 'vR*'. The i-th element
        corresponds to the contribution to the expectation immediately to the
        left of site i, prior to any unitaries being updated.
        If not provided, will be calculated.
    Returns
    -------
    triple of (list of float, tenpy.linalg.np_conserved.Array, list of
    tenpy.linalg.np_conserved.Array)
        For the first element in the triple, the i-th element corresponds to
        the expectation after optimizing the unitary at the i-th site.
        For the second, last left environment found with legs 'vR' and 'vR*'.
        The third is a list of the calculated transfer matrices.
    """
    num_sites = len(mps_tensors)

    last_left_environment = get_left_identity_environment_from_tp_tensor(
        mps_tensor
    )

    if mps_tensors_bra is None:
        mps_tensors_bra = mps_tensors

    if left_environments is None:
        transfer_matrices = [
            get_transfer_matrix_from_tp_unitary_and_b_tensor(b1, u, b2, 1)
            for b1, u, b2 in zip(mps_tensors, unitaries, mps_tensors_bra)
        ]

        # Slice so we avoid calculating the inner most right environment,
        # which is unused.
        left_environments = list(accumulate(
            transfer_matrices[:0:-1], 
            multiply_transfer_matrices_from_left,
            initial=last_left_environment
        ))[::-1]

    current_right_environment = right_environment
    expectations = list()

    quads = zip(
        count(),
        mps_tensors,
        left_environments,
        mps_tensors_bra
    )

    new_transfer_matrices = list()
    for i, b, left_environment, b_bra in quads:
        # Compute gradient
        grad = expectation_gradient_from_environments_and_b_tensor(
            b, left_environment, current_right_environment, b_bra
        )

        # Optimize unitary, update in place
        new_expectation, new_unitary = one_site_optimization(grad)

        unitaries[i] = new_unitary
        expectations.append(new_expectation)

        # Create new left environment
        transfer_matrix = get_transfer_matrix_from_tp_unitary_and_b_tensor(
            b, new_unitary, b_bra, 1
        )

        new_transfer_matrices.append(transfer_matrix)

        current_right_environment = multiply_transfer_matrices_from_right(
            current_right_environment,
            transfer_matrix
        )

    return (expectations, current_right_environment, new_transfer_matrices)
