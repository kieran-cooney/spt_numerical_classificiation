from itertools import accumulate

import tenpy.linalg.np_conserved as npc

from SPTOptimization.utils import (
    get_transfer_matrix_from_tp_unitary,
    get_transfer_matrices_from_tp_unitary_list,
    multiply_transfer_matrices,
    multiply_transfer_matrices_from_right,
    get_right_identity_environment,
    get_left_identity_environment
)

from SPTOptimization.gradients import (
    expectation_gradient_from_environments
)


def right_one_site_optimization(left_environment, psi, unitaries):
    """
    Given an MPS psi with environment immediately to the left of it, and a
    set of one site unitaries, optimize each of the unitaries in turn and
    update in place. Output a list of the expectations found at each step.

    Parameters
    ----------
    left_environment: tenpy.linalg.np_conserved.Array
        Evnironment to the left of psi. Tensor with legs 'vR' and 'vR*'.
    psi: tenpy.networks.mps.MPS
        The MPS the unitaries are optimizing against.
    unitaries: List of tenpy.npc.Array
        List of single site operators. Should be same length as the number of
        sites in psi.
    Returns
    -------
    pair of (list of float, tenpy.linalg.np_conserved.Array)
        For the first element in the pair, the i-th element corresponds to
        the expectation after optimizing the unitary at the i-th site.
        For the second, last left environment found with legs 'vR' and 'vR*'.
    """
    num_sites = psi.L

    last_right_environment = get_right_identity_environment(psi, num_sites)

    transfer_matrices = get_transfer_matrices_from_tp_unitary_list(
        psi, 0, unitaries, form='B'
    )

    right_environments = list(accumulate(
        transfer_matrices[::-1],
        multiply_transfer_matrices_from_right,
        initial=last_right_environment
    ))[::-1]

    current_left_environment = left_environment
    expectations = list()

    for i in range(num_sites):
    # Compute gradient
    grad = expectation_gradient_from_environments(
        psi,
        i,
        left_environment=current_left_environment,
        right_environment=right_environments[i],
        mps_form='B'
    )

    # Optimize unitary, update in place
    U, S, VH = npc.svd(grad.complex_conj(), inner_labels = ['i', 'i*'])

    new_unitary = npc.tensordot(U, VH, [['i',], ['i*',]])
    unitaries[i] = new_unitary
    transfer_matrix = get_transfer_matrix_from_tp_unitary(
        psi, i, new_unitary, form='B'
    )

    # Save new expectation.
    expectations[i] = np.sum(S)

    # Create new left environment
    current_left_environment = multiply_transfer_matrices(
        transfer_matrix,
        current_left_environment
    )

    return (expectations, current_left_environment)


def left_one_site_optimization(right_environment, psi, unitaries):
    """
    Given an MPS psi with environment immediately to the right of it, and a
    set of one site unitaries, optimize each of the unitaries in turn and
    update in place. Output a list of the expectations found at each step.

    Parameters
    ----------
    right_environment: tenpy.linalg.np_conserved.Array
        Evnironment to the right of psi. Tensor with legs 'vL' and 'vL*'.
    psi: tenpy.networks.mps.MPS
        The MPS the unitaries are optimizing against.
    unitaries: List of tenpy.npc.Array
        List of single site operators. Should be same length as the number of
        sites in psi.
    Returns
    -------
    pair of (list of float, tenpy.linalg.np_conserved.Array)
        For the first element in the pair, the i-th element corresponds to
        the expectation after optimizing the unitary at the i-th site.
        For the second, last right environment found with legs 'vL' and 'vL*'.
    """
    psi_inv = psi.copy().spatial_inversion()

    left_environment = right_environmnet.replace_labels(
        ['vL', 'vL*'], ['vR', 'vR*']
    )

    expectations, last_left_environment = right_one_site_optimization(
        left_environment, psi, unitaries
    )

    last_right_environment = last_left_environment.replace_labels(
        ['vR', 'vR*'], ['vL', 'vL*']
    )

    return (expectations, last_left_environment)
