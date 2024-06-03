"""Functions for performing gradient descent to optimize for unitaries.

To-do:
    * Need to re-do a lot of this. Can split into left and right parts.
"""
from functools import reduce
from itertools import accumulate

import numpy as np
from scipy.stats import unitary_group

import tenpy
import tenpy.linalg.np_conserved as npc

from SPTOptimizaton.utils import (
    get_transfer_matrix_from_unitary,
    multiply_transfer_matrices,
    get_transfer_matrices_from_unitary_list
)

UNITARY_TOL = 0.005

unitary_sampler = unitary_group(2)


def get_matrix_gradient(psi, index, environment_left_of_site,
                        environment_right_of_site):
    b = psi.get_B(index)
    t = npc.tensordot(environment_left_of_site, b.conj(), (['vR*',], ['vL*',]))
    t = npc.tensordot(t, b, (['vR',], ['vL',]))
    t = npc.tensordot(t, environment_right_of_site, (['vR','vR*'], ['vL', 'vL*']))

    return t.to_ndarray()


def matrix_element_and_gradients(psi, left_environment, left_transfer_matrices,
    symmetry_block_tm, right_transfer_matrices, starting_index,
    len_symmetry_block):

    ending_index = (
        starting_index +
        len(left_transfer_matrices) +
        len_symmetry_block +
        len(right_transfer_matrices)
        -1
    )
    
    right_environment = npc.eye_like(
        psi.get_B(ending_index),
        axis=-1,
        labels = ['vL', 'vL*'],
    )
    
    all_tms = (
        left_transfer_matrices +
        [symmetry_block_tm,] +
        right_transfer_matrices
    )
    
    tms_products_from_left = list(accumulate(
        all_tms, multiply_transfer_matrices, initial=left_environment
    ))
    
    tms_products_from_right = list(accumulate(
        all_tms[::-1],
        multiply_transfer_matrices_from_right,
        initial=right_environment
    ))

    matrix_element = npc.trace(
        tms_products_from_left[-1], leg1='vR', leg2='vR*'
    )

    left_grads_iter = enumerate(
        tms_products_from_left[:len(left_transfer_matrices)]
    )

    left_matrix_gradients = list()
    for i, left_environment in left_grads_iter:
        right_environment = tms_products_from_right[-i-2]
        psi_index = i + starting_index
        gradient = get_matrix_gradient(
            psi, psi_index, left_environment, right_environment
        )
        left_matrix_gradients.append(gradient)

    right_grads_iter = enumerate(
        tms_products_from_right[:len(right_transfer_matrices)]
    )

    right_matrix_gradients = list()
    for i, right_environment in right_grads_iter:
        left_environment = tms_products_from_left[-i-2]
        psi_index = ending_index - i
        gradient = get_matrix_gradient(
            psi, psi_index, left_environment, right_environment
        )
        right_matrix_gradients.append(gradient)

    return (matrix_element, left_matrix_gradients, right_matrix_gradients[::-1])


def compute_symmetry_block_transfer_matrix(psi, symmetry_op_list, starting_index):
    transfer_matrices = (
        get_transfer_matrix_from_unitary(psi, op, index)
        for index, op in enumerate(symmetry_op_list, start=starting_index)
    )

    symmetry_block_transfer_matrix = reduce(
        multiply_transfer_matrices,
        transfer_matrices
    )

    return symmetry_block_transfer_matrix


def grad_descent_setup(psi, num_unitary_width, symmetry_op_list):
    # Initialize unitaries, using "grid" search.
    # Compute symmetry_block_tm

    num_sites = psi.L
    num_symmetry_sites = len(symmetry_op_list)
    num_unitary_sites = num_unitary_width*2

    starting_index = (num_sites - (num_symmetry_sites + num_unitary_sites))//2
    symmetry_starting_index = starting_index + num_unitary_width

    left_leg = psi.get_B(starting_index).legs[0]
    SL = npc.diag(psi.get_SL(starting_index), left_leg, labels = ['vL', 'vR'])
    left_environment = npc.tensordot(SL, SL.conj(), (['vL',], ['vL*',]))

    symmetry_block_transfer_matrix = compute_symmetry_block_transfer_matrix(
        psi, symmetry_op_list, symmetry_starting_index
    )

    left_unitaries = [unitary_sampler.rvs() for _ in range(num_unitary_width)]
    right_unitaries = [unitary_sampler.rvs() for _ in range(num_unitary_width)]
    
    losses = list()
    unitary_scores = list()

    out = (
        psi,
        left_environment,
        symmetry_block_transfer_matrix,
        left_unitaries,
        right_unitaries,
        starting_index,
        num_symmetry_sites,
        losses,
        unitary_scores
    )

    return out


def projector(U, delta_U):
    M1 = (U.conj().T).dot(delta_U)
    M2 = 0.5*(M1 - M1.conj().T)
    M3 = U.dot(M2)

    return M3


def unitary_test(U):
    return np.max(np.abs(U.conj().T.dot(U)-np.identity(len(U))))


def unitarize_matrix(U):
    L,S,R = np.linalg.svd(U)
    return L.dot(R)


def grad_descent_step(psi, left_environment, symmetry_block_tm, left_unitaries,
                      right_unitaries, starting_index, len_symmetry_block,
                      losses, unitary_scores, eta=0.03):
    
    left_transfer_matrices = get_transfer_matrices_from_unitary_list(
        psi, left_unitaries, starting_index
    )
    right_transfer_matrices = get_transfer_matrices_from_unitary_list(
        psi, right_unitaries, starting_index + len(left_unitaries) + len_symmetry_block
    )
    
    expectation, left_matrix_gradients, right_matrix_gradients = (
        matrix_element_and_gradients(
            psi,
            left_environment,
            left_transfer_matrices,
            symmetry_block_tm,
            right_transfer_matrices,
            starting_index,
            len_symmetry_block
        )
    )

    abs_exp = np.abs(expectation)
    pol_exp = expectation/abs_exp

    """
    if abs_exp < 0.9:
        m = pol_exp
    else:
        m = expectation
    """

    for i, m_grad in enumerate(left_matrix_gradients):
        u = left_unitaries[i]
        u_grad = -2*pol_exp*np.conj(m_grad)
        left_unitaries[i] += -eta*projector(u, u_grad)

    for i, m_grad in enumerate(right_matrix_gradients):
        u = right_unitaries[i]
        u_grad = -2*pol_exp*np.conj(m_grad)
        right_unitaries[i] += -eta*projector(u, u_grad)

    all_unitaries = left_unitaries + right_unitaries

    any_unitary_fail = any(
        unitary_test(u) > UNITARY_TOL
        for u in all_unitaries
    )

    if any_unitary_fail:
        for i in range(len(left_unitaries)):
            u = left_unitaries[i]
            left_unitaries[i] = unitarize_matrix(u)
        for i in range(len(right_unitaries)):
            u = right_unitaries[i]
            right_unitaries[i] = unitarize_matrix(u)

    unitary_score = max(unitary_test(u) for u in left_unitaries + right_unitaries)
    unitary_scores.append(unitary_score)

