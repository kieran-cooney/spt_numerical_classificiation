"""
Functions for calculating first and second order derivatives of MPS
expectation values with respect to one site operators, and associated loss
functions.

To-do:
* Be very careful with the explanation/interpretation of the real/imaginary
  parts of the gradients.
* Most gradients split into the matrix gradient part (multivariate) and a
  loss function of the expectation part (single complex variable). Any way to
  formalise?
* Initially thought I needed the gradients_outside_right_boundary_unitaries
  and corresponding left function, but found a way around. Testing shows there
  is likely an error with the left function.
* Be a bit more uniform with the language in general. Matrix vs expectation
  etc.
"""

from functools import reduce
from itertools import accumulate, count

import numpy as np
import tenpy.linalg.np_conserved as npc

from .utils import (
    multiply_transfer_matrices,
    multiply_transfer_matrices_from_right,
    get_transfer_matrices_from_unitary_list,
    get_left_identity_environment,
    get_right_identity_environment
)


def anti_hermitian_projector(delta_U, U=None):
    """
    Given a pertrubation delta_U to a unitary U, ensure that the resulting
    matrix delta_U + U is also unitary by enforcing a certain anti-Hermitian
    condition on delta_U. This is done by projecting delta_U onto a linear
    subspace of matrices.

    Parameters
    ----------
    delta_U: Square numpy array
        Peturbation matrix to the unitary U.
    U: Square unitary numpy array
        Unitary matrix. If not specified, the identity matrix is used.

    Returns
    -------
    Numpy array of the same shape as delta_U
        Outut array with two legs
    """
    if U is None:
        return 0.5*(delta_U - delta_U.conj().T)
    else:
        M1 = (U.conj().T).dot(delta_U)
        M2 = 0.5*(M1 - M1.conj().T)
        M3 = U.dot(M2)

    return M3


"""
def zero_trace_projector():
    To-do:
    Make function that sets trace of unitary perturbation to 0.
"""


def expectation_gradient_from_environments(psi, site_index,
    left_environment=None, right_environment=None, mps_form='B'):
    """
    Compute the derivative of the expectation of operators on psi at 
    site_index given the virtual vectors (environments) immediately to the
    left and right of site_index.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The MPS being operated on
    site_index: integer
        The index of the site of psi to calculate the derivative of.
    left_environment: tenpy.linalg.np_conserved.Array with legs vR, vR* 
        Represents the expected value contribution from those sites to the
        left of site_index. If the left environment is not specified, it is
        assumed that there are no operators to the left of the site, which
        allows the left environment to be set automatically (under certain
        gauge assumptions).
    right_environment: tenpy.linalg.np_conserved.Array with legs vL, vL* 
        Represents the expected value contribution from those sites to the
        right of site_index. If the right environment is not specified, it is
        assumed that there are no operators to the right of the site, which
        allows the right environment to be set automatically (under certain
        gauge assumptions).
    mps_form: string
        The "gauge" to use when calculating the matrix gradient. Passed to
        MPS.get_B from the tenpy package. This should be chosen based on the
        gauge used for the left and right environments.

    Returns
    -------
    square tenpy array with legs ['p*', 'p'] 
        Elements of this tenpy array correspond to taking the gradient of the
        expected value with respect to a kronecker delta matrix with those
        same indices.

    To-do:
    * Input checks for canonical form choice and 'None' left/right
    environments
    """
    b = psi.get_B(site_index, form=mps_form)

    if left_environment is None:
        t = npc.tensordot(b, b.conj(), (['vL',], ['vL*',]))
    else:
        t = npc.tensordot(left_environment, b.conj(), (['vR*',], ['vL*',]))
        t = npc.tensordot(t, b, (['vR',], ['vL',]))

    if right_environment is None:
        t = npc.trace(t, 'vR', 'vR*')
    else:
        t = npc.tensordot(t, right_environment, (['vR','vR*'], ['vL', 'vL*']))

    return t


def quadratic_expectation_gradient(matrix_gradient, expectation):
    """
    Given the expectation value and gradient of that expectation value with
    respect to operators on one given site, calculate the gradient of the
    square absolute value of the expectation with respect to the same operators
    on that site

    Parameters
    ----------
    matrix_gradient: square numpy array
        The gradient of the expectation value, for example the output of
        expectation_gradient_from_environments.
    expectation: complex float
        The expected value of the MPS when operator on by some one site
        operators.

    Returns
    -------
    square numpy array
        Elements of this numpy array correspond to taking the gradient of the
        squared absolute value of the expectation with respect to a kronecker
        delta matrix with those same indices.

    """
    return 2*expectation*(matrix_gradient.conj())


def linear_expectation_gradient(matrix_gradient, expectation):
    """
    Given the expectation value and gradient of that expectation value with
    respect to operators on one given site, calculate the gradient of the
    absolute value of the expectation with respect to the same operators
    on that site

    Parameters
    ----------
    matrix_gradient: square numpy array
        The gradient of the expectation value, for example the output of
        expectation_gradient_from_environments.
    expectation: complex float
        The expected value of the MPS when operator on by some one site
        operators.

    Returns
    -------
    square numpy array
        Elements of this numpy array correspond to taking the gradient of the
        absolute value of the expectation with respect to a kronecker delta
        matrix with those same indices.
    """
    expectation_phase = expectation/np.abs(expectation)
    return expectation_phase*(matrix_gradient.conj())


def first_order_correction_from_gradient_one_site(gradient, delta_u):
    """
    Given a one site gradient "gradient" and a perturbation to a one site
    operator "delta_u", compute the perturbation to the target function.

    For the perturbation to the bare expectation, a simple tensor contraction
    will suffice. For real value functions of the expectation, such as the
    magnitude or squared magnitude, this function is required.

    Parameters
    ----------
    gradient: square numpy array
        The gradient of some real function with respect to operators on a
        given site.
    delta_U: square numpy array
        The perturbation to an operator on the same site that "gradient" was
        calcualted.

    Returns
    -------
    real float
        The first order change to the target function as a result of 
        perturbating the on site operator by delta_U.
    """
    return (
        np.sum(np.real(gradient)*np.real(delta_u))
        + np.sum(np.imag(gradient)*np.imag(delta_u))
    )


def first_order_correction_from_gradients(gradients, delta_us):
    """
    Given gradients and perturbations to operators on a string of sites,
    operator "delta_u", compute the perturbation to the target function.

    The two lists should be the same length, ordered so that the i-th element
    of both lists refers to the same site.

    Parameters
    ----------
    gradients: list of square numpy array
        The gradients of some real function with respect to operators on a
        given site.
    delta_u: list of square numpy array
        The perturbations to operators on the same sites that "gradients" was
        calcualted.

    Returns
    -------
    real float
        The first order change to the target function as a result of 
        perturbating the on site operator by delta_u.
    """
    return sum(
        first_order_correction_from_gradient_one_site(g, d)
        for g, d in zip(gradients, delta_Us)
    )

def expectation_gradient_from_transfer_matrices(psi, left_transfer_matrices,
    right_transfer_matrices, site_index, left_environment=None,
    right_environment=None, mps_form='B'):
    """
    Compute the derivative of the expectation of operators on psi at 
    site_index given the transfer matrices immediately to the left and right
    of site_index.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The MPS being operated on
    left_transfer_matrices: list of tenpy.linalg.np_conserved.Array
        List of transfer matrices to the left of site_index represented as
        tenpy arrays with 4 legs. Ordered in the opposite direction to the MPS
        site ordering, so that the first transfer matrix is immediately to the
        left of site_index.
    right_transfer_matrices: list of tenpy.linalg.np_conserved.Array
        List of transfer matrices to the right of site_index represented as
        tenpy arrays with 4 legs. Ordered in the same direction to the MPS
        site ordering, so that the first transfer matrix is immediately to the
        right of site_index.
    site_index: integer
        The index of the site of psi to calculate the derivative of.
    left_environment: tenpy.linalg.np_conserved.Array with legs vR, vR* 
        Represents the expected value contribution from those sites to the
        left of the left_transfer_matrices. If the left environment is not
        specified, it is assumed that there are no operators to the left of the
        site, which allows the left environment to be set automatically (under
        certain gauge assumptions).
    right_environment: tenpy.linalg.np_conserved.Array with legs vL, vL* 
        Represents the expected value contribution from those sites to the
        right of the right_transfer_matrices. If the right environment is not
        specified, it is assumed that there are no operators to the right of the
        site, which allows the right environment to be set automatically (under
        certain gauge assumptions).
    mps_form: string
        The "gauge" to use when calculating the matrix gradient. Passed to
        MPS.get_B from the tenpy package. This should be chosen based on the
        gauge used for the left and right environments.

    Returns
    -------
    square tenpy array with legs ['p*', 'p']
        Elements of this numpy array correspond to taking the gradient of the
        expected value with respect to a kronecker delta matrix with those
        same indices.
    """

    environment_left_of_site = reduce(
        multiply_transfer_matrices,
        left_transfer_matrices
    )

    environment_right_of_site = reduce(
        multiply_transfer_matrices_from_right,
        right_transfer_matrices
    )

    gradient = expectation_gradient_from_environments(
        psi,
        site_index,
        environment_left_of_site,
        environment_right_of_site,
        mps_form
    )

    return gradient


def expectation_gradients(psi, transfer_matrices, left_site_index,
    left_environment=None, right_environment=None, mps_form='B'):
    """
    Compute the derivatives of the expectation of operators on psi at 
    a range of site indices begining at left_site_index using the list
    of matrices transfer_matrices. 

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The MPS being operated on
    transfer_matrices: list of tenpy.linalg.np_conserved.Array
        List of transfer matrices beginning at left_site_index and continuing
        on sequentially. Ordered in the same direction to the MPS site
        ordering.
    left_site_index: integer
        The index of the site of psi to correspdonding to the first element of
        transfer_matrices.
    left_environment: tenpy.linalg.np_conserved.Array with legs vR, vR* 
        Represents the expected value contribution from those sites to the
        left of left_site_index. If the left environment is not
        specified, it is assumed that there are no operators to the left of the
        site, which allows the left environment to be set automatically (under
        certain gauge assumptions).
    right_environment: tenpy.linalg.np_conserved.Array with legs vL, vL* 
        Represents the expected value contribution from those sites to the
        right of the transfer_matrices. If the right environment is not
        specified, it is assumed that there are no operators to the right of the
        site, which allows the right environment to be set automatically (under
        certain gauge assumptions).
    mps_form: string
        The "gauge" to use when calculating the matrix gradient. Passed to
        MPS.get_B from the tenpy package. This should be chosen based on the
        gauge used for the left and right environments.

    Returns
    -------
    list of square tenpy numpy arrays with legs ['p*', 'p']
        List of gradients for each site, one for each element of
        transfer_matrices.

    To-do:
    Fix optional environment logic.
    """

    if left_environment is None:
        left_environment = get_left_identity_environment(
            psi, left_site_index
        )

    left_environments = list(accumulate(
        transfer_matrices[:-1],
        multiply_transfer_matrices,
        initial=left_environment
    ))

    right_site_index = left_site_index + len(transfer_matrices) - 1

    if right_environment is None:
        right_environment = get_right_identity_environment(
            psi, right_site_index
        )

    right_environments = list(accumulate(
        transfer_matrices[:0:-1],
        multiply_transfer_matrices_from_right,
        initial=right_environment
    ))

    iterator = enumerate(
        zip(left_environments, right_environments[::-1]),
        start = left_site_index
    )

    gradients = [
        expectation_gradient_from_environments(psi, i, l_env, r_env, mps_form)
        for i, (l_env, r_env) in iterator
    ]

    return gradients



def gradients_outside_left_boundary_unitaries(psi, right_environment,
    num_outside_sites, left_site_index):
    """
    Compute the derivatives of the expectation of operators on psi at 
    a range of site indices begining immediately to the left of left_site_index
    and moving to the left such that only identity operators act to the left of
    left_site_index.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The MPS being operated on
    right_environment: tenpy.linalg.np_conserved.Array with legs vL, vL* 
        Represents the expected value contribution from those sites to the
        right of the left_site_index (inclusive).
    num_outside_sites: int
        The number of sites to the left of left_site_index (exclusive) to
        compute derivatives for.
    left_site_index: integer
        Gradients will be computed for sites immediately to the left of
        this site index.

    Returns
    -------
    list of square numpy array
        List of gradients of length num_outside_sites with the gradient for
        each site. The site ordering of this list is opposite to that of the
        MPS, i.e. the first element of the list is always the gradient at the
        site with index left_site_index.
    """
    # Transfer matrices ordered from right to left
    transfer_matrices = get_transfer_matrices_from_unitary_list(
        psi,
        left_site_index - 1,
        num_outside_sites-1,
        form='A'
    )

    # List of all right environments for each of the sites
    right_environments = list(accumulate(
        transfer_matrices,
        multiply_transfer_matrices_from_right,
        initial=right_environment
    ))

    site_indices = count(start=left_site_index - 1, step=-1)
        
    gradients = [
        expectation_gradient_from_environments(
            psi, i, right_environment=r_env, mps_form='A'
        )
        for i, r_env in  zip(site_indices, right_environments)
    ]

    return gradients


def gradients_outside_right_boundary_unitaries(psi, left_environment,
    num_outside_sites, right_site_index):
    """
    Compute the derivatives of the expectation of operators on psi at 
    a range of site indices begining immediately to the right of
    right_site_index and moving to the right such that only identity operators
    act to the right of right_site_index.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The MPS being operated on
    left_environment: tenpy.linalg.np_conserved.Array with legs vR, vR* 
        Represents the expected value contribution from those sites to the
        left of the right_site_index (inclusive).
    num_outside_sites: int
        The number of sites to the right of right_site_index (exclusive) to
        compute derivatives for.
    right_site_index: integer
        Gradients will be computed for sites immediately to the right of
        this site index.

    Returns
    -------
    list of square numpy array
        List of gradients of length num_outside_sites with the gradient for
        each site. The site ordering of this list is the same as that of the
        MPS.
    """
    transfer_matrices = get_transfer_matrices_from_unitary_list(
        psi,
        right_site_index + 1,
        num_outside_sites-1,
    )

    left_environments = list(accumulate(
        transfer_matrices,
        multiply_transfer_matrices,
        initial=left_environment
    ))

    gradients = [
        expectation_gradient_from_environments(psi, i, left_environment=l_env)
        for i, l_env in enumerate(left_environments, start=right_site_index+1)
    ]

    return gradients
    
