"""A collection of utility functions to be used in other functions and
scripts.

To-do:
    * Generalise to_npc_array to other cases
    * Combine the left and right versions of multiply transfer matrices.
      Maybe only need one, as the other can be got by composing with a swap
      arguments function...?
    * Get transfer matrix from unitary works for any operator, not just unitaries...!
    * Currently assuming that all the sites are qubits, should probably put in checks for
      sites with more than 2 degrees of freedom.
    * A lot of functions here, should be broken down into sub-packages.

Dependent code breaking changes:
    * Arguments of get_transfer_matrix_from_unitary updated.
    * Arguments of get_transfer_matrices_from_unitary_list updated.
"""
import re
from functools import reduce

import numpy as np
import tenpy.linalg.np_conserved as npc

from SPTOptimization.tenpy_leg_label_utils import (
    get_physical_leg_labels,
    conjugate_leg_label,
    extract_single_physical_leg_label_from_tensor,
    get_num_legs_block_unitary,
    is_single_physical_leg_label,
    is_grouped_physical_leg_label
)

MAX_VIRTUAL_BOND_DIM = 8
MAX_INTERMEDIATE_VIRTUAL_BOND_DIM = 2*MAX_VIRTUAL_BOND_DIM
SVD_CUTOFF = 1e-3

def unitarize_matrix(X):
    """
    Convert a square numpy array to a unitary matrix.
    The formula used actually produces the closest unitary to X for a specific
    norm:
    https://math.stackexchange.com/questions/3002970/show-the-polar-factor-is-the-closest-unitary-matrix-using-the-spectral-norm

    Parameters
    ----------
    X: Square numpy array
        Input numy array to be converted

    Returns
    -------
    Unitary square numpy array of same shape as X
    """
    U, S, Vh = np.linalg.svd(X)
    return np.dot(U, Vh)


def to_npc_array(np_X):
    """
    Convert a numpy array to an array usable by tenpy.
    Currently implicitly assumes that the array has two vertical legs, ie has
    the form of a 1 site operator on an MPS.

    Parameters
    ----------
    np_X: Square numpy array
        Input numy array to be converted

    Returns
    -------
    tenpy.linalg.np_conserved.Array of complex type.
        Outut array with two legs
    """
    npc_X = (
        npc.Array
        .from_ndarray_trivial(
            np.array(np_X, dtype='complex'),
            dtype=np.complex_,
            labels=['p', 'p*']
        )
    )
    return npc_X


def reshape_1D_array_to_square(a):
    """
    Take a 1d numpy array of len N**2 and reshape into a numpy array of shape
    (N,N).

    Parameters
    ----------
    a: 1D numpy array
        Must be of length N**2 for N a non-negative integer.

    Returns
    -------
    2D numpy array
        array of shape (N,N)
    """
    assert len(a.shape) == 1
    # Notational clash with docstring.
    N = len(a)

    # Find square root of N, and ensure it's square.
    sqrt_N = np.sqrt(N)
    assert int(sqrt_N)**2 == N
    sqrt_N = int(sqrt_N)

    return a.reshape((sqrt_N, sqrt_N))

    
def multiply_transfer_matrices(t1, t2):
    """
    Given two tenpy arrays representing adjacent "sites" vertically contracted
    MPS states (possibly with sandwiched MPOs), contract over the linking legs
    to get an overall transfer matrix for union of inovlved sites.

    Parameters
    ----------
    t1: tenpy.linalg.np_conserved.Array with legs vR, vR* and possibly others
        The left transfser matrix
    t2: tenpy.linalg.np_conserved.Array with legs vL, vL* and possibly others
        The right transfser matrix

    Returns
    -------
    tenpy.linalg.np_conserved.Array
        The result of contracting vR with vL and vR* with vL* between t1 and
        t2.
    """
    return npc.tensordot(t1, t2, (['vR', 'vR*'], ['vL', 'vL*']))


def multiply_transfer_matrices_from_right(t1, t2):
    """
    Given two tenpy arrays representing adjacent "sites" vertically contracted
    MPS states (possibly with sandwiched MPOs), contract over the linking legs
    to get an overall transfer matrix for union of inovlved sites.

    Multiplies in the opposite order to multiply_transfer_matrices which can
    be useful when using reduce.
    Parameters
    ----------
    t1: tenpy.linalg.np_conserved.Array with legs vL, vL* and possibly others
        The right transfser matrix
    t2: tenpy.linalg.np_conserved.Array with legs vR, vR* and possibly others
        The left transfser matrix

    Returns
    -------
    tenpy.linalg.np_conserved.Array
        The result of contracting vL with vR and vL* with vR* between t1 and
        t2.
    """
    return npc.tensordot(t1, t2, (['vL', 'vL*'], ['vR', 'vR*']))


def get_transfer_matrix_from_tp_unitary_and_b_tensor(b, unitary, b_bra=None,
    leg_label='p', leg_label_h=None):
    """
    Given one b tensor representing a site of some mps psi, and a unitary u
    acting at that site, return the transfer matrix resulting from the
    appropriate contractions.
    
    To-do:
        * The logic handling the tensor axis labels is painful.

    Parameters
    ----------
    b: tenpy.linalg.np_conserved.Array
        tensor with legs 'vR', 'vL' and 'p', representing a single site of an
        mps.
    unitary: tenpy.npc.Array
        The single site operator. If None set the operator to the identity
        with appropriate dimension.
    b_bra: None or tenpy.linalg.np_conserved.Array
        tensor with legs 'vR', 'vL' and 'p', representing the bra that the
        mps of b is contracted against after operating with u. If not
        specified, b is used.
    leg_label: str or int
        Label of the leg of b to contract against when constructing transfer
        matrix. If an integer, then take that the leg label with that index.
        Set to 'p' by default.
    leg_label_h: str or int
        Label of the leg of b_bra to contract against when constructing transfer
        matrix. If an integer, then take that the leg label with that index.
        Set to leg_label + '*'  by default.
    Returns
    -------
    tenpy.linalg.np_conserved.Array
        The resulting transfer matrix with legs vR, vR*, vL and vL*.
    """
    b_bottom = b if (b_bra is None) else b_bra
    b_bottom = b_bottom.conj()

    if isinstance(leg_label, int):
        ll = b.get_leg_labels()[leg_label]
    elif isinstance(leg_label, str):
        ll = leg_label

    if isinstance(leg_label_h, int):
        llh = b_bottom.get_leg_labels()[leg_label_h]
    elif (leg_label_h is None):
        if isinstance(leg_label, int):
            llh = b_bottom.get_leg_labels()[leg_label]    
        elif isinstance(leg_label, str):
            llh = leg_label + '*'
    elif isinstance(leg_label_h, str):
        llh = leg_label_h

    # Contract with psi...
    t = npc.tensordot(b, unitary, ([ll,], [llh,]))
    
    # ...and psi^dagger
    t = npc.tensordot(t, b_bottom, ([ll,], [llh,]))

    return t


def get_transfer_matrix_from_tp_unitary(psi, index, unitary, form='B'):
    """
    Given an MPS representing a many body wave function psi, contract the
    unitary matrix with psi at the site given by index and contract again
    with the hermitian conjugate at the same location.

    To-do:
        * It doesn't have to be a unitary, could be any operator!
        * Allow "None" for unitary

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The mps to calcualte the transfer matrix from
    index: integer
        The index of the MPS psi to calculate the transfer matrix for
    unitary: tenpy.npc.Array
        The single site operator. If None set the operator to the identity
        with appropriate dimension.
    form: string
        The "gauge" to use when calculating the transfer matrix. Passed to
        MPS.get_B from the tenpy package.
    Returns
    -------
    tenpy.linalg.np_conserved.Array
        The resulting transfer matrix with legs vR, vR*, vL and vL*.
    """
    # Get the B array associated to psi at the index
    b = psi.get_B(index, form=form)

    return get_transfer_matrix_from_tp_unitary_and_b_tensor(b, unitary)


def get_transfer_matrix_from_unitary(psi, index, unitary=None, form='B'):
    """
    Given an MPS representing a many body wave function psi, contract the
    unitary matrix with psi at the site given by index and contract again
    with the hermitian conjugate at the same location.

    To-do:
        * No reason why we we couldn't replace the hermitian conjugate of psi
          with another wavefunction...?
        * Add check for leg labels...?
        * 'unitary' should be really a keyword argument None, where that would
          result in an identity operator. Unfortunately that forces us to
          change the argument ordering, and hence other code...
        * It doesn't have to be a unitary, could be any operator!

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The mps to calcualte the transfer matrix from
    index: integer
        The index of the MPS psi to calculate the transfer matrix for
    unitary: square numpy array or None
        The single site operator. If None set the operator to the identity
        with appropriate dimension.
    form: string
        The "gauge" to use when calculating the transfer matrix. Passed to
        MPS.get_B from the tenpy package.
    Returns
    -------
    tenpy.linalg.np_conserved.Array
        The resulting transfer matrix with legs vR, vR*, vL and vL*.
    """
    # Convert u to suitable tenpy form
    if unitary is None:
        dim_site = psi.dim[index]
        u = to_npc_array(np.identity(dim_site))
    else: 
        u = to_npc_array(unitary)

    return get_transfer_matrix_from_tp_unitary(psi, index, u, form)


def get_transfer_matrix_from_tp_unitary_list(psi, starting_index, unitaries,
                                             form='B'):
    """
    Calculate the transfer matrix of psi and psi conjugate sandwiching the
    list of unitaries. The unitaries are assumed to be adjacent, and ordered
    with the first entry at the leftmost site.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The mps to calcualte the transfer matrices from
    starting_index: integer
        The index of the MPS psi to calculate the first transfer matrix for.
        The next will be for starting_index + 1 and so on.
    unitaries: List of tenpy.npc.Array
        List of single site operators. 
    form: string
        The "gauge" to use when calculating the transfer matrices. Passed to
        MPS.get_B from the tenpy package.
    Returns
    -------
    list of tenpy.linalg.np_conserved.Array
        The resulting transfer matrices with legs vR, vR*, vL and vL*.
    """
    transfer_matrices = (
        get_transfer_matrix_from_tp_unitary(psi, i, u, form=form)
        for i, u in enumerate(unitaries, start=starting_index)
    )

    out = reduce(multiply_transfer_matrices, transfer_matrices)
    
    return transfer_matrices


def get_transfer_matrices_from_tp_unitary_list(psi, starting_index, unitaries,
                                            form='B'):
    """
    Calculate the transfer matrices of psi and psi conjugate sandwiching the
    list of unitaries. The unitaries are assumed to be adjacent, and ordered
    with the first entry at the leftmost site.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The mps to calcualte the transfer matrices from
    starting_index: integer
        The index of the MPS psi to calculate the first transfer matrix for.
        The next will be for starting_index + 1 and so on.
    unitaries: List of tenpy.npc.Array
        List of single site operators. 
    form: string
        The "gauge" to use when calculating the transfer matrices. Passed to
        MPS.get_B from the tenpy package.
    Returns
    -------
    list of tenpy.linalg.np_conserved.Array
        The resulting transfer matrices with legs vR, vR*, vL and vL*.
    """
    transfer_matrices = [
        get_transfer_matrix_from_tp_unitary(psi, i, u, form=form)
        for i, u in enumerate(unitaries, start=starting_index)
    ]

    return transfer_matrices

def get_transfer_matrices_from_unitary_list(psi, starting_index, unitaries, 
                                            form='B'):
    """
    Calculate the transfer matrices of psi and psi conjugate sandwiching the
    list of unitaries. The unitaries are assumed to be adjacent, and ordered
    with the first entry at the leftmost site.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The mps to calcualte the transfer matrices from
    starting_index: integer
        The index of the MPS psi to calculate the first transfer matrix for.
        The next will be for starting_index + 1 and so on.
    unitaries: List of square numpy arrays or a single integer
        List of single site operators or integer. If any single element is
        "None", that element is taken to be the identity. If an integer,
        interpreted that there are that many indentity operators.
    form: string
        The "gauge" to use when calculating the transfer matrices. Passed to
        MPS.get_B from the tenpy package.
    Returns
    -------
    list of tenpy.linalg.np_conserved.Array
        The resulting transfer matrices with legs vR, vR*, vL and vL*.
    """
    if isinstance(unitaries, int):
        transfer_matrices = [
            get_transfer_matrix_from_unitary(psi, i, form=form)
            for i in range(starting_index, starting_index+unitaries)
        ]
    else:
        transfer_matrices = [
            get_transfer_matrix_from_unitary(psi, i, u, form=form)
            for i, u in enumerate(unitaries, start=starting_index)
        ]

    return transfer_matrices


def tenpy_to_np_transfer_matrix(transfer_matrix):
    """
    Take transfer_matrix of type tenpy.linalg.np_conserved.Array and return
    the correspdoning 2-dimensional numpy array.

    Parameters
    ----------
    transfer_matrix: tenpy.linalg.np_conserved.Array
        The input transfer matrix to be converted. Should have leg labels
        'vL', 'vL*', 'vR' and 'vR*'.
    Returns
    -------
    numpy.ndarray
        The transfer matrix as a 2-dimensional numpy array. The first and second
        indices correspond to the left and right legs of the transfer matrix
        respectively.
    """
    np_transfer_matrix = (
        transfer_matrix
        .combine_legs([['vL', 'vL*'], ['vR', 'vR*']])
        .to_ndarray()
    )

    return np_transfer_matrix


def matrix_element(left_environment, left_transfer_matrices,
                   symmetry_block_tm, right_transfer_matrices):
    """
    Given the transfer matrices and left environment of an MPS being acted on
    by an MPO, calcualte the resulting matrix element. Assumes that the
    transfer matrices were calculated in the "right" canonical form, see
    tenpy paper for more information.

    To-do:
        * Do we really need to separate out the 3 transfer matrices? Could
          just have one list argument...
        * The name for this function is also... just bad.
    Parameters
    ----------
    left_environment: tenpy.linalg.np_conserved.Array
        A tenpy array with legs vR and vR* representing the MPS to the left of
        the transfer matrices.
    left_transfer_matrices: List of tenpy.linalg.np_conserved.Array
        List of transfer matrices to the left of the symmetry operations
    symmetry_block_tm: tenpy.linalg.np_conserved.Array
        Single transfer matrix representing the symmetry action on a finite
        (but potentially large) interval of sites on the MPS
    left_transfer_matrices: List of tenpy.linalg.np_conserved.Array
        List of transfer matrices to the right of the symmetry operations
    Returns
    -------
    complex double/float
        The resulting matrix element by computing the expectation of the left
        and right unitaries and symmetry operations.
    """
    transfer_matrices = (
        left_transfer_matrices +
        [symmetry_block_tm,] +
        right_transfer_matrices
    )

    t_with_open_right_end = reduce(
        multiply_transfer_matrices,
        transfer_matrices,
        left_environment
    )

    return npc.trace(t_with_open_right_end, leg1='vR', leg2='vR*')


def get_left_identity_environment_from_tp_tensor(mps_tensor):
    """
    Given an MPS tensor, construct an environment with just 1's on the diagonal
    which fits in immediately to the left of the tensor.

    Paramters
    ---------
    mps_tensor: tenpy.linalg.np_conserved.Array,
        Tensor with a leg 'vL'.

    Returns
    -------
    tenpy.linalg.np_conserved.Array   
        A diagonal tenpy array with legs 'vR' and 'vR*' with 1's on the
        diagonal.
    """
    left_leg = mps_tensor.get_leg('vL')
    left_environment = npc.diag(1, left_leg, labels = ['vR', 'vR*'])

    return left_environment


def get_left_identity_environment(psi, site_index):
    """
    Given the MPS psi, construct an environment with just 1's on the diagonal
    which fits in immediately to the left of site_index.

    To-do: Could add keyword argument for diagonal values.

    Paramters
    ---------
    psi: tenpy.networks.mps.MPS
        MPS representing a many body wavefunction
    site_index: integer
        The index of the site in psi for which to get the identity environment
        immediately to the left

    Returns
    -------
    tenpy.linalg.np_conserved.Array   
        A diagonal tenpy array with legs 'vR' and 'vR*' with 1's on the
        diagonal.
    """
    t = psi.get_B(site_index)

    return get_left_identity_environment_from_tp_tensor(t)


def get_right_identity_environment_from_tp_tensor(mps_tensor):
    """
    Given an MPS tensor, construct an environment with just 1's on the diagonal
    which fits in immediately to the right of the tensor.

    To-do: Could add keyword argument for diagonal values.

    Paramters
    ---------
    psi: tenpy.networks.mps.MPS
        MPS representing a many body wavefunction
    site_index: integer
        The index of the site in psi for which to get the identity environment
        immediately to the right.

    Returns
    -------
    tenpy.linalg.np_conserved.Array   
        A diagonal tenpy array with legs 'vR' and 'vR*' with 1's on the
        diagonal.
    """
    right_leg = mps_tensor.get_leg('vR')
    right_environment = npc.diag(1, right_leg, labels = ['vL', 'vL*'])

    return right_environment


def get_right_identity_environment(psi, index):
    """
    Given the MPS psi, construct an environment with just 1's on the diagonal
    which fits in immediately to the right of site_index.

    To-do: Could add keyword argument for diagonal values.

    Paramters
    ---------
    psi: tenpy.networks.mps.MPS
        MPS representing a many body wavefunction
    site_index: integer
        The index of the site in psi for which to get the identity environment
        immediately to the right.

    Returns
    -------
    tenpy.linalg.np_conserved.Array   
        A diagonal tenpy array with legs 'vL' and 'vL*' with 1's on the
        diagonal.
    """
    t = psi.get_B(index)

    return get_right_identity_environment_from_tp_tensor(t)


def get_left_environment(psi, index):
    """
    COMMENT

    (Do we even need this function?)

    To-do:
        * Should output tenpy vs numpy array?
        * Naming convention differs from right version.

    Paramters
    ---------
    psi: tenpy.networks.mps.MPS
        MPS representing a many body wavefunction
    index: integer
        The index of the site in psi for which to get the left environment
        immediately to the left

    Returns
    -------
    tenpy.linalg.np_conserved.Array   
        A tenpy array with legs 'vR' and 'vR*' representing the two remaining 
        legs of all the sites to the left of the site in psi at index.
    """
    # Need to be careful with 'legs' when using tenpy.
    left_leg = psi.get_B(index).legs[0]
    SL = npc.diag(psi.get_SL(index), left_leg, labels = ['vL', 'vR'])

    # Schmidt values should all be real, so conjugate not really necessary here
    left_environment = (
        npc.tensordot(SL, SL.conj(), (['vL',], ['vL*',]))
        .combine_legs([['vR', 'vR*'],])
        .to_ndarray()
    )

    return left_environment

def get_identity_operator(psi, site_index):
    """
    Return a square identity numpy array with dimension that of the physical
    dimension of psi at site_index.

    Paramters
    ---------
    psi: tenpy.networks.mps.MPS
        MPS representing a many body wavefunction
    site_index: integer
        The index of the site in psi for which to get the identity operator.
        This is used just to find the correct dimension.

    Returns
    -------
    square numpy.ndarray
        A square identity numpy array with dimension that of the physical
        dimension of psi at site_index.
    """
    dim = psi.dim[site_index]
    X = np.identity(dim, dtype='complex')

    return X


def get_npc_identity_operator(mps_tensor):
    p_leg_label = get_physical_leg_labels(mps_tensor)[0]
    p_leg = mps_tensor.get_leg(p_leg_label)
    p_leg_label_conj = conjugate_leg_label(p_leg_label)

    out = npc.diag(
        1,
        leg=p_leg,
        dtype='complex',
        labels=[p_leg_label, p_leg_label_conj]
    )

    return out


def get_npc_random_operator(mps_tensor):
    p_leg_label = get_physical_leg_labels(mps_tensor)[0]
    p_leg = mps_tensor.get_leg(p_leg_label)
    p_leg_label_conj = conjugate_leg_label(p_leg_label)

    dim = p_leg.ind_len

    out = npc.Array.from_func(
        np.random.standard_normal,
        legcharges=[p_leg, p_leg.conj()],
        dtype='complex',
        labels=[p_leg_label, p_leg_label_conj]
    )

    return out


###############################################################################
# Functions for spllitng/grouping tenpy tensors.

def group_elements(l, group_size, offset=0):
    """
    Given a list l, integers group_size and offset, return
    [
        [l[0], l[1], ..., l[offset - 1]],
        [l[offset], ..., l[offset - 1 + group_size]],
        ...
    ]
    """
    first, rest = l[:offset], l[offset:]

    num_rest_groups = ((len(rest)-1)//group_size) + 1

    groups = [first,] if first else list()

    for i in range(num_rest_groups):
        first_index = i*group_size
        last_index = (i+1)*group_size
        groups.append(rest[first_index:last_index])

    return groups


def combine_tensors(tensors):
    """
    Combine a group of tenpy Arrays with virtual legs 'vR' and 'vL' so that
    virtual legs are contracted and the physical legs are grouped.
    """
    contract_virtual_legs = lambda tl, tr: npc.tensordot(tl, tr, ['vR', 'vL'])

    out = reduce(contract_virtual_legs, tensors)

    leg_labels = [
        extract_single_physical_leg_label_from_tensor(t)
        for t in tensors
    ]

    out = out.combine_legs(leg_labels)

    return out


def combine_b_tensors(b_tensors):
    renamed_tensors = [
        b.replace_label('p', f'p{i}')
        for i, b in enumerate(b_tensors)
    ]

    return combine_tensors(renamed_tensors)


def svd_reduce_split_tensor(t, max_inner_dim=MAX_VIRTUAL_BOND_DIM,
                           normalise=True, svd_cutoff=SVD_CUTOFF):
    """
    Apply the svd decomposition to a tensor in the virtual direction, and
    truncate according to svd_cutoff and max_inner_dim.
    """
    U, S, VH = npc.svd(
        t,
        compute_uv=True,
        inner_labels=['vR', 'vL'],
        cutoff=svd_cutoff
    )

    # Truncate tensors:
    U = U[:, :max_inner_dim]
    S = S[:max_inner_dim]
    VH = VH[:max_inner_dim, :]

    if normalise:
        new_norm = np.sqrt(np.sum(S**2))
        S = S/new_norm

    """
    leg = VH.get_leg('vL')

    schmidt_values = npc.diag(S, leg, labels=['vL', 'vR'])
    """

    return U, S, VH


def split_combined_b(b, leftmost_schmidt_values,
                     max_virtual_bond_dim=MAX_INTERMEDIATE_VIRTUAL_BOND_DIM,
                     p_leg_labels=None):
    """
    Given an MPS b tensor with virtual legs 'vL' and 'vR' and a single
    physical leg, split b into multiple individual tensors contracted along
    virtual legs.

    To-do: Better doc/reference for what's going on here.
    """
    t = b.split_legs()

    num_sites = t.ndim - 2

    if p_leg_labels is None:
        p_leg_labels = [f'p{i}' for i in range(num_sites)]

    out_bs = list()
    out_schmidt_values = list()

    current_left_schmidt_values = leftmost_schmidt_values

    for i, ll in enumerate(p_leg_labels[:-1]):
        # In case the bond dimension has been truncated. May need to add in a
        # case if have less schmidt values than the bond dim...
        bond_dim = t.get_leg('vL').get_block_sizes()[0]
        t.iscale_axis(current_left_schmidt_values[:bond_dim], axis='vL')

        tail_legs = p_leg_labels[(i+1):]
        
        t = t.combine_legs([['vL', ll], ['vR', *tail_legs]])

        U, S, VH = svd_reduce_split_tensor(
            t,
            max_inner_dim=max_virtual_bond_dim,
            normalise=True
        )

        bl = (
            U
            .split_legs()
            .replace_label(ll, 'p')
        )
        bl.iscale_axis(1/current_left_schmidt_values[:bond_dim], axis='vL')
        bl.iscale_axis(S, axis='vR')
        bl.itranspose(['vL', 'p', 'vR'])
        out_bs.append(bl)

        out_schmidt_values.append(S)
        current_left_schmidt_values=S

        t = VH.split_legs()

    bl = t.replace_label(p_leg_labels[-1], 'p')
    bl.itranspose(['vL', 'p', 'vR'])
    out_bs.append(bl)

    return out_bs, out_schmidt_values


def split_b(b, max_virtual_bond_dim=MAX_INTERMEDIATE_VIRTUAL_BOND_DIM,
                     p_leg_labels=None):
    """
    Splits the MPS b tensor according to split_combined_b if the physical leg
    is grouped, otherwise does nothing.
    """
    leg_label = get_physical_leg_labels(b)[0]

    if is_single_physical_leg_label(leg_label):
        return b
    elif is_grouped_physical_leg_label(leg_label):
        return split_combined_b(b, max_virtual_bond_dim, p_leg_labels)
    else:
        raise ValueError


def combine_grouped_b_tensors(grouped_bs):
    """
    Takes a list of lists of b tensors, and returns a list of grouped b tensors.

    To-do: Probably doesn't need a function in utils? Usage seems specific,
    local...
    """
    out = list()

    for group in grouped_bs:
        if len(group) == 1:
            out.append(group[0])
        else:
            out.append(combine_b_tensors(group))

    return out

###############################################################################

def inner_product_b_tensors(b_tensors, b_bra_tensors=None, left_environment=None,
                            right_environment=None):
    """
    Contract b_tensors against b_bra_tensors, including the left and right
    environments if provided.
    """
    if b_bra_tensors is None:
        b_bra_tensors = b_tensors

    b = b_tensors[0]
    b_bra = b_bra_tensors[0]

    if left_environment is None:
        t = npc.tensordot(b, b_bra.conj(), [['vL',], ['vL*',]])
    else:
        t = npc.tensordot(left_environment, b, [['vR',], ['vL',]])
        t = npc.tensordot(t, b_bra.conj(), [['vR*', 'p'], ['vL*', 'p*']])

    for b, b_bra in zip(b_tensors[1:], b_bra_tensors[1:]):
        t = npc.tensordot(t, b, [['vR',], ['vL',]])
        t = npc.tensordot(t, b_bra.conj(), [['vR*', 'p'], ['vL*', 'p*']])

    if right_environment is None:
        out = npc.trace(t)
    else:
        out = npc.tensordot(t, right_environment, [['vR', 'vR*'], ['vL', 'vL*']])

    return out


def get_left_side_right_symmetry_environment(
    right_top_b_tensors, right_bottom_b_tensors, symmetry_transfer_matrix,
    left_side_environment=False, on_site_operators=None
    ):
    """
    Given symmetry_transfer_matrix and two sets of MPS tensors immediately to
    the right, contract the tensors to evalute the resulting symmetry
    environment from symmetry_transfer_matrix on it's left side. 

    If on_site_operators is present, sandwich between top and bottom b tensors.

    (On the left side of said transfer matrix, the environment will be on the
    right of any relevant tensors, hence the name.)
    """
    if right_bottom_b_tensors is None:
        right_bottom_b_tensors = right_top_b_tensors

    ops = on_site_operators

    t = get_right_identity_environment_from_tp_tensor(right_top_b_tensors[-1])

    if ops is None:
        for tb, bb in zip(right_top_b_tensors[::-1], right_bottom_b_tensors[::-1]):
            t = npc.tensordot(t, tb, [['vL',], ['vR']])
            leg_label = get_physical_leg_labels(t)[0]
            leg_label_conj = conjugate_leg_label(leg_label)
            t = npc.tensordot(
                t,
                bb.conj(),
                [['vL*', leg_label], ['vR*', leg_label_conj]]
            )

    else:
        triples = zip(
            right_top_b_tensors[::-1],
            ops[::-1],
            right_bottom_b_tensors[::-1]
        )
        for tb, u, bb in triples:
            t = npc.tensordot(t, tb, [['vL',], ['vR']])
            leg_label = get_physical_leg_labels(t)[0]
            leg_label_conj = conjugate_leg_label(leg_label)
            t = npc.tensordot(t, u, [[leg_label], [leg_label_conj]])
            t = npc.tensordot(
                t,
                bb.conj(),
                [['vL*', leg_label], ['vR*', leg_label_conj]]
            )

    out = npc.tensordot(
        t,
        symmetry_transfer_matrix,
        [['vL', 'vL*',], ['vR', 'vR*']]
    )

    if left_side_environment:
        return (out, t)
    else:
        return out

def group_by_lengths(l, lengths):
    """
    Given a list l and a list of numbers lengths, return a list out where
    out[0] is the first lengths[0] elements of l, out[1] is the next
    lengths[1] elements and so on.

    To-do: Could combine with group elements...?
    """
    out = list()

    current_index = 0

    for n in lengths:
        current_group = l[current_index:current_index+n]
        out.append(current_group)

        current_index += n

    return out


def multiply_blocked_unitaries_against_mps(unitaries, b_tensors,
    left_schmidt_values, max_virtual_bond_dim=MAX_VIRTUAL_BOND_DIM):
    """
    Given a list of blocked (on sites) unitaries, apply against MPS b tensors
    and return a new set of b tensors.

    Would this work on a list of unblocked unitaries...? Probably not.
    """
    site_group_lens = [get_num_legs_block_unitary(u) for u in unitaries]

    grouped_bs = group_by_lengths(b_tensors, site_group_lens)
    grouped_schmidt_values = group_by_lengths(
        left_schmidt_values,
        site_group_lens
    )

    combined_bs = combine_grouped_b_tensors(grouped_bs)

    for i, u in enumerate(unitaries):
        b = combined_bs[i]
        ll = get_physical_leg_labels(b)[0]
        llh = conjugate_leg_label(ll)
    
        new_b = npc.tensordot(b, u, [[ll,], [llh,]])
    
        combined_bs[i] = new_b

    new_top_bs = list()
    # Is the copy necessary...?
    new_left_schmidt_values = left_schmidt_values.copy()

    for b, s in zip (combined_bs, grouped_schmidt_values):
        leg_label = get_physical_leg_labels(b)[0]
        if is_single_physical_leg_label(leg_label):
            new_top_bs.append(b)
            new_left_schmidt_values.extend(s)
        elif is_grouped_physical_leg_label(leg_label):
            bs, schmidt_vals = split_combined_b(
                b,
                s[0],
                max_virtual_bond_dim
            )
            new_top_bs.extend(bs)
            new_left_schmidt_values.extend(s)

    return new_top_bs, new_left_schmidt_values


def multiply_stacked_unitaries_against_mps(unitaries, b_tensors,
    left_schmidt_values, max_virtual_bond_dim=MAX_VIRTUAL_BOND_DIM):
    """
    The input unitaries should be indexed as
    list[layer_index][brick_index], which would then return an appropriate
    block unitary.

    This function then applies the unitary represented by unitaries against
    b_tensors, returning a new set of b_tensors.

    Note: unitaries seems like a confusing name here, considering it's just one
    unitary...
    """
    out_b_tensors = b_tensors.copy()
    out_left_schmidt_values = left_schmidt_values.copy()

    for l in unitaries:
        out = multiply_blocked_unitaries_against_mps(
            l,
            out_b_tensors,
            out_left_schmidt_values,
            max_virtual_bond_dim
        )

        out_b_tensors, out_left_schmidt_values = out

    return out_b_tensors, out_left_schmidt_values

###############################################################################

def get_physical_dim(mps_tensor):
    leg_label = get_physical_leg_labels(mps_tensor)[0]
    out = mps_tensor.get_leg(leg_label).ind_len
    return out
