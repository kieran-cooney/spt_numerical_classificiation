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
"""

import numpy as np
import tenpy.linalg.np_conserved as npc

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



def get_transfer_matrix_from_unitary(psi, unitary, index, form='B'):
    """
    Given an MPS representing a many body wave function psi, contract the
    unitary matrix with psi at the site given by index and contract again
    with the hermitian conjugate at the same location.

    To-do:
        * No reason why we we couldn't replace the hermitian conjugate of psi
          with another wavefunction...?
        * Add check for leg labels...?

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The mps to calcualte the transfer matrix from
    unitary: square numpy array
        The single site operator
    index: integer
        The index of the MPS psi to calculate the transfer matrix for
    Returns
    form: string
        The "gauge" to use when calculating the transfer matrix. Passed to
        MPS.get_B from the tenpy package.
    -------
    tenpy.linalg.np_conserved.Array
        The resulting transfer matrix with legs vR, vR*, vL and vL*.
    """
    # Convert u to suitable tenpy form
    u = to_npc_array(unitary)
    # Get the B array associated to psi at the index
    b = psi.get_B(index, form=form)

    # Contract with psi...
    t = npc.tensordot(b, u, (['p',], ['p*']))
    # ...and psi^dagger
    t = npc.tensordot(t, b.conj(), (['p',], ['p*']))

    return t


def get_transfer_matrices_from_unitary_list(psi, unitaries, starting_index,
                                            form='B'):
    """
    Calculate the transfer matrices of psi and psi conjugate sandwiching the
    list of unitaries. The unitaries are assumed to be adjacent, and ordered
    with the first entry at the leftmost site.

    Parameters
    ----------
    psi: tenpy.networks.mps.MPS
        The mps to calcualte the transfer matrices from
    unitaries: List of square numpy arrays
        List of single site operators
    starting_index: integer
        The index of the MPS psi to calculate the first transfer matrix for.
        The next will be for starting_index + 1 and so on.
    form: string
        The "gauge" to use when calculating the transfer matrices. Passed to
        MPS.get_B from the tenpy package.
    Returns
    -------
    list of tenpy.linalg.np_conserved.Array
        The resulting transfer matrices with legs vR, vR*, vL and vL*.
    """
    transfer_matrices = [
        get_transfer_matrix_from_unitary(psi, u, i, form=form)
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


def get_left_environment(psi, index):
    """
    
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

    # Schmidt values should all be real, so conjugate not really necessary here.
    left_environment = (
        npc.tensordot(SL, SL.conj(), (['vL',], ['vL*',]))
        .combine_legs([['vR', 'vR*'],])
        .to_ndarray()
    )

    return left_environment
