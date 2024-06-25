"""
To-do:
    * Calculate correlation lengths.
    * Careful with ordering of left boundary unitaries... Natural to keep it
      reversed wrt to the site ordering.
    * There should be a lot more checks for compatibile bond dimensions,
      site dimensions etc.
    * If repeating calculations, it may be worthwhile to add flags checking
      if certain calcualtions (expectations, etc.) have been completed since
      the last parameter update.
    * Add example code. Docs with diagrams of MPS and operators?
    * There is an awkward bounding back and forth between tenpy and numpy
      APIs. Should specify somewhere the when it was decided to use each one.
    * Need to explain (or even just right down the formulas) for the
      decomposition of the expectation into a product of left and right terms
      via svd.
    * Add better documentation, overall docstring, include methods, etc.
"""
from functools import reduce
from itertools import accumulate
import numpy as  np

import tenpy.linalg.np_conserved as npc

from SPTOptimization.utils import (
    get_transfer_matrices_from_unitary_list,
    to_npc_array,
    multiply_transfer_matrices,
    multiply_transfer_matrices_from_right,
    tenpy_to_np_transfer_matrix,
    reshape_1D_array_to_square
)

class SymmetryActionWithBoundaryUnitaries:
    """
    Represents
        * A 1-dimensional manybody wavefunction psi
        * A global symmetry action which only acts on finitely many sites of psi
        * Left/right boundary unitaries which act on those sites in psi
          immediately to the left/right of the symmetry operations.

    Attributes
    ----------
    psi: tenpy.networks.mps.MPS
        Many body wavefunction represented by a tenpy MPS
    symmetry_operations: list of 2d numpy arrays
        A list of one site unitary operators to act on contiguous set of sites
        on psi. Intended to be a finite portion of a global symmetry operation
        on psi.
    num_sites: non-negative integer
        The total number of sites in the MPS psi.
    num_symmetry_sites: non-negative integer
        The total number of sites (and operations) that the symmetry
        operation acts on.
    left_symmetry_index: non-negative integer
        The index of the left most site of psi which the symmetry operations
        act on. If not specified, will be automatically chosen so that the
        symmetry operations act as centrally on psi as possible.
    right_symmetry_index: non-negative integer
        The index of the right most site of psi whcih the symmetry operations
        act on.
    left_boundary_unitaries: list of 2d numpy arrays 
        The one site unitaries acting immediately to the left of the symmetry
        operations. Listed in reverse order to the site ordering, so that the
        first element of the list is always immediately to the left of the
        symmetry operations. Defaults to an empty list.
    num_left_unitary_sites: non-negative integer
        The number of unitaries in left_boundary_unitaries.
    right_boundary_unitaries: list of 2d numpy arrays 
        The one site unitaries acting immediately to the right of the symmetry
        operations. Listed in the same order as the site ordering, so that the
        first element of the list is always immediately to the right of the
        symmetry operations. Defaults to an empty list.
    num_right_unitary_sites: non-negative integer
        The number of unitaries in right_boundary_unitaries.
    expectation: complex float
        The expectation of psi when operated on by the symmetry operations and
        the left and right boundary unitaries simultaneously.
    symmetry_transfer_matrices: list of 2d numpy arrays 
        List of transfer matrices, one per one site symmetry operation, in
        the "B"/G-L convention.
    npc_symmetry_transfer_matrix: tenpy.linalg.np_conserved.Array
        Overall transfer matrix for the symmetry operations with four legs in
        the L-G-...-G-L convention.
    np_symmetry_transfer_matrix: 2D numpy.ndarray
        Overall 2D transfer matrix for the symmetry operations with the four
        legs combined into 2 pairs. In L-G-...-G-L convention.
    left_projected_symmetry_state: npc.Array with legs 'vL', 'vL*'
        The left dominant singular vector of the symmetry transfer matrix
    right_projected_symmetry_state: npc.Array with legs 'vR', 'vR*'
        The right dominant singular vector of the symmetry transfer matrix
    symmetry_transfer_matrix_singular_vals: 1D numpy.ndarray
        The singular values of the symmetry transfer matrix
    right_transfer_matrices: list of tenpy.linalg.np_conserved.Array
        The transfer matrices associated to the right boundary unitaries
        in the "B"/G-L convention, ordered with the site index.
    right_transfer_vectors: list of npc.Array with legs 'vR', 'vR*'
        The virtual vectors obtained by multiplying the right projected
        symmetry state by the right boundary transfer matrices iteratively.
        In the "B"/G-L convention.
    right_expectation: complex float
        The expectation of the right boundary unitaries, assuming the
        symmetry transfer matrix is well approximated by the dominant
        singular value.
    left_transfer_matrices: list of tenpy.linalg.np_conserved.Array
        The transfer matrices associated to the left boundary unitaries
        in the "A"/L-G convention, ordered opposite to the site index.
    left_transfer_vectors: list of npc.Array with legs 'vL', 'vL*'
        The virtual vector obtained by multiplying the left environment
        virtual vector by the left boundary transfer matrices iteratively.
        In the "A"/L-G convention.
    left_expectation: complex float
        The expectation of the left boundary unitaries, assuming the
        symmetry transfer matrix is well approximated by the dominant
        singular value.
    svd_approximate_expectation: complex float
        The expectation of psi when operated on by the symmetry operations and
        the left and right boundary unitaries simultaneously, while
        approximating the symmetry transfer matrix by it's dominant eigenvalue.
    """
    def __init__(self, psi, symmetry_operations, left_symmetry_index=None,
                 left_boundary_unitaries=None, right_boundary_unitaries=None):
        """
        Parameters
        ----------
        psi: tenpy.networks.mps.MPS
            Many body wavefunction represented by a tenpy MPS
        symmetry_operations: list of 2d numpy arrays
            A list of one site unitary operators to act on contiguous set of sites
            on psi. Intended to be a finite portion of a global symmetry operation
            on psi.
        left_symmetry_index: non-negative integer
            The index of the left most site of psi whcih the symmetry operations
            act on. If not specified, will be automatically chosen so that the
            symmetry operations act as centrally on psi as possible.
        left_boundary_unitaries: list of 2d numpy arrays 
            The one site unitaries acting immediately to the left of the symmetry
            operations. Listed in reverse order to the site ordering, so that the
            first element of the list is always immediately to the left of the
            symmetry operations. Defaults to an empty list.
        right_boundary_unitaries: list of 2d numpy arrays 
            The one site unitaries acting immediately to the right of the symmetry
            operations. Listed in the same order as the site ordering, so that the
            first element of the list is always immediately to the right of the
            symmetry operations. Defaults to an empty list.
        """

        self.psi = psi
        self.symmetry_operations = symmetry_operations
        self.num_sites = psi.L
        self.num_symmetry_sites = len(symmetry_operations)

        # Set left_symmetry_index if not specified
        if left_symmetry_index is None:
            self.left_symmetry_index = (
                self.psi.L - len(self.symmetry_operations)
            )//2 
        else:
            self.left_symmetry_index = left_symmetry_index
        
        self.right_symmetry_index = self.left_symmetry_index + len(self.symmetry_operations) - 1

        # Initialize the left and right boundary unitaries if not specified.
        if left_boundary_unitaries is None:
            self.left_boundary_unitaries = list()
        else: self.left_boundary_unitaries = left_boundary_unitaries
        self.num_left_unitary_sites = len(self.left_boundary_unitaries)

        if right_boundary_unitaries is None:
            self.right_boundary_unitaries = list()
        else: self.right_boundary_unitaries = right_boundary_unitaries
        self.num_right_unitary_sites = self.right_boundary_unitaries

    def compute_expectation(self):
        """
        Compute the expectation of psi when acted on by the symmetry
        operations and the left and right boundary unitaries.

        Returns
        -------
        complex float
            The resulting expectation
        """
        # Need to reverse the left boundary unitaries due to ordering convention.
        operators = (
            self.left_boundary_unitaries[::-1]
            + self.symmetry_operations
            + self.right_boundary_unitaries
        )

        npc_operators = [to_npc_array(op) for op in operators]

        self.expectation = self.psi.expectation_value_multi_sites(
            npc_operators,
            self.left_symmetry_index - self.num_left_unitary_sites
        )

        return self.expectation

    def compute_svd_symmetry_action(self):
        """
        Compute the singular value decomposition of the transfer matrix
        derived from the symmetry operations, and save relevant quantities.

        The transfer matrix used is not the traditional one; we use a
        convention that is useful for calculations.

        If the MPS is represented as a sequence ...-L-G-L-G-L-... where L are
        schmidt values and G are 3-leg tensors, then the transfer matrix we
        use is derived from L-G-L-...-L-G-L.

        The singular values and dominant left/right singular vectors are
        saved.
        """
        # Obtain list of transfer matrices on each site the symmetry
        # operations act on (G-L or "B" convention).
        self.symmetry_transfer_matrices = (
            get_transfer_matrices_from_unitary_list(
                self.psi,
                self.left_symmetry_index,
                self.symmetry_operations
            )
        )

        # Obtain overall G-L transfer matrix for all sites the symmetry
        # operations act on.
        npc_symmetry_transfer_matrix = reduce(
            multiply_transfer_matrices,
            self.symmetry_transfer_matrices
        )

        # Compute a 2-leg tensor of schmidt values representing 'L1' in
        # L1-G-L-...
        # Using tenpy, so need to be careful with legs.
        left_leg = self.psi.get_B(self.left_symmetry_index).legs[0]
        SL = npc.diag(
            self.psi.get_SL(self.left_symmetry_index),
            left_leg,
            labels = ['vL', 'vR']
        )
        
        # Prepend L1 schmidt values to overall symmetry transfer matrix via
        # tensor contractions.
        npc_symmetry_transfer_matrix = npc.tensordot(
            SL,
            npc_symmetry_transfer_matrix,
            [['vR',],  ['vL',]]
        )

        # SL should be real, so conj shouldn't be necessary...
        npc_symmetry_transfer_matrix = npc.tensordot(
            SL.conj(),
            npc_symmetry_transfer_matrix,
            [['vR*',],  ['vL*',]]
        )

        self.npc_symmetry_transfer_matrix = npc_symmetry_transfer_matrix

        self.np_symmetry_transfer_matrix = (
            self.npc_symmetry_transfer_matrix
            .combine_legs([['vL', 'vL*'], ['vR', 'vR*']])
            .to_ndarray()
        )

        two_dim_tm = (
            self.npc_symmetry_transfer_matrix
            .combine_legs([['vL', 'vL*'], ['vR', 'vR*']])
        )

        U, S, Vh = npc.svd(two_dim_tm, full_matrices=True)

        self.left_projected_symmetry_state = U[:,0].split_legs()
        self.right_projected_symmetry_state = Vh[0].split_legs()
        self.symmetry_transfer_matrix_singular_vals = S

    def compute_right_boundary_expectation(self):
        """
        Compute the contribution of the right boundary unitaries to the
        overall expectation.

        This is only valid if the largest singular value of the symmetry
        operations transfer matrix is significantly larger than all the
        others.

        Returns
        -------
        complex float
            The right expectation.
        """
        # Obtain the transfer matrices for each of the right boundary
        # unitaries
        right_transfer_matrices = get_transfer_matrices_from_unitary_list(
            self.psi,
            self.right_symmetry_index + 1,
            self.right_boundary_unitaries
        )

        self.right_transfer_matrices = right_transfer_matrices

        # Iteratively multiply these against the dominant right singular
        # vector for the symmetry opeations transfer matrix.
        self.right_transfer_vectors = list(accumulate(
            self.right_transfer_matrices,
            multiply_transfer_matrices,
            initial=self.right_projected_symmetry_state
        ))
        
        # Dot against the right environment vector, which in the convention
        # we use is the same as taking the trace.
        self.right_expectation = npc.trace(self.right_transfer_vectors[-1])

        return self.right_expectation

    def compute_left_boundary_expectation(self):
        """
        Compute the contribution of the left boundary unitaries to the
        overall expectation.

        This is only valid if the largest singular value of the symmetry
        operations transfer matrix is significantly larger than all the
        others.

        Returns
        -------
        complex float
            The left expectation.
        """
        # Calculate the site index that the leftmost left boundary unitary
        # acts on.
        left_edge_index = (
            self.left_symmetry_index 
            - self.num_left_unitary_sites
        )

        # Obtain the transfer matrices for each of the left boundary
        # unitaries. Need to use 'A' (=L-G) convention.
        left_transfer_matrices = get_transfer_matrices_from_unitary_list(
            self.psi,
            left_edge_index,
            self.left_boundary_unitaries[::-1],
            form='A'
        )

        self.left_transfer_matrices = left_transfer_matrices[::-1]

        # Iteratively multiply these against the dominant left singular
        # vector for the symmetry opeations transfer matrix.
        self.left_transfer_vectors = list(accumulate(
            self.left_transfer_matrices,
            multiply_transfer_matrices_from_right,
            initial=self.left_projected_symmetry_state
        ))
        
        # Dot against the left environment vector, which in the convention
        # we use is the same as taking the trace.
        self.left_expectation = npc.trace(self.left_transfer_vectors[-1])

        return self.left_expectation

    def compute_svd_approximate_expectation(self):
        """
        Calculate the expectation of the boundary unitaries and symmetry
        operations acting on psi by taking an approximation of the transfer
        matrix of the symmetry operations where only the largest singular
        value of the SVD is kept. This allows us to split the expectation into
        left and right terms, along witht the dominant singular value.

        This is only valid if the largest singular value of the symmetry
        operations transfer matrix is significantly larger than all the
        others.

        Returns
        -------
        complex float
            The approximate expectation.
        """
        self.compute_svd_symmetry_action()
        self.compute_right_boundary_expectation()
        self.compute_left_boundary_expectation()

        self.svd_approximate_expectation = (
            self.symmetry_transfer_matrix_singular_vals[0]
            * self.left_expectation
            * self.right_expectation
        )

        return self.svd_approximate_expectation
