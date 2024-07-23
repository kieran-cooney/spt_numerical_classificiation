"""
To-do:
    * Add "pad" method to add identity unitaries at the boundary.
    * Calculate correlation lengths.
    * Careful with ordering of left boundary unitaries... Natural to keep it
      reversed wrt to the site ordering.
    * There should be a lot more checks for compatibile bond dimensions,
      site dimensions etc.
    * If repeating calculations, it may be worthwhile to add flags checking
      if certain calcualtions (expectations, etc.) have been completed since
      the last parameter update.
    * Add example code. Docs with diagrams of MPS and operators?
    * There is an awkward bouncing back and forth between tenpy and numpy
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

from SPTOptimization.gradients import (
    expectation_gradients,
    quadratic_expectation_gradient,
    anti_hermitian_projector,
    get_expectation_gradient_fixed_basis,
    get_expectation_hessian,
    get_quadratic_expectation_hessian
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
    leftmost_boundary_index: non-negative integer
        The index of the left most site of psi which the boundary operations
        act on. 
    rightmost_boundary_index: non-negative integer
        The index of the right most site of psi whcih the boundary operations
        act on.
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
    right_expectation_gradients: list of tenpy.linalg.np_conserved.Array
        List of gradients with respect to right boundary unitaries, one for each
        unitary. Each element has legs ['p', p*']. The elements of each array
        correspond to taking the derivative of the expectation with respect to a
        kronecker delta with the same ['p', p*'] indices.
    right_projected_expectation_gradients: list of tenpy.linalg.np_conserved.Array
        List of gradients with respect to right boundary unitaries, one for each
        unitary. Each element has legs ['p', p*']. Same as
        right_expectation_gradients but each gradient has been projected onto the
        space of variations which preserve the unitarity of the relevant right
        boundary unitary.
    right_expectation_gradient_fixed_basis: numpy array of shape (n, 3)
        n=len(right_boundary_unitaries). The [i,j] element corresponds to the
        derivative of the expectation when the i-th unitary is varied in the j-th
        direction.
    left_expectation_gradients: list of tenpy.linalg.np_conserved.Array
        List of gradients with respect to left boundary unitaries, one for each
        unitary. Ordered the same as left_boundary_unitaries. Each element has
        legs ['p', p*']. The elements of each array correspond to taking the
        derivative of the expectation with respect to a kronecker delta with the
        same ['p', p*'] indices.
    left_projected_expectation_gradients: list of square numpy arrays
        List of gradients with respect to left boundary unitaries, one for each
        unitary. Each element has legs ['p', p*']. Same as
        left_expectation_gradients but each gradient has been projected onto the
        space of variations which preserve the unitarity of the relevant left
        boundary unitary.
    left_expectation_gradient_fixed_basis: list of square numpy arrays
        n=len(right_boundary_unitaries). The [i,j] element corresponds to the
        derivative of the expectation when the i-th unitary is varied in the j-th
        direction.
    right_quadratic_norm_gradients: list of square numpy arrays
        List of gradients with respect to right boundary unitaries, one for each
        unitary. The elements of each array correspond to taking the derivative of
        the abs(expectation)**2 with respect to a kronecker delta with the same
        indices.
    right_linear_norm_gradients: list of square numpy arrays
        List of gradients with respect to right boundary unitaries, one for each
        unitary. The elements of each array correspond to taking the derivative of
        the abs(expectation) with respect to a kronecker delta with the same
        indices.
    left_quadratic_norm_gradients: list of square numpy arrays
        List of gradients with respect to left boundary unitaries, one for each
        unitary. The elements of each array correspond to taking the derivative of
        the abs(expectation)**2 with respect to a kronecker delta with the same
        indices.
    left_linear_norm_gradients: list of square numpy arrays
        List of gradients with respect to left boundary unitaries, one for each
        unitary. The elements of each array correspond to taking the derivative of
        the abs(expectation) with respect to a kronecker delta with the same
        indices.
    right_expectation_hessian: numpy array of shape (nR, 3, nR, 3)
        The hessian of the total expectation with respect to variations  in the
        right boundary unitaries. The [i,j,k,l] element corresponds to varying the
        i-th unitary in the j-th direction and the k-th unitary in the l-th
        direction.
    left_expectation_hessian: numpy array of shape (nL, 3, nL, 3)
        The hessian of the total expectation with respect to variations  in the
        left boundary unitaries. The [i,j,k,l] element corresponds to varying the
        i-th unitary in the j-th direction and the k-th unitary in the l-th
        direction.
    right_quadratic_norm_hessian: symmetric square numpy array of dimension nR*3
        Symmetry hessian matrix of abs(expectation)**2 with respect to specific
        variations in each unitary of the right boundary unitaries.
    left_quadratic_norm_hessian: symmetric square numpy array of dimension nL*3
        Symmetry hessian matrix of abs(expectation)**2 with respect to specific
        variations in each unitary of the left boundary unitaries.
    right_hessian_eigenvalues: numpy array of shape (nR*3)
        Eigenvalues of self.right_quadratic_norm_hessian.
    left_hessian_eigenvalues: numpy array of shape (nL*3)
        Eigenvalues of self.left_quadratic_norm_hessian.
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
        
        self.right_symmetry_index = (
            self.left_symmetry_index + len(self.symmetry_operations) - 1
        )

        # Initialize the left and right boundary unitaries if not specified.
        if left_boundary_unitaries is None:
            self.left_boundary_unitaries = list()
        else: self.left_boundary_unitaries = left_boundary_unitaries
        self.num_left_unitary_sites = len(self.left_boundary_unitaries)

        if right_boundary_unitaries is None:
            self.right_boundary_unitaries = list()
        else: self.right_boundary_unitaries = right_boundary_unitaries
        self.num_right_unitary_sites = len(self.right_boundary_unitaries)

        self.rightmost_boundary_index = (
            self.right_symmetry_index + self.num_right_unitary_sites
        )

        self.leftmost_boundary_index = (
            self.left_symmetry_index - self.num_left_unitary_sites
        )

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

    def compute_right_expectation_gradients(self):
        """
        Calculate the gradient of the expectation value with respect to
        variations in right_boundary_unitaries. Three versions are calculated,
        right_expectation_gradients, right_projected_expectation_gradients
        and right_expectation_gradient_fixed_basis.

        Returns
        -------
        None
        """
        # Gradients of right (not total) expectation.
        bare_right_expectation_gradients = expectation_gradients(
            self.psi,
            self.right_transfer_matrices,
            self.right_symmetry_index+1,
            left_environment=self.right_projected_symmetry_state,
            right_environment=None
        )

        # The bare gradient is the gradient with respect to the right 
        # expectation. To get the gradient with respect to the total 
        # expectation, must multiply by the missing factors.
        self.right_expectation_gradients = (
            bare_right_expectation_gradients
            * self.symmetry_transfer_matrix_singular_vals[0]
            * self.left_expectation
        )

        self.right_projected_expectation_gradients = list()

        unitary_grad_pairs = zip(
            self.right_boundary_unitaries,
            self.right_expectation_gradients
        )

        for u, g in unitary_grad_pairs:
            self.right_projected_expectation_gradients.append(
                anti_hermitian_projector(g.to_ndarray(), u)
            )

        self.right_expectation_gradient_fixed_basis = (
            get_expectation_gradient_fixed_basis(
                self.right_expectation_gradients,
                self.right_boundary_unitaries
            )
        )

    def compute_left_expectation_gradients(self):
        """
        Calculate the gradient of the expectation value with respect to
        variations in left_boundary_unitaries. Three versions are calculated,
        left_expectation_gradients, left_projected_expectation_gradients
        and left_expectation_gradient_fixed_basis.

        Returns
        -------
        None
        """
        # Gradients of left (not total) expectation.
        reversed_bare_left_expectation_gradients = expectation_gradients(
            self.psi,
            self.left_transfer_matrices[::-1],
            self.leftmost_boundary_index,
            left_environment=None,
            right_environment=self.left_projected_symmetry_state,
            mps_form='A'
        )

        bare_left_expectation_gradients = (
            reversed_bare_left_expectation_gradients[::-1]
        )

        # The bare gradient is the gradient with respect to the left 
        # expectation. To get the gradient with respect to the total 
        # expectation, must multiply by the missing factors.
        self.left_expectation_gradients = (
            bare_left_expectation_gradients[::-1]
            * self.symmetry_transfer_matrix_singular_vals[0]
            * self.right_expectation
        )

        self.left_projected_expectation_gradients = list()

        unitary_grad_pairs = zip(
            self.left_boundary_unitaries,
            self.left_expectation_gradients
        )

        for u, g in unitary_grad_pairs:
            self.left_projected_expectation_gradients.append(
                anti_hermitian_projector(g.to_ndarray(), u)
            )

        self.left_expectation_gradient_fixed_basis = (
            get_expectation_gradient_fixed_basis(
                self.left_expectation_gradients,
                self.left_boundary_unitaries
            )
        )

    def compute_right_norm_gradients(self):
        """
        Calculate the gradient of the abs(e) and abs(e)**2 = e*(conjugate(e))
        with respect to the right boundary unitaries where e is the 
        expectation value.

        Returns
        -------
        None
        """
        self.right_quadratic_norm_gradients = list()
        self.right_linear_norm_gradients = list()

        for g in self.right_projected_expectation_gradients:
            np_g = g.to_ndarray()

            self.right_quadratic_norm_gradients.append(
                quadratic_expectation_gradient(np_g, self.expectation)
            )

            self.right_linear_norm_gradients.append(
                linear_expectation_gradient(np_g, self.expectation)
            )

    def compute_left_quadratic_norm_gradients(self):
        """
        Calculate the gradient of the abs(e) and abs(e)**2 = e*(conjugate(e))
        with respect to the left boundary unitaries where e is the 
        expectation value.

        Returns
        -------
        None
        """
        self.left_quadratic_norm_gradients = list()
        self.left_linear_norm_gradients = list()

        for g in self.left_projected_expectation_gradients:
            np_g = g.to_ndarray()

            self.left_quadratic_norm_gradients.append(
                quadratic_expectation_gradient(np_g, self.expectation)
            )

            self.left_linear_norm_gradients.append(
                linear_expectation_gradient(np_g, self.expectation)
            )

    def compute_left_right_expectation_hessians(self):
        """
        Calculate the hessians of the expectation value with respect to
        specific variations in the left and right boundary unitaries. There
        are two hessians, one for the left and right boundary unitaries.

        Returns
        -------
        None
        """
        # Hessian for right expectation
        self.right_expectation_hessian = get_expectation_hessian(
            self.psi,
            self.right_symmetry_index+1,
            self.right_boundary_unitaries,
            self.right_transfer_matrices,
            left_environment=self.right_projected_symmetry_state,
            right_environment=None,
            mps_form='B'
        )

        # Multiply by missing factor to get hessian for total expectation
        self.right_expectation_hessian *= (
            self.left_expectation
            * self.symmetry_transfer_matrix_singular_vals[0]
        )

        # Hessian for left expectation
        left_expectation_hessian = get_expectation_hessian(
            self.psi,
            self.leftmost_boundary_index,
            self.left_boundary_unitaries[::-1],
            self.left_transfer_matrices[::-1],
            left_environment=None,
            right_environment=self.left_projected_symmetry_state,
            mps_form='A'
        )

        self.left_expectation_hessian = (
            left_expectation_hessian[::-1, :, ::-1, ::]
        )

        # Multiply by missing factor to get hessian for total expectation
        self.left_expectation_hessian *= (
            self.right_expectation
            * self.symmetry_transfer_matrix_singular_vals[0]
        )

    def compute_quadratic_norm_hessians(self):
        """
        Calculate the hessians of abs(expectation)**2 value with respect to
        specific variations in the left and right boundary unitaries. There
        are two hessians, one for the left and right boundary unitaries.

        Returns
        -------
        None
        """
        self.right_quadratic_norm_hessian = get_quadratic_expectation_hessian(
            self.expectation,
            self.right_expectation_gradient_fixed_basis,
            self.right_expectation_hessian
        )

        self.left_quadratic_norm_hessian = get_quadratic_expectation_hessian(
            self.expectation,
            self.left_expectation_gradient_fixed_basis,
            self.left_expectation_hessian
        )

    def compute_hessian_eigenvalues(self):
        """
        Calculate the eigenvalues of self.right_quadratic_norm_hessian and
        self.left_quadratic_norm_hessian.

        Returns
        -------
        None
        """
        self.right_hessian_eigenvalues = np.linalg.eigvalsh(
            self.right_quadratic_norm_hessian
        )

        self.left_hessian_eigenvalues = np.linalg.eigvalsh(
            self.left_quadratic_norm_hessian
        )
