"""
To-do:
    * Track unitaries as optimising.
    * Add "current expectation" function.
    * Add a "to SymmetryActionaWithBoundaryUnitaries" method.
    * Currently tolerance is checking left_expectation and right_expectation,
      which could be quite different. Should really check wrt the overall
      expectation.
    * The left_right_environments and right_left_environments both have a
      fixed first entry, should remove.
    * Is it strictly necessary to compute the right_right_environments and
      left_left_environments at initialisation?
    * Docstring at top of file.
    * Provide optimisation formulae/derivation somewhere.
"""
from itertools import count, accumulate

import numpy as np
from scipy.stats import unitary_group

import tenpy.linalg.np_conserved as npc

from SPTOptimization.utils import (
    unitarize_matrix,
    get_transfer_matrix_from_unitary,
    multiply_transfer_matrices,
    multiply_transfer_matrices_from_right,
    get_right_identity_environment,
    get_left_identity_environment,
    get_identity_operator
)

from SPTOptimization.SymmetryActionWithBoundaryUnitaries import (
    SymmetryActionWithBoundaryUnitaries
)

from SPTOptimization.gradients import expectation_gradient_from_environments
DEFAULT_TOLERANCE = 1e-3

class OneSiteSolver:
    """
    Optimises unitaries of SymmetryActionWithBoundaryUnitaries instance
    under the assumption that the boundary unitaries are all one site.

    Attributes
    ----------
    psi: tenpy.networks.mps.MPS
        Many body wavefunction represented by a tenpy MPS
    left_symmetry_index: non-negative integer
        The index of the left most site of psi which the symmetry operations
        act on. 
    right_symmetry_index: non-negative integer
        The index of the right most site of psi whcih the symmetry operations
        act on.
    num_right_unitary_sites: non-negative integer
        The number of unitaries in right_boundary_unitaries.
    num_left_unitary_sites: non-negative integer
        The number of unitaries in left_boundary_unitaries.
    left_boundary_unitaries: list of 2d numpy arrays 
        The one site unitaries acting immediately to the left of the symmetry
        operations. Listed in reverse order to the site ordering, so that the
        first element of the list is always immediately to the left of the
        symmetry operations. 
    right_boundary_unitaries: list of 2d numpy arrays 
        The one site unitaries acting immediately to the right of the symmetry
        operations. Listed in the same order as the site ordering, so that the
        first element of the list is always immediately to the right of the
        symmetry operations. 
    leftmost_boundary_index: non-negative integer
        The index of the left most site of psi which the boundary operations
        act on. 
    rightmost_boundary_index: non-negative integer
        The index of the right most site of psi whcih the boundary operations
        act on.
    left_projected_symmetry_state: npc.Array with legs 'vL', 'vL*'
        The left dominant singular vector of the symmetry transfer matrix
    right_projected_symmetry_state: npc.Array with legs 'vR', 'vR*'
        The right dominant singular vector of the symmetry transfer matrix
    symmetry_singular_value: float
        The largest singular value of the SVD of the symmetry transfer matrix.
    right_transfer_matrices: list of tenpy.linalg.np_conserved.Array
        The transfer matrices associated to the right boundary unitaries
        in the "B"/G-L convention, ordered with the site index.
    left_transfer_matrices: list of tenpy.linalg.np_conserved.Array
        The transfer matrices associated to the left boundary unitaries
        in the "A"/L-G convention, ordered opposite to the site index.
    tolerance: float in (0,1)
        If the absolute value of the epxectation is not increasing by this
        amount over an iteration, the  loop will stop.
    left_right_environments: list of npc.Array with legs 'vR', 'vR*'
        The i-th entry is a two leg tensor representing the contribution to
        the expectation from all sites to the left of the site corresponding
        to the i-th right boundary unitary.
        In the "B"/G-L convention.
    right_right_environments: list of npc.Array with legs 'vL', 'vL*'
        The i-th entry is a two leg tensor representing the contribution to
        the expectation from all sites to the right of the site corresponding
        to the (i-1)-th right boundary unitary.
        In the "B"/G-L convention.
    right_left_environments: list of npc.Array with legs 'vL', 'vL*'
        The i-th entry is a two leg tensor representing the contribution to
        the expectation from all sites to the right of the site corresponding
        to the i-th left boundary unitary.
        In the "A"/L-G convention.
    left_left_environments: list of npc.Array with legs 'vR', 'vR*'
        The i-th entry is a two leg tensor representing the contribution to
        the expectation from all sites to the left of the site corresponding
        to the (i-1)-th left boundary unitary.
        In the "A"/L-G convention.
    left_expectation: complex float
        The expectation of the left boundary unitaries, assuming the
        symmetry transfer matrix is well approximated by the dominant
        singular value.
    right_expectation: complex float
        The expectation of the right boundary unitaries, assuming the
        symmetry transfer matrix is well approximated by the dominant
        singular value.
    right_abs_expectations: list of 1d numpy array of float
        The j-th entry of the i-th array is the right absolute value of the
        right expectation after updating the j-th right boundary unitary in
        the i-th sweep over the right boundary unitaries.
    left_abs_expectations: list of 1d numpy array of float
        The j-th entry of the i-th array is the left absolute value of the
        left expectation after updating the j-th left boundary unitary in
        the i-th sweep over the left boundary unitaries.
    auto_pad: boolean
        Boolean indicating wether additional boundary untiaries should be
        added or not during optimisation.
    """
    def __init__(self, initial_conditions, num_right_boundary_unitaries=None,
                 num_left_boundary_unitaries=None, random_unitaries=False,
                 tolerance=DEFAULT_TOLERANCE, auto_pad=True):
        """
        Parameters
        ----------
        initial_conditions: SPTOptimization.SymmetryActionaWithBoundaryUnitaries.SymmetryActionaWithBoundaryUnitaries
            Class detailing the initial conditions of the problem, including
            the MPS, the symmetry operations, initial boundary unitaries and
            more.
        num_right_boundary_unitaries: integer
            The number of right boundary unitaries to optimise over. Takes as
            many from initial_conditions as possible. If more needed, identity
            matrices are used.
        num_left_boundary_unitaries: integer
            The number of left boundary unitaries to optimise over. Takes as
            many from initial_conditions as possible. If more needed, identity
            matrices are used.
        random_unitaries: boolean
            If True, randomly initialise all boundary unitaries, ignorning
            values from initial_conditions.
        tolerance: float
            A float between zero and 1 specifying when the iterative 
            optimization should stop. Set to default value if not specified.
        auto_pad: boolean
            Boolean indicating wether additional boundary untiaries should be
            added or not during optimisation. True by default.
        """
        ic = initial_conditions

        # Read in paramaters from initial_conditions.
        # Probably a better way to do this. Maybe subclassing?
        self.psi = ic.psi
        self.left_symmetry_index = ic.left_symmetry_index
        self.right_symmetry_index = ic.right_symmetry_index

        if num_right_boundary_unitaries is None:
            self.num_right_boundary_unitaries = ic.num_right_unitary_sites
        else:
            self.num_right_boundary_unitaries = num_right_boundary_unitaries

        if num_left_boundary_unitaries is None:
            self.num_left_boundary_unitaries = ic.num_left_unitary_sites
        else:
            self.num_left_boundary_unitaries = num_left_boundary_unitaries

        self.rightmost_boundary_index = (
            self.right_symmetry_index
            + self.num_right_boundary_unitaries
        )
        self.leftmost_boundary_index = (
            self.left_symmetry_index
            - self.num_left_boundary_unitaries
        )

        # Physical dimensions of the boundary sites
        right_boundary_dims = self.psi.dim[
            self.right_symmetry_index + 1:
            self.rightmost_boundary_index + 1
        ]

        left_boundary_dims = self.psi.dim[
            self.left_symmetry_index - 1:
            self.leftmost_boundary_index - 1:
            -1
        ]

        # If random_unitaries is true, randomly initialise each boundary
        # unitary.
        if random_unitaries:
            self.right_boundary_unitaries = [
                unitary_group.rvs(d) for d in right_boundary_dims
            ]

            self.left_boundary_unitaries = [
                unitary_group.rvs(d) for d in left_boundary_dims
            ]

        else:
            # Number of right boundary unitaries to take from ic
            num_rbu_from_ic = min(
                ic.num_right_unitary_sites,
                self.num_right_boundary_unitaries
            )
            self.right_boundary_unitaries = (
                ic.right_boundary_unitaries.copy()[:num_rbu_from_ic]
            )

            # Fill leftovers with identities
            for d in right_boundary_dims[num_rbu_from_ic:]:
                self.right_boundary_unitaries.append(np.identity(d))

            # Repeat for left hand side.
            num_lbu_from_ic = min(
                ic.num_left_unitary_sites,
                self.num_left_boundary_unitaries
            )

            self.left_boundary_unitaries = (
                ic.left_boundary_unitaries.copy()[:num_lbu_from_ic]
            )

            for d in left_boundary_dims[num_lbu_from_ic:]:
                self.left_boundary_unitaries.append(np.identity(d))

        self.left_projected_symmetry_state = ic.left_projected_symmetry_state
        self.right_projected_symmetry_state = ic.right_projected_symmetry_state
        self.symmetry_singular_value = (
            ic.symmetry_transfer_matrix_singular_vals[0]
        )

        self.right_transfer_matrices = [
            get_transfer_matrix_from_unitary(self.psi, i, u, form='B')
            for i, u in enumerate(
                self.right_boundary_unitaries,
                start=self.right_symmetry_index + 1
            )
        ]

        self.left_transfer_matrices = [
            get_transfer_matrix_from_unitary(self.psi, i, u, form='A')
            for i, u in zip(
                count(self.left_symmetry_index -1, -1),
                self.left_boundary_unitaries
            )
        ]

        self.tolerance = tolerance

        # Initialize the 4 environment variables
        self.left_right_environments = list(accumulate(
            self.right_transfer_matrices,
            multiply_transfer_matrices,
            initial=self.right_projected_symmetry_state
        ))

        rightmost_right_environment = get_right_identity_environment(
            self.psi,
            self.rightmost_boundary_index
        )

        self.right_right_environments = list(accumulate(
            self.right_transfer_matrices[::-1],
            multiply_transfer_matrices_from_right,
            initial=rightmost_right_environment
        ))[::-1]

        self.right_left_environments = list(accumulate(
            self.left_transfer_matrices,
            multiply_transfer_matrices_from_right,
            initial=self.left_projected_symmetry_state
        ))

        leftmost_left_environment = get_left_identity_environment(
            self.psi,
            self.leftmost_boundary_index
        )

        self.left_left_environments = list(accumulate(
            self.left_transfer_matrices[::-1],
            multiply_transfer_matrices,
            initial=leftmost_left_environment
        ))[::-1]

        # Initialize the abs expectataions for future use.
        self.right_abs_expectations = list()
        self.left_abs_expectations = list()

        self.auto_pad = auto_pad

    def pad_right(self, num_sites=1):
        """
        Increase the number of right_boundary_unitaries by num_sites by adding
        on identity operators to the right.

        Parameters
        ----------
        num_sites: float
            The number of additional sites/operators to add to
            right_boundary_unitaries.
        """
        new_rightmost_boundary_index = self.rightmost_boundary_index + num_sites

        new_site_indices = list(range(
            self.rightmost_boundary_index + 1,
            new_rightmost_boundary_index + 1
        ))

        new_boundary_unitaries = [
            get_identity_operator(self.psi, site_index)
            for site_index in new_site_indices
        ]

        new_transfer_matrices = [
            get_transfer_matrix_from_unitary(self.psi, i, form='B')
            for i in new_site_indices
        ]

        new_left_right_environments = list(accumulate(
            new_transfer_matrices,
            multiply_transfer_matrices,
            initial=self.left_right_environments[-1]
        ))

        # As we are in the "B" gauge, we know that the right environments for
        # identity operators are just identity matrices.
        new_right_right_environments = [
            get_right_identity_environment(self.psi, i)
            for i in new_site_indices
        ]

        # Update relevant attributes.
        self.right_boundary_unitaries.extend(new_boundary_unitaries)
        self.rightmost_boundary_index = new_rightmost_boundary_index
        self.right_transfer_matrices.extend(new_transfer_matrices)
        self.left_right_environments.extend(new_left_right_environments)
        self.right_right_environments.extend(new_right_right_environments)

    def pad_left(self, num_sites=1):
        """
        Increase the number of left_boundary_unitaries by num_sites by adding
        on identity operators to the left.

        Parameters
        ----------
        num_sites: float
            The number of additional sites/operators to add to
            left_boundary_unitaries.
        """
        new_leftmost_boundary_index = self.leftmost_boundary_index - num_sites

        new_site_indices = list(range(
            self.leftmost_boundary_index - 1,
            new_leftmost_boundary_index - 1,
            -1
        ))

        new_boundary_unitaries = [
            get_identity_operator(self.psi, site_index)
            for site_index in new_site_indices
        ]

        new_transfer_matrices = [
            get_transfer_matrix_from_unitary(self.psi, i, form='A')
            for i in new_site_indices
        ]

        new_right_left_environments = list(accumulate(
            new_transfer_matrices,
            multiply_transfer_matrices_from_right,
            initial=self.right_left_environments[-1]
        ))

        # As we are in the "A" gauge, we know that the left environments for
        # identity operators are just identity matrices.
        new_left_left_environments = [
            get_left_identity_environment(self.psi, i)
            for i in new_site_indices
        ]

        # Update relevant attributes.
        self.left_boundary_unitaries.extend(new_boundary_unitaries)
        self.leftmost_boundary_index = new_leftmost_boundary_index
        self.left_transfer_matrices.extend(new_transfer_matrices)
        self.right_left_environments.extend(new_right_left_environments)
        self.left_left_environments.extend(new_left_left_environments)

    def solve_one_right_site(self, site_index, boundary_unitary_index):
        """
        Optimise the absolute expectation value over the right boundary
        unitary located at site_index.

        To-do:
            * 1 index would do, rather than 2?

        Parameters
        ----------
        site_index: integer
            The site index of self.psi of the right boundary unitary to be
            optimised.
        boundary_unitary_index: integer
            The index of the right boundary unitary in
            right_boundary_unitaries.

        Returns
        -------
        complex float
            Right expectation after optimising unitary.
        """
        # Compute the gradient of the expectation with respect to the
        # operator to be optimised.
        le = self.left_right_environments[boundary_unitary_index]
        re = self.right_right_environments[boundary_unitary_index+1]
        tp_grad = expectation_gradient_from_environments(
            self.psi,
            site_index,
            left_environment=le,
            right_environment=re,
            mps_form='B'
        )

        grad = tp_grad.to_ndarray()

        # Simple formula to find the optimised unitary, and then update.
        new_unitary = unitarize_matrix(grad.conj())
        self.right_boundary_unitaries[boundary_unitary_index] = new_unitary

        # Update relevant variables and output new expectation.
        new_tm = get_transfer_matrix_from_unitary(
            self.psi, 
            site_index,
            unitary=new_unitary,
            form='B'
        )

        self.right_transfer_matrices[boundary_unitary_index] = new_tm
        new_le = multiply_transfer_matrices(le, new_tm)
        self.left_right_environments[boundary_unitary_index+1]=new_le

        new_expectation = npc.tensordot(
            new_le, re, (['vR', 'vR*'], ['vL', 'vL*'])
        )

        return new_expectation

    def sweep_right_unitaries(self):
        """
        Optimise each unitary of right_boundary_unitaries in turn.
        """
        # Initialize right_right_environments after possibly updating the
        # unitaries on a previous sweep, as these are not updated during the
        # optimisation sweep.
        self.right_right_environments = list(accumulate(
            self.right_transfer_matrices[::-1],
            multiply_transfer_matrices_from_right,
            initial=self.right_right_environments[-1]
        ))[::-1]

        num_sites = len(self.right_boundary_unitaries)
        abs_expectations = np.zeros(num_sites)

        for boundary_unitary_index in range(num_sites):
            site_index = self.right_symmetry_index + 1 + boundary_unitary_index
            exp = self.solve_one_right_site(site_index, boundary_unitary_index)
            abs_expectations[boundary_unitary_index] = np.abs(exp)
        
        self.right_abs_expectations.append(abs_expectations)

    def optimize_right_unitaries(self):
        """
        Repeatedly optimize over all of right_boundary_unitaries until an
        iteration does not improve the absolute expectation by self.tolerance,
        even after adding one more site and optimising over that also.

        To-do:
            * Way to do without improving variable? Break out of while loop
              instead of setting to False?
            * Nest too deep at the end of the function.
        """
        # Variable which checks if the optimisation sweeps are improving the
        # expectation sufficiently wrt self.tolerance.
        improving = True

        while improving:
            # Initialise the previous absolute expectation value to check
            # against.
            if self.right_abs_expectations:
                prev_abs_expectation = self.right_abs_expectations[-1][-1]
            else:
                prev_abs_expectation = 0

            # Update and get new score.
            self.sweep_right_unitaries()

            new_abs_expectation = self.right_abs_expectations[-1][-1]

            improvement = new_abs_expectation - prev_abs_expectation

            # If not improving sufficiently, check if adding an additional
            # site would improve sufficiently.
            if (improvement < self.tolerance) and self.auto_pad:
                # Compute gradient of new site.
                tp_grad = expectation_gradient_from_environments(
                    self.psi,
                    self.rightmost_boundary_index+1,
                    left_environment=self.left_right_environments[-1],
                    mps_form='B'
                )

                S = np.linalg.svd(tp_grad.to_ndarray(), compute_uv=False)

                # Calculate score without explicitly finding optimal unitary.
                potential_exp = np.sum(S)

                potential_improvement = (
                    potential_exp
                    - prev_abs_expectation
                )

                if potential_improvement > self.tolerance:
                   self.pad_right()
                else:
                    improving=False
            elif (improvement < self.tolerance):
                improving = False

    def solve_one_left_site(self, site_index, boundary_unitary_index):
        """
        Optimise the absolute expectation value over the left boundary
        unitary located at site_index.

        To-do:
            * 1 index would do, rather than 2?

        Parameters
        ----------
        site_index: integer
            The site index of self.psi of the left boundary unitary to be
            optimised.
        boundary_unitary_index: integer
            The index of the left boundary unitary in
            left_boundary_unitaries.

        Returns
        -------
        complex float
            Left expectation after optimising unitary.
        """
        # Compute the gradient of the expectation with respect to the
        # operator to be optimised.
        le = self.left_left_environments[boundary_unitary_index+1]
        re = self.right_left_environments[boundary_unitary_index]

        tp_grad = expectation_gradient_from_environments(
            self.psi,
            site_index,
            left_environment=le,
            right_environment=re,
            mps_form='A'
        )

        grad = tp_grad.to_ndarray()

        # Simple formula to find the optimised unitary, and then update.
        new_unitary = unitarize_matrix(grad.conj())
        self.left_boundary_unitaries[boundary_unitary_index] = new_unitary

        # Update relevant variables and output new expectation.
        new_tm = get_transfer_matrix_from_unitary(
            self.psi, 
            site_index,
            unitary=new_unitary,
            form='A'
        )

        self.left_transfer_matrices[boundary_unitary_index] = new_tm
        new_re = multiply_transfer_matrices_from_right(re, new_tm)
        self.right_left_environments[boundary_unitary_index+1]=new_re

        new_expectation = npc.tensordot(
            le, new_re, (['vR', 'vR*'], ['vL', 'vL*'])
        )

        return new_expectation

    def sweep_left_unitaries(self):
        """
        Optimise each unitary of left_boundary_unitaries in turn.
        """
        # Initialize left_left_environments after possibly updating the
        # unitaries on a previous sweep, as these are not updated during the
        # optimisation sweep.
        self.left_left_environments = list(accumulate(
            self.left_transfer_matrices[::-1],
            multiply_transfer_matrices,
            initial=self.left_left_environments[-1]
        ))[::-1]

        num_sites = len(self.left_boundary_unitaries)
        abs_expectations = np.zeros(num_sites)

        for boundary_unitary_index in range(num_sites):
            site_index = self.left_symmetry_index - 1 - boundary_unitary_index
            exp = self.solve_one_left_site(site_index, boundary_unitary_index)
            abs_expectations[boundary_unitary_index] = np.abs(exp)
        
        self.left_abs_expectations.append(abs_expectations)

    def optimize_left_unitaries(self):
        """
        Repeatedly optimize over all of left_boundary_unitaries until an
        iteration does not improve the absolute expectation by self.tolerance,
        even after adding one more site and optimising over that also.

        To-do:
            * Way to do without improving variable? Break out of while loop
              instead of setting to False?
            * Nest too deep later in the function.
        """
        # Variable which checks if the optimisation sweeps are improving the
        # expectation sufficiently wrt self.tolerance.
        improving = True

        while improving:
            # Initialise the previous absolute expectation value to check
            # against.
            if self.left_abs_expectations:
                prev_abs_expectation = self.left_abs_expectations[-1][-1]
            else:
                prev_abs_expectation = 0

            # Update and get new score.
            self.sweep_left_unitaries()

            new_abs_expectation = self.left_abs_expectations[-1][-1]

            improvement = new_abs_expectation - prev_abs_expectation 

            # If not improving sufficiently, check if adding an additional
            # site would improve sufficiently.
            if (improvement < self.tolerance) and self.auto_pad:
                # Compute gradient of new site.
                tp_grad = expectation_gradient_from_environments(
                    self.psi,
                    self.leftmost_boundary_index - 1,
                    right_environment=self.right_left_environments[-1],
                    mps_form='A'
                )
                
                S = np.linalg.svd(tp_grad.to_ndarray(), compute_uv=False)
                
                # Calculate score without explicitly finding optimal unitary.
                potential_exp = np.sum(S)

                potential_improvement = (
                    potential_exp
                    - prev_abs_expectation
                )

                if potential_improvement > self.tolerance:
                   self.pad_left()
                else:
                    improving=False
            elif (improvement < self.tolerance):
                improving = False

    def optimize(self):
        """
        Optimise over the right_boundary_unitaries and then
        left_boundary_unitaries.
        """
        self.optimize_right_unitaries()
        self.optimize_left_unitaries()

    def get_abs_expectation(self):
        """
        Return the absolute value of the overall expectation.

        Returns
        -------
        float
            Absolute value of overall expectation value.
        """
        right_exp = self.right_abs_expectations[-1][-1]
        left_exp = self.left_abs_expectations[-1][-1]

        return left_exp * self.symmetry_singular_value * right_exp

    def to_SymmetryActionWithBoundaryUnitaries(self):
        out = SymmetryActionWithBoundaryUnitaries(
            self.psi,
            ic.symmetry_operations,
            self.left_symmetry_index,
            self.left_boundary_unitaries,
            self.right_boundary_unitaries
        )

        out.left_projected_symmetry_state = self.left_projected_symmetry_state
        out.right_projected_symmetry_state = (
            self.right_projected_symmetry_state
        )

        out.npc_symmetry_transfer_matrix = ic.npc_symmetry_transfer_matrix

        out.symmetry_transfer_matrix_singular_vals = (
            ic.symmetry_transfer_matrix_singular_vals
        )

        out.right_transfer_matrices = self.right_transfer_matrices
        out.left_transfer_matrices = self.left_transfer_matrices

        right_transfer_vectors=self.left_right_environments
        left_transfer_vector=self.right_left_environmentss

        return out
