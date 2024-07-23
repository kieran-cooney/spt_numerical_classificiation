"""
To-do:
    * Track unitaries as optimising.
    * Add "current expectation" function.
    * Currently tolerance is checking left_expectation and right_expectation,
      which could be quite different. Should really check wrt the overall
      expectation.
    * Docstring at top of file.
    * Provide optimisation formulae/derivation somewhere.
"""
from itertools import accumulate

import numpy as np

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

DEFAULT_TOLERANCE = 1e-3

class OneSiteSolver:
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
    left_symmetry_index: non-negative integer
        The index of the left most site of psi which the symmetry operations
        act on. 
    right_symmetry_index: non-negative integer
        The index of the right most site of psi whcih the symmetry operations
        act on.
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
    """
    def __init__(self, initial_conditions, tolerance=DEFAULT_TOLERANCE):
        """
        Parameters
        ----------
        initial_conditions: SPTOptimization.SymmetryActionaWithBoundaryUnitaries.SymmetryActionaWithBoundaryUnitaries
            Class detailing the initial conditions of the problem, including
            the MPS, the symmetry operations, initial boundary unitaries and
            more.
        tolerance: float
            A float between zero and 1 specifying when the iterative 
            optimization should stop. Set to default value if not specified.
        """
        ic = initial_conditions

        # Read in paramaters from initial_conditions.
        # Probably a better way to do this. Maybe subclassing?
        self.psi = ic.psi
        self.left_symmetry_index = ic.left_symmetry_index
        self.right_symmetry_index = ic.right_symmetry_index
        self.leftmost_boundary_index = ic.leftmost_boundary_index
        self.rightmost_boundary_index = ic.rightmost_boundary_index
        self.left_boundary_unitaries = ic.left_boundary_unitaries
        self.right_boundary_unitaries = ic.right_boundary_unitaries
        self.left_projected_symmetry_state = ic.left_projected_symmetry_state
        self.right_projected_symmetry_state = ic.right_projected_symmetry_state
        self.symmetry_singular_value = (
            ic.symmetry_transfer_matrix_singular_vals[0]
        )
        self.right_transfer_matrices = ic.right_transfer_matrices
        self.left_transfer_matrices = ic.left_transfer_matrices

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

    def pad_right(num_sites=1):
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
            new_rightmost_boundary_index
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
        self.right_boundary_unitaries.append(new_boundary_unitaries)
        self.rightmost_boundary_index = new_rightmost_boundary_index
        self.right_transfer_matrices.append(new_transfer_matrices)
        self.left_right_environments.append(new_left_right_environments)
        self.right_right_environments.append(new_right_right_environments)

    def pad_left(num_sites=1):
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
            new_leftmost_boundary_index,
            step=-1
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
        self.left_boundary_unitaries.append(new_boundary_unitaries)
        self.leftmost_boundary_index = new_leftmost_boundary_index
        self.left_transfer_matrices.append(new_transfer_matrices)
        self.right_left_environments.append(new_right_left_environments)
        self.left_left_environments.append(new_left_left_environments)

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

        grad = expectation_gradient_from_environments(
            self.psi,
            site_index,
            left_environment=le,
            right_environment=re,
            mps_form='B'
        )

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
            self.right_transfer_matrices,
            multiply_transfer_matrices_from_right,
            initial=self.right_right_environments[-1]
        ))

        num_sites = len(self.right_boundary_unitaries)
        abs_expectations = np.zeros(num_sites)

        for boundary_unitary_index in range(num_sites):
            site_index = self.right_symmetry_index + 1 + boundary_unitary_index
            exp = solve_one_right_site(site_index, boundary_unitary_index)
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

            improvement = prev_abs_expectation - new_abs_expectation

            # If not improving sufficiently, check if adding an additional
            # site would improve sufficiently.
            if improvement < self.tolerance:
                # Compute gradient of new site.
                grad = expectation_gradient_from_environments(
                    self.psi,
                    self.rightmost_boundary_index+1,
                    left_environment=self.left_right_environments[-1],
                    mps_form='B'
                )

                # Calculate score without explicitly finding optimal unitary.
                potential_exp = np.abs(np.trace(grad))

                potential_improvement = (
                    prev_abs_expectation
                    - potential_exp
                )

                if potential_improvement > self.tolerance:
                   self.pad_right()
                else:
                    improving=False

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

        grad = expectation_gradient_from_environments(
            self.psi,
            site_index,
            left_environment=le,
            right_environment=re,
            mps_form='A'
        )

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
        new_re = multiply_transfer_matrices_from_right(new_tm, re)
        self.right_left_environments[boundary_unitary_index+1]=new_re

        new_expectation = npc.tensordot(
            new_re, le, (['vR', 'vR*'], ['vL', 'vL*'])
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
            self.left_transfer_matrices,
            multiply_transfer_matrices,
            initial=self.left_left_environments[-1]
        ))

        num_sites = len(self.left_boundary_unitaries)
        abs_expectations = np.zeros(num_sites)

        for boundary_unitary_index in range(num_sites):
            site_index = self.left_symmetry_index - 1 - boundary_unitary_index
            exp = solve_one_left_site(site_index, boundary_unitary_index)
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

            improvement = prev_abs_expectation - new_abs_expectation

            # If not improving sufficiently, check if adding an additional
            # site would improve sufficiently.
            if improvement < self.tolerance:
                # Compute gradient of new site.
                grad = expectation_gradient_from_environments(
                    self.psi,
                    self.leftmost_boundary_index - 1,
                    right_environment=self.right_left_environments[-1],
                    mps_form='A'
                )

                # Calculate score without explicitly finding optimal unitary.
                potential_exp = np.abs(np.trace(grad))

                potential_improvement = (
                    prev_abs_expectation
                    - potential_exp
                )

                if potential_improvement > self.tolerance:
                   self.pad_left()
                else:
                    improving=False

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
