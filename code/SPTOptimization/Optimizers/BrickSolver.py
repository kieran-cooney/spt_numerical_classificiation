"""
"""
import re 

import numpy as np
import tenpy.linalg.np_conserved as npc

"""
from SPTOptimization.utils import (
    unitarize_matrix,
    get_transfer_matrix_from_unitary,
    multiply_transfer_matrices,
    multiply_transfer_matrices_from_right,
    get_right_identity_environment,
    get_left_identity_environment,
    get_identity_operator
)
"""

from SPTOptimization.utils import (
    get_right_identity_environment_from_tp_tensor,
    get_left_identity_environment_from_tp_tensor,

"""
from SPTOptimization.SymmetryActionWithBoundaryUnitaries import (
    SymmetryActionWithBoundaryUnitaries
)
"""

from SPTOptimization.Optimizers.utils import (
    one_site_optimization_sweep_right,
    one_site_optimization_sweep_left,
    max_expectation
)

# from SPTOptimization.gradients import expectation_gradient_from_environments
WIDTH_TOLERANCE = 1e-3
DEPTH_TOLERANCE = 1e-3
GLOBAL_TOLERANCE = 1e-2

P_LEG_LABEL_REGEX_STRING = r"^p\d*$"
p_leg_pattern = re.compile(P_LEG_LABEL_REGEX_STRING)

class BrickSolver:
    """
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

    top_right_bs
    top_right_grouped_bs
    bottom_right_bs
    bottom_right_grouped_bs
    top_left_as
    top_left_grouped_as
    bottom_left_as
    bottom_left_grouped_as

    left_symmetry_environment
    right_symmetry_environment

    lots of indices

    expectations(r/l)
    current_expectations(r/l)

    unitaries(r/l)
    current_unitaries(r/l)

    fringe_right_b1
    fringe_right_b2
    fringe_right_grouped_b
    fringe_left_a1
    fringe_left_a2
    fringe_left_grouped_a
    fringe_right_grad_tensor
    fringe_left_grad_tensor

    symmetry_transfer_matrix

    current_right_transfer_matrices
    current_left_transfer_matrices

    left_left_environments
    right_right_environments

    rightmost_right_environment
    leftmost_left_environment
    """
    def __init__(self, initial_conditions, width_tolerance=WIDTH_TOLERANCE,
                 depth_tolerance=DEPTH_TOLERANCE):
        """
        Parameters
        ----------
        initial_conditions: SPTOptimization.SymmetryActionaWithBoundaryUnitaries.SymmetryActionaWithBoundaryUnitaries
            Class detailing the initial conditions of the problem, including
            the MPS, the symmetry operations, initial boundary unitaries and
            more.
        width_tolerance: float
            A float between zero and 1 specifying when to stop increasing the
            number of sites in a layer of unitaries. Set to default value if
            not specified.
        depth_tolerance: float
            A float between zero and 1 specifying when to stop increasing the
            number of layers of unitaries. Set to default value if not
            specified.
        """
        ic = initial_conditions

        # Read in paramaters from initial_conditions.
        # Probably a better way to do this. Maybe subclassing?
        self.psi = ic.psi
        self.left_symmetry_index = ic.left_symmetry_index
        self.right_symmetry_index = ic.right_symmetry_index

        self.symmetry_transfer_matrix = ic.npc_symmetry_transfer_matrix
        self.left_symmetry_environment = (
            ic.left_projected_symmetry_state
        )
        self.right_symmetry_environmnent = (
            ic.right_projected_symmetry_state
        )

        self.width_tolerance = width_tolerance
        self.depth_tolerance = depth_tolerance

        self.num_layers = 0

        self.right_Bs = list()
        self.left_As = list()

        self.right_expectations = list()
        self.left_expectations = list()

    @staticmethod
    def extract_p_leg_label_from_tensor(b):
        out = next(
            l for l in b.get_leg_labels()
            if (p_leg_pattern.match(l))
        )

        return out

    @staticmethod
    def combine_b_tensors(b1, b2):
        b = npc.tensordot(b1, b2, ['vR', 'vL'])

        p1 = extract_p_leg_label_from_tensor(b1)
        p2 = extract_p_leg_label_from_tensor(b2)

        b = b.combine_legs([p1, p2])

        return b

    def expand_right_unitaries_once(self, grad=None):
        if grad is None:
            grad = multiply_transfer_matrices(
                self.fringe_right_grad_tensor,
                self.rightmost_left_environment
            )

        new_exp, new_unitary = one_site_optimization(grad)

        self.current_right_unitaries.append(new_unitary)
        self.current_right_expectations.append(new_exp)
        self.expectation = new_exp

        old_fringe_bs = [self.fringe_right_b1, self.fringe_right_b2]
        self.top_right_bs.extend(old_fringe_bs)
        self.bottom_right_bs.extend(old_fringe_bs)

        self.top_right_grouped_bs.append(self.fringe_right_grouped_b)
        self.bottom_right_grouped_bs.append(self.fringe_right_grouped_b)

        # NEXT: figure out environment variables, and move on to update layer once
        self.rightmost_right_environment
        
        transfer_matrix = get_transfer_matrix_from_tp_unitary_and_b_tensor(
            self.fringe_right_grouped_b,
            new_unitary,
            self.fringe_right_grouped_b,
        )

        self.current_right_transfer_matrices.append(transfer_matrix)

        self.fringe_right_b1 = (
            self.psi
            .get_B(self.rightmost_site_index + 1, form='B')
            .replace_label('p', f'p{num_right_boundary_sites + 1}')
        )

        self.fringe_right_b2 = (
            self.psi
            .get_B(self.rightmost_site_index + 2, form='B')
            .replace_label('p', f'p{num_right_boundary_sites + 2}')
        )

        self.fringe_right_grouped_b = self.combine_b_tensors(
            self.fringe_right_b1,
            self.fringe_right_b2
        )
        
        b = self.fringe_right_grouped_b
        self.fringe_right_grad_tensor = npc.tensordot(
            b, b.con(), [['vR',], ['vR*']]
        )

        self.num_right_boundary_sites += 2
        self.rightmost_site_index += 2

    def expand_right_unitaries(self):
        expanding = True

        while expanding:
            potential_grad = multiply_transfer_matrices(
                self.fringe_right_grad_tensor,
                self.rightmost_left_environment
            )

            potential_score = max_expectation(potential_grad)

            if (potential_score > self.expectation + self.width_tolerance):
                self.expand_right_unitaries_once(potential_grad)
            else:
                expanding = False

    def optimize_right_unitaries(self):

        opt_triple = one_site_optimization_sweep_right(
            self.left_symmetry_environment,
            self.top_right_grouped_bs,
            self.current_right_unitaries,
            self.bottom_right_grouped_bs
        )

        exps = opt_triple[0]
        self.rightmost_left_environment = opt_triple[1]
        self.current_right_transfer_matrices = opt_triple[2]

        self.current_right_expectations = expectations
        self.expectation = expectations[-1]

        self.expand_right_unitaries()

    def expand_left_unitaries_once(self, grad=None):
        if grad is None:
            grad = multiply_transfer_matrices(
                self.leftmost_right_environment,
                self.fringe_left_grad_tensor
            )

        new_exp, new_unitary = one_site_optimization(grad)

        self.current_left_unitaries.append(new_unitary)
        self.current_left_expectations.append(new_exp)
        self.expectation = new_exp

        old_fringe_as = [self.fringe_right_a1, self.fringe_right_a2]
        self.top_left_as.extend(old_fringe_as)
        self.bottom_right_as.extend(old_fringe_as)

        self.top_left_grouped_as.append(self.fringe_left_grouped_a)
        self.bottom_left_grouped_as.append(self.fringe_left_grouped_a)

        transfer_matrix = get_transfer_matrix_from_tp_unitary_and_b_tensor(
            self.fringe_left_grouped_a,
            new_unitary,
            self.fringe_left_grouped_a,
        )

        self.current_left_transfer_matrices.append(transfer_matrix)

        self.fringe_left_a1 = (
            self.psi
            .get_B(self.leftmost_site_index - 1, form='A')
            .replace_label('p', f'p{num_left_boundary_sites + 1}')
        )

        self.fringe_left_a2 = (
            self.psi
            .get_B(self.leftmost_site_index - 2, form='A')
            .replace_label('p', f'p{num_right_boundary_sites + 2}')
        )

        self.fringe_left_grouped_a = self.combine_b_tensors(
            self.fringe_left_a1,
            self.fringe_left_a2
        )
        
        a = self.fringe_left_grouped_a
        self.fringe_left_grad_tensor = npc.tensordot(
            a, a.con(), [['vL',], ['vL*']]
        )

        self.num_left_boundary_sites += 2
        self.leftmost_site_index -= 2

    def expand_left_unitaries(self):
        expanding = True

        while expanding:
            potential_grad = multiply_transfer_matrices(
                self.fringe_left_site_tensor,
                self.rightmost_left_environment
            )

            potential_score = max_expectation(potential_grad)

            if (potential_score > self.expectation + self.width_tolerance):
                self.expand_left_unitaries_once(potential_grad)
            else:
                expanding = False

    def optimize_left_unitaries(self):

        opt_triple = one_site_optimization_sweep_left(
            self.left_symmetry_environment,
            self.top_left_grouped_as,
            self.current_left_unitaries,
            self.bottom_left_grouped_as
        )

        exps = opt_triple[0]
        self.leftmost_right_environment = opt_triple[1]
        self.current_left_transfer_matrices = opt_triple[2]

        self.current_left_expectations = expectations
        self.expectation = expectations[-1]

        self.expand_left_unitaries()

    def update_left_left_environments(self):

    def update_right_right_environments(self):

    def optimize_layer_one_iteration(self):
        # Compute left-left environments (don't need to compute later!)
        # Find right symmetry environment
        # Update right unitaries (when are the right-right environments computed initally?)
        # Compute right-right environments
        # Find left symmetry environment
        # Update left unitaries


    def optimize_layer(self):

    def optimize(self):
        self.newsite_tensor = 

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
