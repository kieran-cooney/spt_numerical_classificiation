"""
To-do:
    Function imports/definitions:
    * Currently very static. Add options for manually adding layers, widening,
      etc. Eventually allow for dynamically changing these variables as the
      optimisation demands...?
    * Compare with original BrickSolver.py, duplication?
    * Bricks or blocks...
"""
import tenpy.linalg.np_conserved as npc

from SPTOptimization.Optimizers.utils import (
    one_site_optimization_sweep_right
)

from SPTOptimization.utils import (
    get_npc_identity_operator,
    group_elements,
    split_combined_b,
    get_left_side_right_symmetry_environment,
    combine_grouped_b_tensors,
    multiply_stacked_unitaries_against_mps
)

from SPTOptimization.tenpy_leg_label_utils import (
    swap_left_right_indices,
    get_physical_leg_labels,
    conjugate_leg_label,
    is_single_physical_leg_label,
    is_grouped_physical_leg_label
)

DEFAULT_NUM_SITES = 3
DEFAULT_BLOCK_WIDTH = 3
DEFAULT_NUM_LAYERS = 1
DEFAULT_NUM_ONE_SIDED_ITERATIONS = 3
DEFAULT_NUM_TWO_SIDED_ITERATIONS = 3

MAX_VIRTUAL_BOND_DIM=8

class MPSBrickSolver:
    def __init__(self, symmetry_case, num_sites=DEFAULT_NUM_SITES,
                 block_width=DEFAULT_BLOCK_WIDTH,
                 num_layers=DEFAULT_NUM_LAYERS,
                 num_one_sided_iterations=DEFAULT_NUM_ONE_SIDED_ITERATIONS,
                 num_two_sided_iterations=DEFAULT_NUM_TWO_SIDED_ITERATIONS,
                 max_virtual_bond_dim=MAX_VIRTUAL_BOND_DIM
                 ):
        """
        Parameters
        ----------

        """
        self.symmetry_case = symmetry_case

        self.num_sites=num_sites
        self.block_width=block_width
        self.num_layers=num_layers
        self.num_one_sided_iterations=num_one_sided_iterations
        self.num_two_sided_iterations=num_two_sided_iterations
        self.block_offset = 0

        self.max_virtual_bond_dim = max_virtual_bond_dim

        right_site_indices = list(range(
            self.symmetry_case.right_symmetry_index + 1,
            self.symmetry_case.right_symmetry_index + 1 + self.num_sites
        ))

        self.bottom_right_mps_tensors = [
            self.symmetry_case.psi.get_B(i)
            for k, i in enumerate(right_site_indices)
        ]

        self.current_top_right_mps_tensors = self.bottom_right_mps_tensors
        self.top_right_mps_tensors = self.bottom_right_mps_tensors

        self.original_right_side_left_schmidt_values =  [
            self.symmetry_case.psi.get_SL(i)
            for i in right_site_indices
        ]

        self.current_right_side_left_schmidt_values =  (
            self.original_right_side_left_schmidt_values.copy()
        )

        left_site_indices = list(range(
            self.symmetry_case.left_symmetry_index - 1,
            self.symmetry_case.left_symmetry_index - 1 - self.num_sites,
            -1
        ))

        left_mps_tensors = [
            self.symmetry_case.psi.get_B(i, form='A')
            for k, i in enumerate(left_site_indices)
        ]

        self.bottom_left_mps_tensors = [
            swap_left_right_indices(b) for b in left_mps_tensors
        ]

        self.current_top_left_mps_tensors = self.bottom_left_mps_tensors
        self.top_left_mps_tensors = self.bottom_left_mps_tensors

        self.original_left_side_right_schmidt_values = [
            self.symmetry_case.psi.get_SR(i)
            for i in left_site_indices
        ]

        self.current_left_side_right_schmidt_values = (
            self.original_left_side_right_schmidt_values.copy()
        )
        self.left_unitaries = list()
        self.right_unitaries = list()
        self.left_expectations = list()
        self.right_expectations = list()

        self.__update_left_side_right_symmetry_environment()
        self.__update_right_side_left_symmetry_environment()

    # All the swap_left_right_indices usage is annoying and could be done without...
    def __update_right_side_left_symmetry_environment(self):
        transfer_matrix = self.symmetry_case.npc_symmetry_transfer_matrix

        out = (
            get_left_side_right_symmetry_environment(
                self.current_top_left_mps_tensors,
                self.bottom_left_mps_tensors,
                swap_left_right_indices(transfer_matrix),
                left_side_environment=True
            )
        )

        right_side_left_symmetry_environment, right_side_environment = out

        self.right_side_left_symmetry_environment = swap_left_right_indices(
            right_side_left_symmetry_environment
        )

        self.right_side_environment = swap_left_right_indices(
            right_side_environment
        )

    def __update_left_side_right_symmetry_environment(self):
        transfer_matrix = self.symmetry_case.npc_symmetry_transfer_matrix

        out = (
            get_left_side_right_symmetry_environment(
                self.current_top_right_mps_tensors,
                self.bottom_right_mps_tensors,
                transfer_matrix,
                left_side_environment=True
            )
        )

        left_side_right_symmetry_environment, left_side_environment = out

        self.left_side_right_symmetry_environment = swap_left_right_indices(
            left_side_right_symmetry_environment
        )

        # Name is wonky...
        self.left_side_environment = swap_left_right_indices(
            left_side_environment
        )

    @staticmethod
    def optimise_right_side_one_layer(
            left_environment,
            top_b_tensors,
            left_schmidt_values,
            block_width,
            block_offset,
            unitaries,
            bottom_b_tensors=None,
            num_iterations=1,
            max_virtual_bond_dim=MAX_VIRTUAL_BOND_DIM,
        ):

        if bottom_b_tensors is None:
            bottom_b_tensors = top_b_tensors

        group = lambda x: group_elements(x, block_width, block_offset)
        top_grouped_bs = group(top_b_tensors)
        bottom_grouped_bs = group(bottom_b_tensors)
        grouped_schmidt_values = group(left_schmidt_values)

        top_combined_bs = combine_grouped_b_tensors(top_grouped_bs)
        bottom_combined_bs = combine_grouped_b_tensors(bottom_grouped_bs)

        expectations = list()

        for _ in range(num_iterations):
            exps, *_ = one_site_optimization_sweep_right(
                left_environment,
                top_combined_bs,
                unitaries,
                bottom_combined_bs
            )

            expectations.append(exps)

        for i, u in enumerate(unitaries):
            b = top_combined_bs[i]
            ll = get_physical_leg_labels(b)[0]
            llh = conjugate_leg_label(ll)
        
            new_b = npc.tensordot(b, u, [[ll,], [llh,]])
        
            top_combined_bs[i] = new_b

        new_top_bs = list()
        #new_left_schmidt_values = left_schmidt_values.copy()
        new_left_schmidt_values = list()

        # To-do:
        # Don't need to do this step always, just when moving to a new layer.
        # Might be useful to make into a separate function, perform similar
        # logic in utils.multiply_blocked_unitaries_against_mps
        for b, s in zip (top_combined_bs, grouped_schmidt_values):
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

        return new_top_bs, new_left_schmidt_values, expectations, unitaries

    def optimise_right_layer(self):
        out = MPSBrickSolver.optimise_right_side_one_layer(
            self.right_side_left_symmetry_environment,
            self.top_right_mps_tensors,
            self.current_right_side_left_schmidt_values,
            self.block_width,
            self.block_offset,
            self.right_current_layer_unitaries,
            bottom_b_tensors=self.bottom_right_mps_tensors,
            num_iterations=self.num_one_sided_iterations,
            max_virtual_bond_dim=self.max_virtual_bond_dim
        )

        new_top_bs, new_schmidt_values, expectations, _ = out

        self.current_top_right_mps_tensors = new_top_bs
        self.future_right_side_left_schmidt_values = new_schmidt_values
        self.right_expectations[-1].append(expectations)
        
        self.__update_left_side_right_symmetry_environment()

    def optimise_left_layer(self):
        out = MPSBrickSolver.optimise_right_side_one_layer(
            self.left_side_right_symmetry_environment,
            self.top_left_mps_tensors,
            self.current_left_side_right_schmidt_values,
            self.block_width,
            self.block_offset,
            self.left_current_layer_unitaries,
            bottom_b_tensors=self.bottom_left_mps_tensors,
            num_iterations=self.num_one_sided_iterations,
            max_virtual_bond_dim=self.max_virtual_bond_dim
        )

        new_top_bs, new_schmidt_values, expectations, _ = out

        self.current_top_left_mps_tensors = new_top_bs
        self.future_left_side_right_schmidt_values = new_schmidt_values
        self.left_expectations[-1].append(expectations)

        self.__update_right_side_left_symmetry_environment()

    def two_sided_optimise_layer_one_iteration(self):
        self.optimise_left_layer()
        self.optimise_right_layer()

    def two_sided_optimise_layer(self):
        for _ in range(self.num_two_sided_iterations):
            self.two_sided_optimise_layer_one_iteration()

    def initialize_unitaries(self):
        # There is some duplication of code here with
        # optimise_right_side_one_layer, not good.

        group = lambda x: group_elements(x, self.block_width, self.block_offset)

        # Left side
        top_left_grouped_bs = group(self.top_left_mps_tensors)
        top_left_combined_bs = combine_grouped_b_tensors(top_left_grouped_bs)

        self.left_current_layer_unitaries = [
            get_npc_identity_operator(t) for t in top_left_combined_bs
        ]

        # Right side
        top_right_grouped_bs = group(self.top_right_mps_tensors)
        top_right_combined_bs = combine_grouped_b_tensors(top_right_grouped_bs)

        self.right_current_layer_unitaries = [
            get_npc_identity_operator(t) for t in top_right_combined_bs
        ]

    def add_new_layer(self):
        self.left_expectations.append(list())
        self.right_expectations.append(list())

        self.initialize_unitaries()

    def finish_current_layer(self):
        self.block_offset += self.block_width//2
        self.block_offset = (self.block_offset)//self.block_width

        self.left_unitaries.append(self.left_current_layer_unitaries)
        self.right_unitaries.append(self.right_current_layer_unitaries)

        self.top_right_mps_tensors = self.current_top_right_mps_tensors
        self.current_right_side_left_schmidt_values = (
            self.future_right_side_left_schmidt_values
        )
        self.top_left_mps_tensors = self.current_top_left_mps_tensors
        self.current_left_side_right_schmidt_values = (
            self.future_left_side_right_schmidt_values
        )

    def optimise(self):
        for _ in range(self.num_layers):
            self.add_new_layer()
            self.two_sided_optimise_layer()
            self.finish_current_layer()


    @staticmethod
    def __flatten_list(l):
        return [e for l1 in l for e in l1]

    def flatten_exps(self):
        """
        Returns a flattened version of left_expectations and
        right_expectations which has a flat list of expectations with respect
        to "time".
        """
        gen_1 = (
            MPSBrickSolver.__flatten_list(l)
            for l1 in self.left_expectations
            for l in l1
        )

        gen_2 = (
            MPSBrickSolver.__flatten_list(l)
            for l1 in self.right_expectations
            for l in l1
        )

        out = list()

        for ll, lr in zip(gen_1, gen_2):
            out.extend(ll)
            out.extend(lr)

        return out

    def manual_expectation_value(self):
        # Currently this method only accounts for the "closed" layers.
        top_right_bs, _ = multiply_stacked_unitaries_against_mps(
            self.right_unitaries,
            self.bottom_right_mps_tensors,
            self.original_right_side_left_schmidt_values,
            self.max_virtual_bond_dim
        )

        right_t = npc.tensordot(
            top_right_bs[-1],
            self.bottom_right_mps_tensors[-1].conj(),
            [['vR', 'p'], ['vR*', 'p*']]
        )

        b_pairs = zip(top_right_bs[-2::-1], self.bottom_right_mps_tensors[-2::-1])

        for bt, bb in b_pairs:
            right_t = npc.tensordot(
                right_t,
                bt,
                [['vL',], ['vR',]]
            )

            right_t = npc.tensordot(
                right_t,
                bb.conj(),
                [['vL*', 'p'], ['vR*', 'p*']]
            )

        right_t = npc.tensordot(
            right_t,
            self.symmetry_case.npc_symmetry_transfer_matrix,
            [['vL', 'vL*'], ['vR', 'vR*']]
        )

        top_left_bs, _ = multiply_stacked_unitaries_against_mps(
            self.left_unitaries,
            self.bottom_left_mps_tensors,
            self.original_left_side_right_schmidt_values,
            self.max_virtual_bond_dim
        )

        b_pairs = zip(top_left_bs, self.bottom_left_mps_tensors)

        for bt, bb in b_pairs:
            right_t = npc.tensordot(
                right_t,
                swap_left_right_indices(bt),
                [['vL',], ['vR']]
            )

            right_t = npc.tensordot(
                right_t,
                swap_left_right_indices(bb).conj(),
                [['vL*', 'p'], ['vR*', 'p*']]
            )

        out = npc.trace(right_t, 'vL', 'vL*')

        return out
