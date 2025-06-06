"""
To-do:
    * Add overall description
    * Document class parameters
    * Functionality for variable bond dimension from site to site?
    * There is an additional hyperparamter we can add, unitary loss is like
      sqrt(1+unitarity), could be sqrt(1+a*unitarity). Currently just setting
      a=1.
"""
from functools import reduce
from operator import mul

import numpy as np
import tenpy.linalg.np_conserved as npc

from SPTOptimization.utils import (
    get_right_identity_environment_from_tp_tensor,
    get_left_identity_environment_from_tp_tensor,
    to_npc_array,
    get_physical_dim
)

from SPTOptimization.mpo_utils import (
    partial_mpo_mpo_contraction_from_right,
    partial_order_4_mpo_contraction_from_right,
    partial_mpo_mps_contraction_from_right,
    get_random_mpo_tensors,
    get_identity_mpo_tensors,
    rescale_mpo_tensors,
    mpo_socket_tensor_contraction,
)

from SPTOptimization.tenpy_leg_label_utils import (
    swap_left_right_indices
)

from SPTOptimization.Optimizers.utils import (
    mpo_tensor_raw_to_gradient,
)

from SPTOptimization.Optimizers.AdamTenpy import AdamTenpy

DEFAULT_NUM_SITES=6
DEFAULT_VIRTUAL_BOND_DIM=8
DEFAULT_UNITARITY_LEARNING_RATE=1
DEFAULT_UNITARITY_SHAPE_PARAMETER=1
DEFAULT_OVERLAP_LEARNING_RATE=1
# alpha, beta1 & beta2
DEFAULT_ADAM_PARAMS=(1e-4, 0.35, 0.35)

class MPOGradientDescent:

    @staticmethod
    def mpo_gradient_descent_one_side_one_iteration(mpo_tensors, b_tensors,
        total_dimension, right_overlap_tensors, unitarity_learning_rate,
        unitarity_shape_parameter, overlap_learning_rate, left_environment,
        adam_optimizers, overlap_target=1):
        """
        Absolute mess of a function, should modularise a bit better.
        """
        # Initialise list of gradients to be filled
        grads = list()

        # Initialise variables
        right_unitary_two_tensors = (
            partial_mpo_mpo_contraction_from_right(mpo_tensors)
        )
        right_unitary_four_tensors = (
            partial_order_4_mpo_contraction_from_right(mpo_tensors)
        )
        
        # Why don't I make a function for these two?
        left_unitary_two_tensors = list()
        left_unitary_four_tensors = list()
        left_overlap_tensors = list()

        num_sites = len(mpo_tensors)
        assert len(mpo_tensors) == len(b_tensors)

        # Leftmost site
        w = mpo_tensors[0]
        b = b_tensors[0]

        t = right_unitary_two_tensors[0]

        # Second order terms
        grad_2 = npc.tensordot(t, w, [['vL'], ['vR',]])

        order_2_score = mpo_socket_tensor_contraction(
            grad_2,
            w.conj(),
            [['vL*',], ['vR*',]]
        )
        order_2_score = order_2_score.real

        grad_2 = mpo_tensor_raw_to_gradient(grad_2, w)

        # Fourth order terms
        t = right_unitary_four_tensors[0]

        grad_4 = npc.tensordot(t, w, [['vL'], ['vR',]])
        grad_4 = npc.tensordot(grad_4, w.conj(), [['vL*', 'p'], ['vR*', 'p*']])
        grad_4 = npc.tensordot(grad_4, w, [['vL1', 'p'], ['vR', 'p*']])

        order_4_score = mpo_socket_tensor_contraction(
            grad_4,
            w.conj(),
            [['vL1*',], ['vR*',]]
        )
        order_4_score = order_4_score.real

        grad_4 = mpo_tensor_raw_to_gradient(grad_4, w)

        unitary_score = (
            order_4_score
            - 2*order_2_score
            + total_dimension
        )

        unitary_grad = (
            unitarity_shape_parameter*(grad_4 - grad_2)
            /np.sqrt(1+unitarity_shape_parameter*unitary_score)
        )
        
        # Overlap terms
        t = right_overlap_tensors[0].conj().replace_label('vLm*', 'vLm')

        grad_o = npc.tensordot(t, b, [['vL'], ['vR',]])
        grad_o = npc.tensordot(grad_o, b.conj(), [['vL*',], ['vR*',]])
        grad_o = npc.tensordot(
            grad_o,
            left_environment.conj(),
            [['vL', 'vL*'], ['vR', 'vR*']]
        )

        c_conj = mpo_socket_tensor_contraction(
            grad_o,
            w.conj(),
            [['vLm',], ['vR*',]]
        )
        c = c_conj.conjugate()
        c_abs = np.abs(c)
        
        grad_o_scale = c*(1 - overlap_target/c_abs)
        grad_o = grad_o_scale*grad_o
        grad_o = mpo_tensor_raw_to_gradient(grad_o, w)

        grad = (
            unitarity_learning_rate*unitary_grad +
            overlap_learning_rate*grad_o
        )
        adam_grad = adam_optimizers[0].update(grad)
        grads.append(adam_grad)

        # Create and save left tensors
        t = npc.tensordot(w, w.conj(), [['p', 'p*'], ['p*', 'p']])
        left_unitary_two_tensors.append(t)

        t = npc.tensordot(w, w.conj(), [['p',], ['p*',]])
        t.ireplace_labels(['vR', 'vR*'], ['vR1', 'vR1*'])
        t = npc.tensordot(t, w, [['p',], ['p*',]])
        t = npc.tensordot(t, w.conj(), [['p', 'p*'], ['p*', 'p']])
        
        left_unitary_four_tensors.append(t)

        t = npc.tensordot(b, w.conj(), [['p',], ['p*',]])
        #print(t)
        t.ireplace_label('vR*', 'vRm')
        #print(t)
        t = npc.tensordot(t, left_environment, [['vL',], ['vR',]])
        #print(t)
        t = npc.tensordot(t, b.conj(), [['vR*', 'p'], ['vL*', 'p*']])

        #print(t)

        left_overlap_tensors.append(t)

        # Inner sites
        for i in range(1, num_sites-1):
            w = mpo_tensors[i]
            b = b_tensors[i]
        
            right_two_tensor = right_unitary_two_tensors[i]
            right_four_tensor = right_unitary_four_tensors[i]
            right_overlap_tensor = right_overlap_tensors[i].conj().replace_label('vLm*', 'vLm')

            # Order two terms
            left_two_tensor = left_unitary_two_tensors[-1]

            grad_2 = npc.tensordot(right_two_tensor, w, [['vL'], ['vR',]])
            grad_2 = npc.tensordot(grad_2, left_two_tensor, [['vL'], ['vR',]])

            grad_2 = mpo_tensor_raw_to_gradient(grad_2, w)

            # Order four terms
            left_four_tensor = left_unitary_four_tensors[-1]

            grad_4 = npc.tensordot(right_four_tensor, w, [['vL'], ['vR',]])
            grad_4 = npc.tensordot(grad_4, w.conj(), [['vL*', 'p'], ['vR*', 'p*']])

            grad_4 = npc.tensordot(
                grad_4,
                w.replace_label('vL', 'vL1'),
                [['vL1', 'p'], ['vR', 'p*']]
            )

            grad_4 = npc.tensordot(
                grad_4,
                left_four_tensor,
                [['vL', 'vL*', 'vL1'], ['vR', 'vR*', 'vR1']]
            )

            grad_4 = mpo_tensor_raw_to_gradient(grad_4, w)

            unitary_grad = (
                unitarity_shape_parameter*(grad_4 - grad_2)
                /np.sqrt(1+unitarity_shape_parameter*unitary_score)
            )

            # Overlap terms
            left_overlap_tensor = left_overlap_tensors[-1]

            grad_o = npc.tensordot(right_overlap_tensor, b, [['vL',], ['vR',]])
            grad_o = npc.tensordot(grad_o, b.conj(), [['vL*',], ['vR*',]])
            grad_o = npc.tensordot(
                grad_o,
                left_overlap_tensor,
                [['vL*', 'vL'], ['vR', 'vR*',]]
            )

            grad_o = grad_o_scale*grad_o
            grad_o = mpo_tensor_raw_to_gradient(grad_o, w)

            grad = (
                unitarity_learning_rate*unitary_grad +
                overlap_learning_rate*grad_o
            )
            adam_grad = adam_optimizers[i].update(grad)
            grads.append(adam_grad)

            # Update left tensors
            t = npc.tensordot(left_two_tensor, w, [['vR',], ['vL']])
            t = npc.tensordot(
                t,
                w.conj(),
                [['vR*', 'p', 'p*'], ['vL*', 'p*', 'p']]
            )
            
            left_unitary_two_tensors.append(t)
            
            t = npc.tensordot(left_four_tensor, w, [['vR',], ['vL']])
            t = npc.tensordot(t, w.conj(), [['vR*', 'p'], ['vL*', 'p*']])
            t = npc.tensordot(
                t,
                w.replace_label('vR', 'vR1'),
                [['p', 'vR1'], ['p*', 'vL']]
            )
            t = npc.tensordot(
                t,
                w.conj().replace_label('vR*', 'vR1*'),
                [['p', 'p*', 'vR1*'], ['p*', 'p', 'vL*']]
            )
            
            left_unitary_four_tensors.append(t)

            t = left_overlap_tensor.ireplace_label('vR*', 'vR1*')
            t = npc.tensordot(
                left_overlap_tensor,
                w.conj(),
                [['vRm',], ['vL*']]
            )
            t.ireplace_label('vR*', 'vRm')
            t = npc.tensordot(t, b, [['vR', 'p*'], ['vL', 'p']])
            t = npc.tensordot(t, b.conj(), [['vR1*', 'p'], ['vL*', 'p*']])

            left_overlap_tensors.append(t)

        # Last site
        left_two_tensor = left_unitary_two_tensors[-1]
        w = mpo_tensors[-1]
        b = b_tensors[-1]
        
        grad_2 = npc.tensordot(left_two_tensor, w, [['vR'], ['vL',]])
        grad_2 = mpo_tensor_raw_to_gradient(grad_2, w)

        left_four_tensor = left_unitary_four_tensors[-1]
        
        grad_4 = npc.tensordot(left_four_tensor, w, [['vR'], ['vL',]])
        grad_4 = npc.tensordot(grad_4, w.conj(), [['vR*', 'p'], ['vL*', 'p*']])
        grad_4 = npc.tensordot(grad_4, w, [['vR1', 'p'], ['vL', 'p*']])

        grad_4 = mpo_tensor_raw_to_gradient(grad_4, w)
        
        unitary_grad = (
            unitarity_shape_parameter*(grad_4 - grad_2)
            /np.sqrt(1+unitarity_shape_parameter*unitary_score)
        )

        left_overlap_tensor = left_overlap_tensors[-1]
        right_overlap_tensor = right_overlap_tensors[-1].conj()

        grad_o = npc.tensordot(right_overlap_tensor, b, [['vL',], ['vR',]])
        grad_o = npc.tensordot(grad_o, b.conj(), [['vL*',], ['vR*',]])
        grad_o = npc.tensordot(
            grad_o,
            left_overlap_tensor,
            [['vL*', 'vL'], ['vR', 'vR*',]]
        )

        grad_o = grad_o_scale*grad_o
        grad_o = mpo_tensor_raw_to_gradient(grad_o, w)


        grad = (
            unitarity_learning_rate*unitary_grad +
            overlap_learning_rate*grad_o
        )
        adam_grad = adam_optimizers[-1].update(grad)
        grads.append(adam_grad)

        """
        for i, g in enumerate(grads):
            mpo_tensors[i] = mpo_tensors[i] - g
        """
        
        return (grads, unitary_score, c_abs)

    def __init__(self, symmetry_case, num_sites=DEFAULT_NUM_SITES,
        virtual_bond_dim=DEFAULT_VIRTUAL_BOND_DIM,
        unitarity_learning_rate=DEFAULT_UNITARITY_LEARNING_RATE,
        unitarity_shape_parameter=DEFAULT_UNITARITY_SHAPE_PARAMETER,
        overlap_learning_rate=DEFAULT_OVERLAP_LEARNING_RATE,
        adam_params=DEFAULT_ADAM_PARAMS, random_initial_mpo=True,
        left_mpo_tensors=None, right_mpo_tensors=None):
        """
        Parameters
        ----------

        """
        self.symmetry_case = symmetry_case

        self.num_sites=num_sites

        self.virtual_bond_dim = virtual_bond_dim

        right_site_indices = list(range(
            self.symmetry_case.right_symmetry_index + 1,
            self.symmetry_case.right_symmetry_index + 1 + self.num_sites
        ))

        self.right_mps_tensors = [
            self.symmetry_case.psi.get_B(i)
            for k, i in enumerate(right_site_indices)
        ]

        self.right_physical_dims = [
            get_physical_dim(b) for b in self.right_mps_tensors
        ]
        
        self.right_total_dimension = reduce(mul, self.right_physical_dims)

        left_site_indices = list(range(
            self.symmetry_case.left_symmetry_index - 1,
            self.symmetry_case.left_symmetry_index - 1 - self.num_sites,
            -1
        ))

        left_mps_tensors = [
            self.symmetry_case.psi.get_B(i, form='A')
            for k, i in enumerate(left_site_indices)
        ]

        self.left_mps_tensors = [
            swap_left_right_indices(b) for b in left_mps_tensors
        ]

        self.left_physical_dims = [
            get_physical_dim(b) for b in self.left_mps_tensors
        ]

        self.left_total_dimension = reduce(mul, self.left_physical_dims)

        self.virtual_dims = (
            [(None, self.virtual_bond_dim),] +
            [(self.virtual_bond_dim, self.virtual_bond_dim)]*(self.num_sites - 2) +
            [(self.virtual_bond_dim, None),]
        )

        if right_mpo_tensors is not None:
            self.right_mpo_tensors = right_mpo_tensors
        elif random_initial_mpo:
            self.right_mpo_tensors = get_random_mpo_tensors(
                self.right_physical_dims,
                self.virtual_dims
            )
            # Better rescaling? One which gets closest to unitary?
            rescale_mpo_tensors(self.right_mpo_tensors, 1)
        else:
            self.right_mpo_tensors = get_identity_mpo_tensors(
                self.right_physical_dims,
                self.virtual_dims
            )

        if left_mpo_tensors is not None:
            self.left_mpo_tensors = left_mpo_tensors
        elif random_initial_mpo:
            self.left_mpo_tensors = get_random_mpo_tensors(
                self.left_physical_dims,
                self.virtual_dims
            )

            rescale_mpo_tensors(self.left_mpo_tensors, 1)
        else:
            self.left_mpo_tensors = get_identity_mpo_tensors(
                self.left_physical_dims,
                self.virtual_dims
            )

        self.symmetry_transfer_matrix = (
                self.symmetry_case.npc_symmetry_transfer_matrix
        )
        
        self.unitarity_learning_rate = unitarity_learning_rate
        self.unitarity_shape_parameter = unitarity_shape_parameter
        self.overlap_learning_rate = overlap_learning_rate

        self.adam_params = adam_params

        self.left_adam_optimizers = [
            AdamTenpy(*self.adam_params) for _ in range(self.num_sites)
        ]

        self.right_adam_optimizers = [
            AdamTenpy(*self.adam_params) for _ in range(self.num_sites)
        ]

        self.unitarity_scores = list()
        self.mpo_expectations = list()

    def grad_desc_one_step(self):
        # Compute left and right symmetry environments
        # Right symmetry environment for left side first
        self.right_overlap_tensors = partial_mpo_mps_contraction_from_right(
            self.right_mpo_tensors,
            self.right_mps_tensors
        )
        self.right_symmetry_environment = npc.tensordot(
            self.symmetry_transfer_matrix,
            self.right_overlap_tensors[0],
            [['vR', 'vR*'], ['vL', 'vL*']]
        )
        self.right_symmetry_environment = swap_left_right_indices(
            self.right_symmetry_environment
        )

        # Left symmetry environment for right side
        self.left_overlap_tensors = partial_mpo_mps_contraction_from_right(
            self.left_mpo_tensors,
            self.left_mps_tensors
        )
        self.left_symmetry_environment = npc.tensordot(
            self.symmetry_transfer_matrix,
            swap_left_right_indices(self.left_overlap_tensors[0]),
            [['vL', 'vL*'], ['vR', 'vR*']]
        )

        # Get right gradients
        gd_out = MPOGradientDescent.mpo_gradient_descent_one_side_one_iteration(
            self.right_mpo_tensors,
            self.right_mps_tensors,
            self.right_total_dimension,
            self.right_overlap_tensors[1:],
            self.unitarity_learning_rate,
            self.unitarity_shape_parameter,
            self.overlap_learning_rate,
            self.left_symmetry_environment,
            self.right_adam_optimizers
        )

        right_grads, unitarity_score, c_abs = gd_out

        self.unitarity_scores.append(unitarity_score)
        self.mpo_expectations.append(c_abs)

        for i, g in enumerate(right_grads):
            self.right_mpo_tensors[i] = self.right_mpo_tensors[i] - g

        # Get left gradients
        gd_out = MPOGradientDescent.mpo_gradient_descent_one_side_one_iteration(
            self.left_mpo_tensors,
            self.left_mps_tensors,
            self.left_total_dimension,
            self.left_overlap_tensors[1:],
            self.unitarity_learning_rate,
            self.unitarity_shape_parameter,
            self.overlap_learning_rate,
            self.right_symmetry_environment,
            self.left_adam_optimizers
        )
        left_grads, *_ = gd_out

        for i, g in enumerate(left_grads):
            self.left_mpo_tensors[i] = self.left_mpo_tensors[i] - g

    def compute_mpo_expectation(self):
        out = self.symmetry_case.mpo_expectation(
            self.left_mpo_tensors,
            self.right_mpo_tensors
        )
        return out

    def compute_training_losses(self):
        U = np.array(self.unitarity_scores)
        O = np.array(self.mpo_expectations)

        unitarity_losses = (
            self.unitarity_learning_rate
            *np.sqrt(1+self.unitarity_shape_parameter*U)
        )

        self.training_losses = (
            unitarity_losses
            + self.overlap_learning_rate*((O-1)*(np.conj(O)-1))
        )

