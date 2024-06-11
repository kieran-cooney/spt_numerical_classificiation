from functools import reduce

import numpy as np

from SPTOptimization.utils import (
    get_transfer_matrix_from_unitary,
    get_transfer_matrices_from_unitary_list,
    multiply_transfer_matrices,
    get_left_environment
)

from super_fibonacci import super_fibonacci

np_I = np.array([[1,0],[0,1]])
np_X = np.array([[0,1],[1,0]])
np_Y = np.array([[0,-1j],[1j,0]])
np_Z = np.array([[1,0],[0,-1]])

base_unitaries = np.array([np_I, 1j*np_X, 1j*np_Y, 1j*np_Z])

def s3_to_unitary(p):
    X = p[0]*np_I + 1j*(p[1]*np_X + p[2]*np_Y + p[3]*np_Z)
    return X

class TransferMatrixSearch:
    def __init__(self, psi, symmetry_operations, index=None,
                 max_num_virtual_points=10000, num_search_points=1000):
        self.psi = psi
        self.symmetry_operations = symmetry_operations

        if index is None:
            self.left_symmetry_index = (self.psi.L - len(self.symmetry_operations))//2 
        else:
            self.left_symmetry_index = index
        self.right_symmetry_index = self.left_symmetry_index + len(self.symmetry_operations) - 1

        self.symmetry_transfer_matrices = (
            get_transfer_matrices_from_unitary_list(
                self.psi,
                self.symmetry_operations,
                self.left_symmetry_index
            )
        )

        self.npc_symmetry_transfer_matrix = reduce(
            multiply_transfer_matrices,
            self.symmetry_transfer_matrices
        )

        self.np_symmetry_transfer_matrix = (
            self.npc_symmetry_transfer_matrix
            .combine_legs([['vL', 'vL*'], ['vR', 'vR*']])
            .to_ndarray()
        )

        U, S, Vh = np.linalg.svd(self.np_symmetry_transfer_matrix)
        assert S[0]/np.sum(S) > 0.9999

        self.left_projected_symmetry_state = U[:,0]
        self.right_projected_symmetry_state = Vh[0]
        self.symmetry_transfer_matrix_op_norm = S[0]

        self.right_virtual_points = list()
        self.left_virtual_points = list()
        self.right_s3_points = list()
        self.left_s3_points = list()
        self.right_ovelraps = list()
        self.left_overlaps = list()

        self.right_max_overlap = 0
        self.left_max_overlap = 0
        self.max_overlap = 0

        self.right_max_s3_points = None
        self.left_max_s3_points = None

        self.max_num_virtual_points = max_num_virtual_points
        self.num_search_points = num_search_points

        s3_points = super_fibonacci(2*num_search_points)
        self.s3_search_points = s3_points[s3_points[:, 0] >=  0]
        self.num_s3_search_points = len(self.s3_search_points)

        self.current_right_depth = 0
        self.current_left_depth = 0

    def update_max_overlap(self):
        self.max_overlap = (
            self.symmetry_transfer_matrix_op_norm
            *self.right_max_overlap
            *self.left_max_overlap
        )

    def search_step_right(self):
        previous_depth = self.current_right_depth
        self.current_right_depth += 1

        site_index = self.right_symmetry_index + self.current_right_depth

        bond_dimension = self.psi.chi[site_index]

        right_environment = np.identity(bond_dimension).reshape((bond_dimension**2,))
        
        base_transfer_matrices = np.array([
            get_transfer_matrix_from_unitary(self.psi, u, site_index)
            .combine_legs([['vL', 'vL*'], ['vR', 'vR*']])
            .to_ndarray()
            for u in base_unitaries
        ])

        if self.current_right_depth == 1:
            previous_points = self.right_projected_symmetry_state[np.newaxis, :]
        elif self.current_right_depth > 1:
            previous_points = self.right_virtual_points[previous_depth-1]

        base_vectors = np.matmul(previous_points, base_transfer_matrices)
        base_overlaps = np.dot(base_vectors, right_environment)

        overlaps = np.abs(np.tensordot(self.s3_search_points, base_overlaps, [[1,], [0,]]))
        target_percentile = 100.0*(1.0 - min(1, self.max_num_virtual_points/(overlaps.size)))
        overlap_threshold = np.percentile(overlaps, target_percentile)

        overlaps_filter = (overlaps > overlap_threshold)

        all_next_points = np.tensordot(
            self.s3_search_points,
            base_vectors,
            [[1,], [0,]]
        )

        if self.current_right_depth == 1:
            assert previous_points.shape[0] == 1
            all_next_s3_points = self.s3_search_points[:, np.newaxis, np.newaxis, :]
        elif self.current_right_depth > 1:
            prev_s3_points = self.right_s3_points[previous_depth-1]
            prev_num_s3_points = prev_s3_points.shape[0]

            all_next_s3_points = np.zeros(
                (
                    self.num_s3_search_points,
                    prev_num_s3_points,
                    self.current_right_depth,
                    4
                )
            )

            all_next_s3_points[:, :, :-1, :] = prev_s3_points[np.newaxis, ...]
            all_next_s3_points[:, :, -1, :] = self.s3_search_points[:, np.newaxis, :]

        filtered_next_points = np.reshape(
            all_next_points[overlaps_filter],
            (-1, bond_dimension**2)
        )
        filtered_next_s3_points = np.reshape(
            all_next_s3_points[overlaps_filter],
            (-1, self.current_right_depth, 4)
        )
        filtered_overlaps = overlaps[overlaps_filter].flatten()

        self.right_virtual_points.append(filtered_next_points)
        self.right_s3_points.append(filtered_next_s3_points)
        self.right_ovelraps.append(filtered_overlaps)

        max_overlap = np.max(filtered_overlaps)
        if max_overlap > self.right_max_overlap:
            self.right_max_overlap = max_overlap
            max_arg = np.argmax(filtered_overlaps)
            self.right_max_s3_points = filtered_next_s3_points[max_arg]
            self.update_max_overlap()

    def search_step_left(self):
        previous_depth = self.current_left_depth
        self.current_left_depth += 1

        site_index = self.left_symmetry_index - self.current_left_depth

        bond_dimension = self.psi.chi[site_index]

        left_environment = get_left_environment(self.psi, site_index)
        
        base_transfer_matrices = np.array([
            get_transfer_matrix_from_unitary(self.psi, u, site_index)
            .combine_legs([['vL', 'vL*'], ['vR', 'vR*']])
            .to_ndarray()
            for u in base_unitaries
        ])

        if self.current_left_depth == 1:
            previous_points = self.left_projected_symmetry_state[np.newaxis, :]
        elif self.current_left_depth > 1:
            previous_points = self.left_virtual_points[previous_depth-1]

        base_vectors = np.tensordot(
            previous_points,
            base_transfer_matrices,
            [[-1,], [2,]]
        )
        base_overlaps = np.dot(base_vectors, left_environment)

        overlaps = np.abs(np.tensordot(self.s3_search_points, base_overlaps, [[1,], [1,]]))
        target_percentile = 100.0*(1.0 - min(1, self.max_num_virtual_points/(overlaps.size)))
        overlap_threshold = np.percentile(overlaps, target_percentile)

        overlaps_filter = (overlaps > overlap_threshold)

        all_next_points = np.tensordot(
            self.s3_search_points,
            base_vectors,
            [[1,], [1,]]
        )

        if self.current_left_depth == 1:
            assert previous_points.shape[0] == 1
            all_next_s3_points = self.s3_search_points[:, np.newaxis, np.newaxis, :]
        elif self.current_left_depth > 1:
            prev_s3_points = self.left_s3_points[previous_depth-1]
            prev_num_s3_points = prev_s3_points.shape[0]

            all_next_s3_points = np.zeros(
                (
                    self.num_s3_search_points,
                    prev_num_s3_points,
                    self.current_left_depth,
                    4
                )
            )
            
            all_next_s3_points[:, :, :-1, :] = prev_s3_points[np.newaxis, ...]
            all_next_s3_points[:, :, -1, :] = self.s3_search_points[:, np.newaxis, :]

        filtered_next_points = np.reshape(
            all_next_points[overlaps_filter],
            (-1, bond_dimension**2)
        )
        filtered_next_s3_points = np.reshape(
            all_next_s3_points[overlaps_filter],
            (-1, self.current_left_depth, 4)
        )
        filtered_overlaps = overlaps[overlaps_filter].flatten()

        self.left_virtual_points.append(filtered_next_points)
        self.left_s3_points.append(filtered_next_s3_points)
        self.left_overlaps.append(filtered_overlaps)

        max_overlap = np.max(filtered_overlaps)
        if max_overlap > self.left_max_overlap:
            self.left_max_overlap = max_overlap
            max_arg = np.argmax(filtered_overlaps)
            self.left_max_s3_points = filtered_next_s3_points[max_arg]
            self.update_max_overlap()