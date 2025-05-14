"""
A collection of utility functions for dealing with tenpy MPOs.

"""
import tenpy.linalg.np_conserved as npc

from SPTOptimization.utils import (
    get_right_identity_environment_from_tp_tensor,
)


def mpo_frobenius_inner_product(mpo1_tensors, mpo2_tensors=None):
    """
    Given two operators M1 and M2 represented as MPOs, calculate
    trace(M1 M2.conj()). If M2 is not given, M2=M1.
    """
    if mpo2_tensors is None:
        mpo2_tensors = mpo1_tensors

    w1 = mpo1_tensors[0]
    w2 = mpo2_tensors[0]

    t = npc.tensordot(w1, w2.conj(), [['p', 'p*'], ['p*', 'p']])

    for w1, w2 in zip(mpo1_tensors[1:], mpo2_tensors[1:]):

        t = npc.tensordot(t, w1, [['vR', ], ['vL']])
        t = npc.tensordot(t, w2.conj(), [['vR*', 'p', 'p*'], ['vL*', 'p*', 'p']])

    return t


def partial_mpo_mpo_contraction_from_right(mpo1_tensors, mpo2_tensors):
    """
    Given two operators M1 and M2 represented as MPOs, calculate
    trace(M1 M2.conj()) but only partially contract tensors and return
    results. Contract from right.

    If the output is "out", out[1] will be all of the tensors contracted
    except the first of mpo1_tensors and mpo2_tensors. out[2] will be the
    same but now the second tensors will not be included in the contraction.

    Each output should have virtual legs 'vL' and 'vL*'.

    len(out) = len(mpo1_tensors) - 1, assuming len(mpo1_tensors) ==
    len(mpo2_tensors).

    If M2 is not given, M2=M1.
    """
    if mpo2_tensors is None:
        mpo2_tensors = mpo1_tensors

    out = list()

    w1 = mpo1_tensors[-1]
    w2 = mpo2_tensors[-1]
    t = npc.tensordot(w1, w2.conj(), [['p', 'p*'], ['p*', 'p']])

    out.append(t)

    for w1, w2 in zip(mpo1_tensors[-2:0:-1], mpo2_tensors[-2:0:-1]):
        t = npc.tensordot(t, w1, [['vL',], ['vR']])
        t = npc.tensordot(
            t,
            w2.conj(),
            [['vL*', 'p', 'p*'], ['vR*', 'p*', 'p']]
        )

        out.append(t)

    return out[::-1]


def partial_order_4_mpo_contraction_from_right(mpo_tensors):
    """
    Similar to partial_mpo_mpo_contraction_from_right but now for the expression
    trace(M M.conj() M M.conj()). This is needed to evaluate a unitarity cost
    function.

    Output is a list, and each element should have legs 'vL', 'vL*', 'vL1',
    'vL1*'.
    """
    out = list()

    w = mpo_tensors[-1]
    t = npc.tensordot(w, w.conj(), [['p',], ['p*',]])
    t.ireplace_labels(['vL', 'vL*'], ['vL1', 'vL1*'])
    t = npc.tensordot(t, w, [['p',], ['p*',]])
    t = npc.tensordot(t, w.conj(), [['p', 'p*'], ['p*', 'p']])
    
    out.append(t)

    for w in mpo_tensors[-2:0:-1]:
        t = npc.tensordot(t, w, [['vL',], ['vR',]])
        t = npc.tensordot(t, w.conj(), [['vL*', 'p'], ['vR*', 'p*']])

        w = w.replace_label('vL', 'vL1')
    
        t = npc.tensordot(t, w, [['vL1', 'p',], ['vR', 'p*']])
        t = npc.tensordot(t, w.conj(), [['vL1*', 'p', 'p*'], ['vR*', 'p*', 'p']])

        out.append(t)

    return out[::-1]


def partial_mpo_mps_contraction_from_right(mpo_tensors, mps_tensors):
    """
    Given an operator M and a state a, one can compute the expectation of
    M on a via a tensor network contraction.

    Here we contract the tensor network from the right, returning the partial
    results along the way.

    If the output is "out", out[1] will be all of the tensors contracted
    except the first of mpo_tensors and mps_tensors. out[2] will be the
    same but now the second tensors will not be included in the contraction.

    Each output should have virtual legs 'vL' and 'vL*'.

    len(out) = len(mpo_tensors) - 1, assuming len(mpo_tensors) ==
    len(mps_tensors). THIS MAY NOT BE TRUE!

    To-do: Could have top and bottom mps? Don't need to be equal.
    """
    out = list()

    t = get_right_identity_environment_from_tp_tensor(mps_tensors[-1])

    out.append(t)

    # First site
    b = mps_tensors[-1]
    w = mpo_tensors[-1]
    
    t = npc.tensordot(t, b, [['vL',], ['vR',]])
    t = npc.tensordot(
        t,
        w.replace_label('vL', 'vLm'),
        [['p',], ['p*',]]
    )
    t = npc.tensordot(t, b.conj(), [['p', 'vL*',], ['p*', 'vR*',]])

    out.append(t)

    # Inner sites
    for w, b in zip(mpo_tensors[-2:0:-1], mps_tensors[-2:0:-1]):
        t = npc.tensordot(t, b, [['vL',], ['vR',]])
        t = npc.tensordot(
            t,
            w.replace_label('vL', 'vLm'),
            [['p', 'vLm'], ['p*', 'vR']]
        )
        t = npc.tensordot(t, b.conj(), [['p', 'vL*',], ['p*', 'vR*',]])
    
        out.append(t)

    # Last site
    b = mps_tensors[0]
    w = mpo_tensors[0]

    t = npc.tensordot(t, b, [['vL',], ['vR',]])
    t = npc.tensordot(
        t,
        w,
        [['p', 'vLm'], ['p*', 'vR']]
    )
    t = npc.tensordot(t, b.conj(), [['p', 'vL*',], ['p*', 'vR*',]])

    out.append(t)

    return out[::-1]


def rescale_mpo_tensors(mpo_tensors, new_norm):
    """
    Rescale mpo_tensors in place so that trace(M M.conj()) (i.e. the square of
    the Frobenius norm) is equal to new_norm.
    """
    num_sites = len(mpo_tensors)

    old_norm = mpo_frobenius_inner_product(mpo_tensors).real
    
    scale_factor = np.power(
        new_norm/old_norm,
        1/(2*num_sites)
    )

    for i in range(num_sites):
        mpo_tensors[i] = scale_factor*mpo_tensors[i]


def generate_random_w_tensor(physical_dim, left_virtual_dim=None,
                             right_virtual_dim=None):
    """
    Generate a random w tensor to fit into an MPO with the specified
    dimensions.

    If either of left_virtual_dim or right_virtual_dim are None, then the
    w tensor will not have that leg.

    Currently the real and imaginary parts of the elements of the w array are
    sampled from a gaussian with zero mean and unit variance.
    """

    if (left_virtual_dim is None) and (right_virtual_dim is None):
        dims = (physical_dim, physical_dim)
    elif (left_virtual_dim is None):
        dims = (physical_dim, physical_dim, right_virtual_dim)
    elif (right_virtual_dim is None):
        dims = (physical_dim, physical_dim, left_virtual_dim)
    else: 
        dims = (
            physical_dim,
            physical_dim,
            left_virtual_dim,
            right_virtual_dim
        )
    
    X1 = rng.normal(size=dims)
    X2 = 1j*rng.normal(size=dims)
    X = X1 + X2

        
    if (left_virtual_dim is None) and (right_virtual_dim is None):
        out = npc.Array.from_ndarray_trivial(X, labels=['p', 'p*'])
    elif right_virtual_dim is None:
        out = npc.Array.from_ndarray_trivial(X, labels=['p', 'p*', 'vL'])
    elif left_virtual_dim is None:
        out = npc.Array.from_ndarray_trivial(X, labels=['p', 'p*', 'vR'])
    else:
        out = npc.Array.from_ndarray_trivial(
            X,
            labels=['p', 'p*', 'vL', 'vR']
        )

    return out


def get_random_mpo_tensors(physical_dims, virtual_dims):
    """
    Generates random w tensors to populate an mpo.
    physical dims should be a list of numbers and virtual dims
    a list of pairs.

    TODO: Add check to virtual dims for consistency.
    """

    w_tensors = [
        generate_random_w_tensor(p_dim, *v_dims)
        for p_dim, v_dims in zip(physical_dims, virtual_dims)
    ]

    return w_tensors


def get_identity_mpo_tensors(physical_dims, virtual_dims):
    """
    Generates w tensors so that the resulting mpo will be the identity
    operator.

    Often physical dims, virtual dims will be the same, so could add
    optional behaviour...
    """

    w_tensors = [
        get_identity_w_tensor(p_dim, *v_dims)
        for p_dim, v_dims in zip(physical_dims, virtual_dims)
    ]

    return w_tensors
