import math


class ModelImportError(Exception):
    pass

class ModelTypeError(Exception):
    pass


def convert_total_vec_number(global_node_no: int, axis_num: int, dim=2) -> int:
    """
    Note:
    Convert to total vector factor number from global node no and axis

    Args:
        global_node_no(int) :global node no
        axis_num(int)       :axis num(x -> 0, y -> 1, z -> 2)
        dim(int)            :default is 2(X and Y)
    Return:
        num: total vector factor number 
    """
    if axis_num not in [0, 1, 2]:
        raise ModelTypeError()

    if global_node_no == 0:
        if axis_num == 0:
            return 0
        elif axis_num == 1:
            return 1
        elif axis_num == 2:
            return 2

    num = global_node_no * dim

    if dim == 2:
        if axis_num == 1:
            num += 1

    elif dim == 3:
        if axis_num == 1:
            num +=1
        elif axis_num == 2:
            num += 2

    return num


def invert_global_and_coodinate(total_vec_num: int, dim=2):
    """
    Note:
    Invert to global node no and axis from total vector factor number 

    Args:
        num      : total vector factor number 
        dim(int) : default is 2(X and Y)
    Return:
        global_node_no(int) :global node no
        axis_num(int)       :axis num(x -> 0, y -> 1, z -> 2)
    """
    if dim != 2:
        raise NotImplementedError('3D is not support')
    
    axis_num = None

    if total_vec_num % 2 == 0:
        axis_num = 0  # X
    elif total_vec_num % 2 == 1:
        axis_num = 1  # Y

    global_node_no = math.floor(total_vec_num / dim)

    return global_node_no, axis_num


def create_k_mat_idx(element):
    idx_vector = []
    for node in element.nodes:
        idx_vector.append(convert_total_vec_number(node.global_node_no, axis_num=0))
        idx_vector.append(convert_total_vec_number(node.global_node_no, axis_num=1))
    
    k_mat_idx_matrix = []
    for x_idx in idx_vector:
        idxs = []
        for y_idx in idx_vector:
            idxs.append([x_idx, y_idx])
        k_mat_idx_matrix.append(idxs)

    return k_mat_idx_matrix
