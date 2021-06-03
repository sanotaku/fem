from abc import ABC

import numpy as np

from fem.utils import create_k_mat_idx
from fem.utils import invert_global_and_coodinate


class DMatrix(ABC):
    def __init__(self, young_module: float, poisson_retio: float):
        self.young_module = young_module
        self.poisson_retio = poisson_retio


class PlateStrainDMatrix(DMatrix):
    # 平面歪み仮定のDマトリクス
    def __init__(self, young_module: float, poisson_retio: float, thickness=1):
        super().__init__(young_module=young_module, poisson_retio=poisson_retio)
        self.thickness = thickness

        d_coef = self.young_module / ((1 - 2 * self.poisson_retio) * (1 + self.poisson_retio))

        self.d_mat = d_coef * np.array([
            [1 - self.poisson_retio, self.poisson_retio, 0],    
            [self.poisson_retio, 1 - self.poisson_retio, 0],    
            [0, 0, (1 - 2 * self.poisson_retio) / 2]
        ])


def solve_2d_static(model_obj):
    # ======================
    # 部分剛性マトリクスを生成 =
    # ======================
    for element in model_obj.elements:
        element.ke_mat = element.d_mat.thickness * element.area * element.b_mat.T @ element.d_mat.d_mat.T @ element.b_mat


    # ======================
    # 全体剛性マトリクスを生成 =
    # ======================
    k_mat = np.zeros((model_obj.dof_total, model_obj.dof_total))  # 全体剛性マトリクスK, 0で初期化

    for element in model_obj.elements:
        for row_vals, row_idxs in zip(element.ke_mat, create_k_mat_idx(element=element)):
            for val, idx in zip(row_vals, row_idxs):
                k_mat[idx[0]][idx[1]] += val
    model_obj.k_mat = k_mat


    # =============================
    # 境界条件を元に拡大係数行列を修正 =
    # =============================
    model_obj.kc_mat = model_obj.k_mat.copy()
    model_obj.u_vector = np.zeros(model_obj.dof_total)

    for i, hold in enumerate(model_obj.u_hold_vec):
        if hold:
            for j in range(model_obj.dof_total):
                if i != j:
                    model_obj.force_vector[i] -= model_obj.kc_mat[i, j] * model_obj.u_vector[i]
            model_obj.kc_mat[:, i] = 0  # 列を0に
            model_obj.kc_mat[i, :] = 0  # 行を0に
            model_obj.kc_mat[i, i] = 1  # 対角成分を1に
            model_obj.force_vector[i] = model_obj.u_vector[i]
    

    # ===============
    # 連立方程式を解く =
    # ===============
    model_obj.result_u_vec = np.linalg.solve(model_obj.kc_mat, model_obj.force_vector)


    # ===============
    # 解を各節点に登録 =
    # ===============
    for element in model_obj.elements:
        for node in element.nodes:

            for total_vec_num in range(len(model_obj.result_u_vec)):
                global_node_no, axis_num = invert_global_and_coodinate(total_vec_num)
                if global_node_no == node.global_node_no:
                    if axis_num == 0:
                        node.x_u = model_obj.result_u_vec[total_vec_num]
                    if axis_num == 1:
                        node.y_u = model_obj.result_u_vec[total_vec_num]


    # ===============
    # 歪みの計算      =
    # ===============
    for element in model_obj.elements:
        element.strain_vector = element.b_mat @ element.u


    # ===============
    # 応力の計算      =
    # ===============
    for element in model_obj.elements:
        element.stress_vector = element.d_mat.d_mat @ element.strain_vector
