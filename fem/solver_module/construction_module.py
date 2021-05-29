from abc import ABC

import numpy as np

from fem.utils import convert_total_vec_number
from fem.utils import create_k_mat_idx


class DMatrix(ABC):
    def __init__(self, young_module: float, poisson_retio: float):
        self._young_module = young_module
        self._poisson_retio = poisson_retio

    @property
    def young_module(self) -> float:
        return self._young_module

    @property
    def poisson_retio(self) -> float:
        return self._poisson_retio
    
    @property
    def d_mat(self) -> np.array:
        return self._d_mat


class PlateStrainDMatrix(DMatrix):
    # 平面歪み仮定のDマトリクス
    def __init__(self, young_module: float, poisson_retio: float, thickness=1):
        super().__init__(young_module=young_module, poisson_retio=poisson_retio)
        self._thickness = thickness

        d_coef = self._young_module / ((1 - 2 * self._poisson_retio) * (1 + self._poisson_retio))

        self._d_mat = d_coef * np.array([
            [1 - self._poisson_retio, self._poisson_retio, 0],    
            [self._poisson_retio, 1 - self._poisson_retio, 0],    
            [0, 0, (1 - 2 * self._poisson_retio) / 2]
        ])

    @property
    def thickness(self) -> float:
        return self._thickness


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


    # ======================
    # 境界条件を設定   　  　 =
    # ======================
    model_obj.force_vector = np.zeros(model_obj.dof_total)
    model_obj.u_vector = np.zeros(model_obj.dof_total)
    model_obj.u_hold_vec = np.full(model_obj.dof_total, fill_value=False)

    for element in model_obj.elements:
        for node in element.nodes:
            model_obj.u_hold_vec[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=0)] = node.x_hold
            model_obj.u_hold_vec[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=1)] = node.y_hold
            
    for element in model_obj.elements:
        for node in element.nodes:
            model_obj.force_vector[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=0)] = node.x_force
            model_obj.force_vector[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=1)] = node.y_force


    # =============================
    # 境界条件を元に拡大係数行列を修正 =
    # =============================
    model_obj.kc_mat = k_mat.copy()

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
