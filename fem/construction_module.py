from abc import ABC, abstractmethod

import math
import numpy as np

from fem import model
from fem.utils import ModelError


class DMatrix(ABC):
    def __init__(self, young_module: float, poisson_retio: float, thickness=1):
        self._young_module = young_module
        self._poisson_retio = poisson_retio
        self._thickness = thickness

    @property
    def young_module(self) -> float:
        return self._young_module

    @property
    def poisson_retio(self) -> float:
        return self._poisson_retio

    @property
    def thickness(self) -> float:
        return self._thickness
    
    @property
    def d_mat(self) -> np.array:
        return self._d_mat


class PlateStrainDMatrix(DMatrix):
    # 平面歪み仮定のDマトリクス
    def __init__(self, young_module: float, poisson_retio: float, thickness=1):
        super().__init__(young_module=young_module, poisson_retio=poisson_retio, thickness=thickness)

        d_coef = self._young_module / ((1 - 2 * self._poisson_retio) * (1 + self._poisson_retio))

        self._d_mat = d_coef * np.array([
            [1 - self._poisson_retio, self._poisson_retio, 0],    
            [self._poisson_retio, 1 - self._poisson_retio, 0],    
            [0, 0, (1 - 2 * self._poisson_retio) / 2]
        ])


def run_static_calcration(model_obj, d_mat):
    if model_obj is type(model.ModelTri2d):
        raise ModelError('model is not triangle 2D Model')

    if d_mat is type(PlateStrainDMatrix):
        raise ModelError('D-matrix is not PlateStrain D-Matrix')


    # ======================
    # 部分剛性マトリクスを生成 =
    # ======================
    for element in model_obj.elements:
        element.append_stack_data(
            key='ke_mat', data=d_mat.thickness * element.area * element.b_mat.T @ d_mat.d_mat.T @ element.b_mat
        )


    # ======================
    # 全体剛性マトリクスを生成 =
    # ======================
    k_mat = np.zeros((model_obj.dof_total, model_obj.dof_total))  # 全体剛性マトリクスK, 0で初期化

    for element in model_obj.elements:
        for i, row in enumerate(element.stack_data(key='ke_mat')):
            for j, col in enumerate(row):
                    kmat_x = element.nodes[math.floor(i / 2) + 1]['node_no'] * 2 - (i + 1) % 2 + 1
                    kmat_y = element.nodes[math.floor(j / 2) + 1]['node_no'] * 2 - (j + 1) % 2 + 1
                    
                    k_mat[kmat_x][kmat_y] += col

    model_obj.append_stack_data('k_mat', k_mat)

    # # ======================
    # # 境界条件を設定   　  　 =
    # # ======================
    # u_vector = np.zeros(model_obj.dof_total)

    # k_mat_copy = k_mat.copy()

    # for i, hold in enumerate(self._u_hold_vector):
    #     if hold:
    #         for j in range(model_obj.dof_total):
    #             if i != j:
    #                 self._f_vector[i] -= k_mat_copy[i, j] * self._u_vector[i]
    #         k_mat[:, i] = 0  # 列を0に
    #         k_mat[i, :] = 0  # 行を0に
    #         k_mat[i, i] = 1  # 対角成分を1に
    #         self._f_vector[i] = self._u_vector[i]