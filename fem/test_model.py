from abc import ABC, abstractmethod
from typing import Dict
import math

import numpy as np
import pandas as pd


class Model(object):
    def __init__(self):

        self._df_model = None

        self._elements = []

        self._node_num = None
        self._dof_node = 2
        self._dof_total = None
        self._node_tria = 3
        self._dof_tria3 = None
        self._elemnt_num = None

        self._k_mat = None
        self._u_vector = None
        self._u_hold_vector = None
        self._f_vector = None

        self._u_answer = None

    def read_model(self, model_filename: str) -> None:
        self._df_model = pd.read_csv(model_filename)  # モデルデータ
        return

    def read_bound_condition(self, u_hold_vector: np.array, f_vector: np.array, u_vector: np.array):
        self._u_hold_vector = u_hold_vector
        self._f_vector = f_vector
        self._u_vector = u_vector
        return

    def create_model(self, young_module: float, poisson_retio: float):
        d_mat = PlateStrainDMatrix(young_module=young_module, poisson_retio=poisson_retio)

        # ======================
        # エレメントを作成    　  =
        # ======================
        for element_no in range(len(self._df_model)):
            node_1 = Node(x=self._df_model['tri_node_1_x'][element_no],
                          y=self._df_model['tri_node_1_y'][element_no],
                          node_no=self._df_model['tri_node_1'][element_no])
            node_2 = Node(x=self._df_model['tri_node_2_x'][element_no],
                          y=self._df_model['tri_node_2_y'][element_no],
                          node_no=self._df_model['tri_node_2'][element_no])
            node_3 = Node(x=self._df_model['tri_node_3_x'][element_no],
                          y=self._df_model['tri_node_3_y'][element_no],
                          node_no=self._df_model['tri_node_3'][element_no])
            self._elements.append(
                Tri2dElement(node_1=node_1, node_2=node_2, node_3=node_3, d_mat=d_mat)
            )

        self._node_num = max(self._df_model["tri_node_1"].max(),
                             self._df_model["tri_node_2"].max(),
                             self._df_model["tri_node_3"].max()) + 1
        self._dof_total = self._node_num * self._dof_node
        self._dof_tria3 = self._node_tria * self._dof_node
        self._elemnt_num = max(self._df_model["element_no"])


        # ======================
        # 全体剛性マトリクスを生成 =
        # ======================
        self._k_mat = np.zeros((self._dof_total, self._dof_total))  # 全体剛性マトリクスK, 0で初期化

        for no, element in enumerate(self._elements):
            for i, row in enumerate(element.ke_mat):
                for j, col in enumerate(row):
                        kmat_x = element.nodes[math.floor(i / 2) + 1]['node_no'] * 2 - (i + 1) % 2 + 1
                        kmat_y = element.nodes[math.floor(j / 2) + 1]['node_no'] * 2 - (j + 1) % 2 + 1
                        
                        self._k_mat[kmat_x][kmat_y] += col
        

        # ======================
        # 境界条件を設定   　  　 =
        # ======================
        self._u_vector = np.zeros(self._dof_total)

        k_mat_copy = self._k_mat.copy()

        for i, hold in enumerate(self._u_hold_vector):
            if hold:
                for j in range(self._dof_total):
                    if i != j:
                        self._f_vector[i] -= k_mat_copy[i, j] * self._u_vector[i]

                self._k_mat[:, i] = 0  # 列を0に
                self._k_mat[i, :] = 0  # 行を0に
                self._k_mat[i, i] = 1  # 対角成分を1に
                self._f_vector[i] = self._u_vector[i]
        return

    def run(self):
        self._u_answer = np.linalg.solve(self._k_mat, self._f_vector)

    @property
    def model_data(self):
        return {
            'node_num': self._node_num,
            'dof_node': self._dof_node,
            'dof_total': self._dof_total,
            'node_tria': self._node_tria,
            'dof_tria3': self._dof_tria3,
            'element_num': self._elemnt_num
        }

    @property
    def elements(self):
        return self._elements

    @property
    def k_mat(self):
        return self._k_mat
    
    @property
    def f_vector(self):
        return self._f_vector

    @property
    def u_vector(self):
        return self._u_vector


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


class Node(object):
    def __init__(self, x: float, y: float, node_no: int):
        self._x = x
        self._y = y
        self._node_no = node_no

        self._init_force_x = 0
        self._init_force_y = 0
        
        self._const_cond_x = False
        self._const_cond_y = False

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def node_no(self) -> int:
        return self._node_no

    @property
    def init_force_x(self) -> float:
        return self._init_force_x

    @property
    def init_force_y(self) -> float:
        return self._init_force_y

    @property
    def const_cond_x(self) -> bool:
        return self._const_cond_x

    @property
    def const_cond_y(self) -> bool:
        return self._const_cond_y


class Element(ABC):
    def __init__(self, d_mat: DMatrix):
        self._d_mat = d_mat.d_mat

    @property
    @abstractmethod
    def nodes(self) -> Dict:
        pass

    @property
    def b_mat(self) -> np.array:
        return self._b_mat

    @property
    def ke_mat(self) -> np.array:
        return self._ke_mat


class Tri2dElement(Element):
    def __init__(self, d_mat: DMatrix, node_1: Node, node_2: Node, node_3: Node):
        super().__init__(d_mat=d_mat)
        self._node_1 = node_1
        self._node_2 = node_2
        self._node_3 = node_3

        self._area = (
            node_1.x * node_2.y - node_1.x * node_3.y + node_2.x * node_3.y - node_2.x * node_1.y + node_3.x * node_1.y - node_3.x * node_2.y) / 2

        coef = 1 / (2 * self._area)

        self._b_mat = np.array([
            [coef * (node_2.y - node_3.y), 0, coef * (node_3.y - node_1.y), 0, coef * (node_1.y - node_2.y), 0],
            [0, coef * (node_3.x - node_2.x), 0, coef * (node_1.x - node_3.x), 0, coef * (node_2.x - node_1.x)],
            [coef * (node_3.x - node_2.x), coef * (node_2.y - node_3.y), coef * (node_1.x - node_3.x), coef * (node_3.y - node_1.y), coef * (node_2.x - node_1.x), coef * (node_1.y - node_2.y)]
        ])

        self._ke_mat = d_mat.thickness * self._area * self._b_mat.T @ d_mat.d_mat.T @ self._b_mat

    @property
    def nodes(self) -> Dict:
        return {
            1: {
                'node_no': self._node_1.node_no,
                'coordinate': (self._node_1.x, self._node_1.y)},
            2: {
                'node_no': self._node_2.node_no,
                'coordinate': (self._node_2.x, self._node_2.y)},
            3: {
                'node_no': self._node_3.node_no,
                'coordinate': (self._node_3.x, self._node_3.y)},
        }


if __name__ == '__main__':
    model_obj = Model()
    model_obj.read_model(model_filename="test_model.csv", young_module=210000, poisson_retio=0.3)
    print("=====END=====")
