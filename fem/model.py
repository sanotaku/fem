from abc import ABC, abstractmethod
from typing import Dict, Tuple
import math

import numpy as np
import pandas as pd


class Model(object):
    def __init__(self):

        self._elements = []

        self._node_num = None
        self._dof_node = 2
        self._dof_total = None
        self._node_tria = 3
        self._dof_tria3 = None
        self._elemnt_num = None

        self._k_mat = None

    def read_model(self, model_filename: str, young_module: float, poisson_retio: float):

        df_model = pd.read_csv(model_filename)  # モデルデータ

        d_mat = PlateStrainDMatrix(young_module=young_module, poisson_retio=poisson_retio)

        for element_no in range(len(df_model)):
            node_1 = Node(x=df_model['tri_node_1_x'][element_no],
                          y=df_model['tri_node_1_y'][element_no],
                          node_no=df_model['tri_node_1'][element_no])
            node_2 = Node(x=df_model['tri_node_2_x'][element_no],
                          y=df_model['tri_node_2_y'][element_no],
                          node_no=df_model['tri_node_2'][element_no])
            node_3 = Node(x=df_model['tri_node_3_x'][element_no],
                          y=df_model['tri_node_3_y'][element_no],
                          node_no=df_model['tri_node_3'][element_no])
            
            self._elements.append(
                Tri2dElement(node_1=node_1, node_2=node_2, node_3=node_3, d_mat=d_mat)
            )

        self._node_num = max(df_model["tri_node_1"].max(), df_model["tri_node_2"].max(), df_model["tri_node_3"].max()) + 1
        self._dof_total = self._node_num * self._dof_node
        self._dof_tria3 = self._node_tria * self._dof_node
        self._elemnt_num = max(df_model["element_no"])

        self._k_mat = np.zeros((self._dof_total, self._dof_total))  # 全体剛性マトリクスK, 0で初期化

        for element in self._elements:
            for i, row in enumerate(element.ke_mat):
                for j, col in enumerate(row):
                        kmat_x = element.nodes[math.floor(i / 2) + 1]['node_no'] * 2 - (i + 1) % 2 + 1
                        kmat_y = element.nodes[math.floor(j / 2) + 1]['node_no'] * 2 - (j + 1) % 2 + 1
                        self._k_mat[kmat_x][kmat_y] += col

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

class DMatrix(ABC):
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
    def __init__(self, young_module: float, poisson_retio: float):
        self._young_module = young_module
        self._poisson_retio = poisson_retio
        self._thickness = 1

        d_coef = self._young_module / ((1 - 2 * self._poisson_retio) * (1 + self._poisson_retio))

        self._d_mat = d_coef * np.array([
            [1 - self._poisson_retio, self._poisson_retio, 0],    
            [self._poisson_retio, 1 - self._poisson_retio, 0],    
            [0, 0, (1 - 2 * self._poisson_retio) / 2]
        ])

    @property
    def d_mat(self):
        return self._d_mat


class Node(object):
    def __init__(self, x: float, y: float, node_no: int):
        self._x = x
        self._y = y
        self._node_no = node_no

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def node_no(self):
        return self._node_no


class Element(ABC):
    pass


class Tri2dElement(Element):
    def __init__(self, node_1: Node, node_2: Node, node_3: Node, d_mat: DMatrix):
        self._node_1 = node_1
        self._node_2 = node_2
        self._node_3 = node_3
        self._d_mat = d_mat.d_mat

        self._area = (
            node_1.x * node_1.y - node_1.x * node_3.y + node_2.x * node_3.y\
            - node_2.x * node_1.y + node_3.x * node_1.y - node_3.x * node_2.y) / 2

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

    @property
    def b_mat(self) -> np.array:
        return self._b_mat

    @property
    def ke_mat(self) -> np.array:
        return self._ke_mat
