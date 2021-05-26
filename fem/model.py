from abc import ABC, abstractmethod
from typing import Dict
import os

import numpy as np
import pandas as pd

from fem.utils import ModelError


class BaseModel:
    def __init__(self):
        """df_model 
        csv_file Tri 2D model Example...
        ---------------------------------------------------------------------------------------------------------------------------
        |element_no|tri_node_1|tri_node_2|tri_node_3|tri_node_1_x|tri_node_1_y|tri_node_2_x|tri_node_2_y|tri_node_3_x|tri_node_3_y|
        ---------------------------------------------------------------------------------------------------------------------------
        |         0|         0|         1|          4|          0|           0|           1|           0|           1|           1|
        |         1|         1|         2|          3|          1|           0|           2|           0|           2|           1|
        |         2|         1|         3|          4|          1|           0|           2|           2|           1|           1|
        |         3|         0|         4|          5|          0|           0|           1|           2|           0|           1|
        ---------------------------------------------------------------------------------------------------------------------------
        """
        self._df_model = None
        self._elements = []
        self._global_node_num = 0
        self._element_num = 0

        self._dof_node = None
        self._dof_total = None
        self._node_tria = None
        self._dof_tria3 = None

        self._stack_data = {}

    @abstractmethod
    def read_model(self, csv_file_path) -> None:
        pass

    @property
    def elements(self):
        return self._elements
    
    @property
    def dof_total(self):
        return self._dof_total

    def stack_data(self, key) -> None:
        return self._stack_data[str(key)]

    def append_stack_data(self, key, data) -> None:
        self._stack_data[str(key)] = data


class ModelTri2d(BaseModel):
    
    def read_model(self, csv_file_path) -> None:
        if not os.path.exists(csv_file_path):
            raise ModelError('csv file is not exists')

        self._df_model = pd.read_csv(csv_file_path)

        self._global_node_num = max(max(self._df_model['tri_node_1']),
                                    max(self._df_model['tri_node_2']),
                                    max(self._df_model['tri_node_3'])) + 1
        self._element_num = len(self._df_model['element_no'])

        self._dof_node = 2
        self._node_tria = 3
        self._dof_tria3 = self._node_tria * self._dof_node
        self._dof_total = self._global_node_num * self._dof_node

        for row in self._df_model.itertuples():
            node_1 = Node2d(x=row.tri_node_1_x, y=row.tri_node_1_y, grobal_node_no=row.tri_node_1)
            node_2 = Node2d(x=row.tri_node_2_x, y=row.tri_node_2_y, grobal_node_no=row.tri_node_2)
            node_3 = Node2d(x=row.tri_node_3_x, y=row.tri_node_3_y, grobal_node_no=row.tri_node_3)
            
            element = ElementTri2d(node_1=node_1, node_2=node_2, node_3=node_3, element_no=row.element_no)
            self._elements.append(element)


class Node2d:
    def __init__(self, x: float, y: float, grobal_node_no: int):
        self._x = x
        self._y = y
        self._grobal_node_no = grobal_node_no

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def grobal_node_no(self) -> int:
        return self._grobal_node_no


class ElementTri2d:
    def __init__(self, node_1: Node2d, node_2: Node2d, node_3: Node2d, element_no: int):
        self._node_1 = node_1
        self._node_2 = node_2
        self._node_3 = node_3
        self._element_no = element_no
        self._stack_data = {}

        self._area = (
            node_1.x * node_2.y - node_1.x * node_3.y + node_2.x * node_3.y - node_2.x * node_1.y + node_3.x * node_1.y - node_3.x * node_2.y) / 2

        coef = 1 / (2 * self._area)

        self._b_mat = np.array([
            [coef * (node_2.y - node_3.y), 0, coef * (node_3.y - node_1.y), 0, coef * (node_1.y - node_2.y), 0],
            [0, coef * (node_3.x - node_2.x), 0, coef * (node_1.x - node_3.x), 0, coef * (node_2.x - node_1.x)],
            [coef * (node_3.x - node_2.x), coef * (node_2.y - node_3.y), coef * (node_1.x - node_3.x), coef * (node_3.y - node_1.y), coef * (node_2.x - node_1.x), coef * (node_1.y - node_2.y)]
        ])

    @property
    def b_mat(self):
        return self._b_mat

    @property
    def area(self):
        return self._area

    @property
    def nodes(self) -> Dict:
        return {
            1: {
                'node_no': self._node_1.grobal_node_no,
                'coordinate': (self._node_1.x, self._node_1.y)},
            2: {
                'node_no': self._node_2.grobal_node_no,
                'coordinate': (self._node_2.x, self._node_2.y)},
            3: {
                'node_no': self._node_3.grobal_node_no,
                'coordinate': (self._node_3.x, self._node_3.y)},
        }

    def stack_data(self, key) -> None:
        return self._stack_data[str(key)]

    def append_stack_data(self, key, data) -> None:
        self._stack_data[str(key)] = data



if __name__ == '__main__':
    model_obj = Model()
    model_obj.read_model(model_filename="test_model.csv", young_module=210000, poisson_retio=0.3)
    print("=====END=====")
