from abc import ABC, abstractmethod
import os

import numpy as np
import pandas as pd

from fem.boundary_condition import ForceCondition2d
from fem.boundary_condition import HoldCondition
from fem.solver_module.construction_module import PlateStrainDMatrix
from fem.model_factor.element import ElementTri2d
from fem.model_factor.node import Node2d
from fem.utils import convert_total_vec_number


class BaseModel(ABC):
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
        self.df_model = None
        self.elements = []
        self.dof_total = None

        self._global_node_num = 0
        self._element_num = 0
        self._dof_node = None
        self._node_tria = None
        self._dof_tria3 = None

        self.k_mat = None
        self.kc_mat = None
        self.force_vector = None
        self.u_vector = None
        self.u_hold_vec = None

        self.result_u_vec = None

    @abstractmethod
    def read_model(self, csv_file_path):
        raise NotImplementedError()

    @abstractmethod
    def set_boundary_condition(self, force_condition, hold_condition):
        raise NotImplementedError()


class ModelTri2d(BaseModel):
    def read_model(self, csv_file_path) -> None:
        if not os.path.exists(csv_file_path):
            raise TypeError()

        self.df_model = pd.read_csv(csv_file_path)

        self._global_node_num = max(max(self.df_model['tri_node_1']),
                                    max(self.df_model['tri_node_2']),
                                    max(self.df_model['tri_node_3'])) + 1
        self._element_num = len(self.df_model['element_no'])

        self._dof_node = 2
        self._node_tria = 3
        self._dof_tria3 = self._node_tria * self._dof_node
        self.dof_total = self._global_node_num * self._dof_node

        for row in self.df_model.itertuples():
            node_0 = Node2d(x=row.tri_node_1_x, y=row.tri_node_1_y, global_node_no=row.tri_node_1)
            node_1 = Node2d(x=row.tri_node_2_x, y=row.tri_node_2_y, global_node_no=row.tri_node_2)
            node_2 = Node2d(x=row.tri_node_3_x, y=row.tri_node_3_y, global_node_no=row.tri_node_3)
            
            element = ElementTri2d(node_0=node_0, node_1=node_1, node_2=node_2, element_no=row.element_no)
            self.elements.append(element)
    
    def set_boundary_condition(self, d_mat: PlateStrainDMatrix, force_condition: ForceCondition2d, hold_condition: HoldCondition):
        if self.df_model is None:
            raise ValueError('df_model is None')

        # Dマトリクスの登録
        for element in self.elements:
            element.d_mat = d_mat

        # 強制荷重の登録
        for element in self.elements:
            for node in element.nodes:
                if node.global_node_no in force_condition.force_condition:
                    node.all_coodinate_force(forces=force_condition.force_condition[node.global_node_no])

        # 拘束条件の登録
        for element in self.elements:
            for node in element.nodes:
                if node.global_node_no in hold_condition.hold_condition:
                    node.all_coodinate_hold(is_hold=hold_condition.hold_condition[node.global_node_no])

        # ======================
        # 境界条件を設定   　  　 =
        # ======================
        self.force_vector = np.zeros(self.dof_total)
        self.u_hold_vec = np.full(self.dof_total, fill_value=False)

        for element in self.elements:
            for node in element.nodes:
                self.u_hold_vec[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=0)] = node.x_hold
                self.u_hold_vec[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=1)] = node.y_hold

        for element in self.elements:
            for node in element.nodes:
                self.force_vector[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=0)] = node.x_force
                self.force_vector[convert_total_vec_number(global_node_no=node.global_node_no, axis_num=1)] = node.y_force
