from abc import ABC
from abc import abstractmethod

import numpy as np

from fem.model_factor.node import Node2d
from fem.model_factor.node import Node3d
from fem.utils import ModelTypeError


class BaseElement(ABC):
    def __init__(self, element_no: int):
        self.element_no = element_no
        self.area = None
        self.b_mat = None
        self.d_mat = None
        self.ke_mat = None

        self.strain_vector = None
        self.stress_vector = None

    @property
    @abstractmethod
    def nodes(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def u(self):
        raise NotImplementedError()


class ElementTri2d(BaseElement):
    def __init__(self, element_no: int, node_0: Node2d, node_1: Node2d, node_2: Node2d):
        super().__init__(element_no)

        if type(node_0) is not Node2d:
            raise ModelTypeError()
        if type(node_1) is not Node2d:
            raise ModelTypeError()
        if type(node_2) is not Node2d:
            raise ModelTypeError()

        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2
        
        self.area = (
            node_0.x * node_1.y - node_0.x * node_2.y + node_1.x * node_2.y - node_1.x * node_0.y + node_2.x * node_0.y - node_2.x * node_1.y) / 2

        coef = 1 / (2 * self.area)

        self.b_mat = np.array([
            [coef * (node_1.y - node_2.y), 0, coef * (node_2.y - node_0.y), 0, coef * (node_0.y - node_1.y), 0],
            [0, coef * (node_2.x - node_1.x), 0, coef * (node_0.x - node_2.x), 0, coef * (node_1.x - node_0.x)],
            [coef * (node_2.x - node_1.x), coef * (node_1.y - node_2.y), coef * (node_0.x - node_2.x), coef * (node_2.y - node_0.y), coef * (node_1.x - node_0.x), coef * (node_0.y - node_1.y)]
        ])

    @property
    def nodes(self):
        return [self.node_0, self.node_1, self.node_2]

    @property
    def u(self):
        if self.node_0.x_u is None:
            raise ValueError('node u is None')

        return np.array([self.node_0.x_u, self.node_0.y_u,
                         self.node_1.x_u, self.node_1.y_u,
                         self.node_2.x_u, self.node_2.y_u])


class ElementSquere3d(BaseElement):
    def __init__(self, element_no: int,
                 node_0: Node3d, node_1: Node3d, node_2: Node3d, node_3: Node3d,
                 node_4: Node3d, node_5: Node3d, node_6: Node3d, node_7: Node3d):
        super().__init__(element_no)
        
        if type(node_0) is not Node3d:
            raise ModelTypeError()
        if type(node_1) is not Node3d:
            raise ModelTypeError()
        if type(node_2) is not Node3d:
            raise ModelTypeError()
        if type(node_3) is not Node3d:
            raise ModelTypeError()
        if type(node_4) is not Node3d:
            raise ModelTypeError()
        if type(node_5) is not Node3d:
            raise ModelTypeError()
        if type(node_6) is not Node3d:
            raise ModelTypeError()
        if type(node_7) is not Node3d:
            raise ModelTypeError()

        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2
        self.node_3 = node_3
        self.node_4 = node_4
        self.node_5 = node_5
        self.node_6 = node_6
        self.node_7 = node_7

        self.area = None
        self.b_mat = None

    @property
    def nodes(self):
        return [self.node_0, self.node_1, self.node_2, self.node_3,
                self.node_4, self.node_5, self.node_6, self.node_7]

    @property
    def u(self):
        if self.node_0.x_u is None:
            raise ValueError('node u is None')

        return np.array([self.node_0.x_u, self.node_0.y_u, self.node_0.z_u,
                         self.node_1.x_u, self.node_1.y_u, self.node_1.z_u,
                         self.node_2.x_u, self.node_2.y_u, self.node_2.z_u,
                         self.node_3.x_u, self.node_3.y_u, self.node_3.z_u,
                         self.node_4.x_u, self.node_4.y_u, self.node_4.z_u,
                         self.node_5.x_u, self.node_5.y_u, self.node_5.z_u,
                         self.node_6.x_u, self.node_6.y_u, self.node_6.z_u,
                         self.node_7.x_u, self.node_7.y_u, self.node_7.z_u,])


