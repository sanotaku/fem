from abc import ABC
from abc import abstractmethod

import numpy as np

from fem.model_factor.node import Node2d
from fem.utils import ModelTypeError


class BaseElement(ABC):
    def __init__(self, element_no: int):
        self.element_no = element_no
        self.area = None
        self.b_mat = None
        self.d_mat = None
        self.ke_mat = None

    @property
    @abstractmethod
    def nodes(self):
        raise NotImplementedError()


class ElementTri2d(BaseElement):
    def __init__(self, node_0: Node2d, node_1: Node2d, node_2: Node2d, element_no: int):
        super().__init__(element_no)

        if type(node_0) is not Node2d:
            raise ModelTypeError()
        if type(node_1) is not Node2d:
            raise ModelTypeError()
        if type(node_2) is not Node2d:
            raise ModelTypeError()

        self._node_0 = node_0
        self._node_1 = node_1
        self._node_2 = node_2
        
        self.area = (
            node_0.x * node_1.y - node_0.x * node_2.y + node_1.x * node_2.y - node_1.x * node_0.y + node_2.x * node_0.y - node_2.x * node_1.y) / 2

        coef = 1 / (2 * self.area)

        self.b_mat = np.array([
            [coef * (node_1.y - node_2.y), 0, coef * (node_2.y - node_0.y), 0, coef * (node_0.y - node_0.y), 0],
            [0, coef * (node_2.x - node_1.x), 0, coef * (node_0.x - node_2.x), 0, coef * (node_1.x - node_0.x)],
            [coef * (node_2.x - node_1.x), coef * (node_1.y - node_2.y), coef * (node_0.x - node_2.x), coef * (node_2.y - node_0.y), coef * (node_1.x - node_0.x), coef * (node_0.y - node_1.y)]
        ])

    @property
    def nodes(self):
        return [self._node_0, self._node_1, self._node_2]

