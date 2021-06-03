from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseNode(ABC):
    def __init__(self, global_node_no):
        self.global_node_no = global_node_no

    @property
    @abstractmethod
    def coordinate(self):
        raise NotImplementedError()

    @abstractmethod
    def all_coodinate_hold(self):
        raise NotImplementedError()

    @abstractmethod
    def all_coodinate_force(self):
        raise NotImplementedError()


class Node2d(BaseNode):
    def __init__(self, x: float, y: float, global_node_no: int):
        super().__init__(global_node_no)
        self.x = x
        self.y = y
        self.x_hold = False
        self.y_hold = False
        self.x_force = 0
        self.y_force = 0
        self.x_u = None
        self.y_u = None

    @property
    def coordinate(self):
        return np.array([self._x, self._y])
    
    def all_coodinate_hold(self, is_hold=True):
        self.x_hold = is_hold
        self.y_hold = is_hold

    def all_coodinate_force(self, forces: List):
        self.x_force = forces[0]
        self.y_force = forces[1]


class Node3d(BaseNode):
    def __init__(self, x: float, y: float, z: float, global_node_no: int):
        super().__init__(global_node_no)
        self.x = x
        self.y = y
        self.z = z
        self.x_hold = False
        self.y_hold = False
        self.z_hold = False
        self.x_force = 0
        self.y_force = 0
        self.z_force = 0

    @property
    def coordinate(self):
        return np.array([self.x, self.y, self.z])

    def all_coodinate_hold(self, is_hold):
        self.x_hold = is_hold
        self.y_hold = is_hold
        self.z_hold = is_hold

    def all_coodinate_force(self, forces: List):
        self.x_force = forces[0]
        self.y_force = forces[1]
        self.z_force = forces[2]
