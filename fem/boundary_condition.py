from typing import List

from fem.utils import ModelTypeError


class HoldCondition:
    def __init__(self) -> None:
        """
        Example...
        hold_condition = {1: True, 5: True}
        """
        self.hold_condition = {}
    
    def set_hold(self, global_node_no: int, is_hold=True):
        if type(is_hold) != bool:
            raise ModelTypeError()
        self.hold_condition[global_node_no] = is_hold


class ForceCondition2d:
    def __init__(self) -> None:
        """
        Example...
        force_condition = {1: [None, -100]}
        """
        self.force_condition = {}
    
    def set_hold(self, global_node_no: int, forces: List):
        if list is not type(forces):
            raise ModelTypeError()
        self.force_condition[global_node_no] = forces
