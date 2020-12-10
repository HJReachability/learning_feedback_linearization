#!/usr/bin/env python

import numpy as np


class PosAttController():
    def __init__(self, pos_controller, att_controller):
        self.pos_controller = pos_controller
        self.att_controller = att_controller
