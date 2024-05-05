import math
import random

import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSlot, QEvent
from PyQt6.QtGui import QPaintEvent, QPainter, QBrush, QColor, QMouseEvent, QFontMetrics
from PyQt6.QtWidgets import QWidget, QToolTip, QLabel, QVBoxLayout
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from src.core.log_system import print_d


class PointGraph3D(QWidget):
    valueChanged = QtCore.pyqtSignal(int)
    onMouseRelease = QtCore.pyqtSignal()
    onMousePress = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(PointGraph3D, self).__init__(*args, **kwargs)

        self.resize(500, 500)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(10, 10, 10, 255)
        # grid = gl.GLGridItem()
        # view.addItem(grid)

        axis = gl.GLAxisItem()
        self.view.addItem(axis)

        self.scatter_point_item = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter_point_item)

        self.widget_layout = QVBoxLayout(self)
        self.widget_layout.addWidget(self.view)

