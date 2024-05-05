import gc
import math
import random
import re
from typing import Optional, Union, List

import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSlot, QEvent, Qt
from PyQt6.QtGui import QPaintEvent, QPainter, QBrush, QColor, QMouseEvent, QFontMetrics, QResizeEvent
from PyQt6.QtWidgets import QWidget, QToolTip, QLabel, QVBoxLayout, QPushButton, QScrollBar, QSlider, QCheckBox, \
    QTextEdit

from src.core.graph_system.qt_widgets import PointGraph3D
from src.core.log_system import print_e, print_traceback


class ClusterModule(QWidget):
    valueChanged = QtCore.pyqtSignal(int)
    onMouseRelease = QtCore.pyqtSignal()
    onMousePress = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(ClusterModule, self).__init__(*args, **kwargs)
        self.resize(500, 500)

        self.left_zone: int = 400

        self.point_size: float = 5.
        self.point_count: int = 0
        self.points: Optional[np.ndarray] = None
        self.sizes: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.px_mode: bool = True

        self.graph_system = PointGraph3D(self)
        self.graph_system.move(self.left_zone, 30)

        self.label_point_size = QLabel("Размер точек:", self)
        self.label_point_size.move(self.left_zone + 10, 10)
        self.label_point_size.adjustSize()

        self.slider_point_size = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_point_size.setMinimum(10)
        self.slider_point_size.setMaximum(800)
        self.slider_point_size.setValue(50)
        self.slider_point_size.move(self.label_point_size.x() + self.label_point_size.width() + 5, 10)
        self.slider_point_size.valueChanged.connect(self.set_point_size)
        
        self.checkbox_px_mode = QCheckBox("PxMode", self)
        self.checkbox_px_mode.setChecked(self.px_mode)
        self.checkbox_px_mode.move(self.slider_point_size.x() + self.slider_point_size.width(), 10)
        self.checkbox_px_mode.stateChanged.connect(self.set_px_mode)

        self.text_field_point_input = QTextEdit(self)
        self.text_field_point_input.move(10, 20)
        self.text_field_point_input.resize(self.left_zone - 20, 150)

        self.button_parse = QPushButton("Парсинг точек", self)
        self.button_parse.move(10, self.text_field_point_input.y() + self.text_field_point_input.height() + 10)
        self.button_parse.clicked.connect(self.parse_input)

        # Task #5 (25)
        self.text_field_point_input.setText("""(20,3,19), (7,18,4), (-5,-5,2), (15,19,20), (11,19,-20), (-3,8,-30), 
        (17,5,13), (6, 15,3), (-8,-3,4), (11,13,18), (18,17,-15), (-4,7,-34), (-6,0,1), (20,10,20), (14,3, 16), 
        (2,16,1), (-9,-4,3), (18,11,11), (11,15,-12), (-1,7,-32), (10,2,18), (7,19,2), (-8,-5,6), (15,14,18), 
        (16,16,-20), (-3,6,-33), (13,6,13), (9,16,4), (2,0,3), (7,2,1)""")

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.graph_system.resize(self.width() - self.left_zone, self.height() - 30)

    @pyqtSlot()
    def parse_input(self) -> None:
        try:
            all_text: str = self.text_field_point_input.toPlainText().replace('\n', '').replace(' ', '')
            assert all_text, "Empty string"
            all_text = re.sub(r'\)(,.?)\(', ')S(', all_text)
            all_text = all_text.replace('(', '').replace(')', '')
            list_of_str = all_text.split('S')
            list_of_points = [str_object.split(',') for str_object in list_of_str]
            points = np.array(list_of_points).astype(float)
            self.clear_point()
            for point in points:
                self.add_point(point)
        except Exception as e:
            print_e()
            print_traceback()

    def add_point(self, point: Union[np.ndarray, List[float]],
                  color: Optional[Union[np.ndarray, List[float]]] = None) -> None:
        if color is None:
            color = [random.random(), random.random(), random.random(), 1.]
        if self.points is None:
            self.points = np.array([point])
            self.colors = np.array([color])
            self.sizes = np.array([[self.point_size]])
        else:
            self.points = np.append(self.points, [point], axis=0)
            self.colors = np.append(self.colors, [color], axis=0)
            self.sizes = np.append(self.sizes, [self.point_size])
        self.update_point_data()

    def clear_point(self) -> None:
        self.points = None
        self.colors = None
        self.sizes = None
        gc.collect()
        self.update_point_data()

    @pyqtSlot(int)
    def set_point_size(self, value: int) -> None:
        if self.px_mode:
            self.point_size = value / 10.0
        else:
            self.point_size = value / 500.0
        if self.sizes is not None:
            self.sizes[:] = self.point_size
        self.update_point_data()

    @pyqtSlot(int)
    def set_px_mode(self, _: int) -> None:
        self.px_mode = self.checkbox_px_mode.isChecked()
        self.set_point_size(self.slider_point_size.value())
        self.update_point_data()

    def update_point_data(self) -> None:
        self.set_scatter_plot_parameters(self.points, self.sizes, self.colors, self.px_mode)

    def set_scatter_plot_parameters(self, pos: Optional[np.ndarray],
                                    size: np.ndarray,
                                    color: np.ndarray,
                                    px_mode: bool) -> None:
        if self.points is not None:
            self.graph_system.scatter_point_item.setData(pos=pos, size=size, color=color, pxMode=px_mode)
        else:
            self.graph_system.scatter_point_item.setData(pos=np.array([[0, 0, 0]]), size=np.array([[0.01]]),
                                                         color=np.array([[0, 0, 0, 0]]))



