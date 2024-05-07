import gc
import math
import random
import re
from typing import Optional, Union, List, TYPE_CHECKING

import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSlot, QEvent, Qt
from PyQt6.QtGui import QPaintEvent, QPainter, QBrush, QColor, QMouseEvent, QFontMetrics, QResizeEvent, QFont
from PyQt6.QtWidgets import QWidget, QToolTip, QLabel, QVBoxLayout, QPushButton, QScrollBar, QSlider, QCheckBox, \
    QTextEdit, QTableView, QInputDialog, QComboBox, QSpinBox

from src.core.graph_system.qt_widgets import PointGraph3D
from src.core.log_system import print_e, print_traceback
from src.function_lib.cluster import clusterization_threshold
from src.core.graph_system import TableModelNumpy
from src.enums import ClusterizationDataMethod

if TYPE_CHECKING:
    from src.forms.MainForm_class import MainForm


class ClusterModule(QWidget):
    valueChanged = QtCore.pyqtSignal(int)
    onMouseRelease = QtCore.pyqtSignal()
    onMousePress = QtCore.pyqtSignal()

    def __init__(self, mf, *args, **kwargs):
        super(ClusterModule, self).__init__(*args, **kwargs)
        self.resize(500, 500)
        self.mf: MainForm = mf

        self.left_zone: int = 400

        self.point_size: float = self.mf.settings.graph_settings.point_size
        self.point_count: int = 0
        self.points: Optional[np.ndarray] = None
        self.sizes: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.px_mode: bool = self.mf.settings.graph_settings.px_mode

        self.cluster_threshold: float = 5.0

        self.graph_system = PointGraph3D(self)
        self.graph_system.move(self.left_zone, 30)

        self.label_point_size = QLabel("Размер точек:", self)
        self.label_point_size.move(self.left_zone + 10, 10)
        self.label_point_size.adjustSize()

        self.slider_point_size = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_point_size.setMinimum(10)
        self.slider_point_size.setMaximum(1500)
        if self.px_mode:
            self.slider_point_size.setValue(int(self.point_size * 10))
        else:
            self.slider_point_size.setValue(int(self.point_size * 500))
        self.slider_point_size.move(self.label_point_size.x() + self.label_point_size.width() + 5, 10)
        self.slider_point_size.valueChanged.connect(self.set_point_size)
        
        self.checkbox_px_mode = QCheckBox("PxMode", self)
        self.checkbox_px_mode.setChecked(self.px_mode)
        self.checkbox_px_mode.move(self.slider_point_size.x() + self.slider_point_size.width(), 10)
        self.checkbox_px_mode.stateChanged.connect(self.set_px_mode)
        self.set_px_mode(0)

        self.data_title_label = QLabel(" == Данные ==", self)
        self.data_title_label.setFont(QFont('Arial', 16))
        self.data_title_label.move(5, 20)

        self.text_field_point_input = QTextEdit(self)
        self.text_field_point_input.move(10, self.data_title_label.y() + self.data_title_label.height() + 10)
        self.text_field_point_input.resize(self.left_zone - 20, 150)

        self.button_parse = QPushButton("Парсинг точек", self)
        self.button_parse.move(10, self.text_field_point_input.y() + self.text_field_point_input.height() + 10)
        self.button_parse.clicked.connect(self.parse_input)

        self.button_generate = QPushButton("Нагенерировать точек", self)
        self.button_generate.move(self.button_parse.x() + self.button_parse.width() + 20,
                                  self.text_field_point_input.y() + self.text_field_point_input.height() + 10)
        self.button_generate.clicked.connect(self.generate_points)

        self.label_cluster_title = QLabel(" == Кластеризация ==", self)
        self.label_cluster_title.setFont(QFont('Arial', 16))
        self.label_cluster_title.move(5, self.button_parse.y() + self.button_parse.height() + 20)

        self.label_cluster_threshold = QLabel(f"Порог кластеризации ({self.cluster_threshold}): ", self)
        self.label_cluster_threshold.setFont(QFont('Arial', 10))
        self.label_cluster_threshold.move(10, self.label_cluster_title.y() + self.label_cluster_title.height() + 10)
        self.label_cluster_threshold.adjustSize()

        self.slider_cluster_threshold = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_cluster_threshold.move(self.label_cluster_threshold.width() + 20,
                                           self.label_cluster_title.y() + self.label_cluster_title.height() + 10)
        self.slider_cluster_threshold.setFixedWidth(self.left_zone - self.label_cluster_threshold.width() - 20)
        self.slider_cluster_threshold.setMaximum(1000)
        self.slider_cluster_threshold.setMinimum(1)
        self.slider_cluster_threshold.setValue(10)
        self.slider_cluster_threshold.valueChanged.connect(self.set_cluster_threshold)

        self.label_cluster_data_method = QLabel(f"Метод перебора точек: ", self)
        self.label_cluster_data_method.setFont(QFont('Arial', 10))
        self.label_cluster_data_method.move(10,
                                            self.slider_cluster_threshold.y() + self.slider_cluster_threshold.height() + 10)
        self.label_cluster_data_method.adjustSize()

        self.cluster_data_method_dict = {
            ClusterizationDataMethod.FORWARD: "Прямой",
            ClusterizationDataMethod.REVERSE: "Обратный",
            ClusterizationDataMethod.SHUFFLE: "Случайны"
        }
        self.combobox_cluster_data_method = QComboBox(self)
        for value in self.cluster_data_method_dict.values():
            self.combobox_cluster_data_method.addItem(value)
        self.combobox_cluster_data_method.move(self.label_cluster_data_method.width() + 20,
                                               self.slider_cluster_threshold.y() + self.slider_cluster_threshold.height() + 10)

        self.label_cluster_seed = QLabel(f"Seed для случайного перебора точек: ", self)
        self.label_cluster_seed.setFont(QFont('Arial', 10))
        self.label_cluster_seed.move(10,
                                     self.combobox_cluster_data_method.y() + self.combobox_cluster_data_method.height() + 10)
        self.label_cluster_seed.adjustSize()

        self.spinbox_cluster_seed = QSpinBox(self)
        self.spinbox_cluster_seed.setRange(-1, 100_000)
        self.spinbox_cluster_seed.move(self.label_cluster_seed.width() + 20,
                                       self.combobox_cluster_data_method.y() + self.combobox_cluster_data_method.height() + 10)

        self.button_cluster = QPushButton("Выполнить", self)
        self.button_cluster.move(10, self.spinbox_cluster_seed.y() + self.spinbox_cluster_seed.height() + 10)
        self.button_cluster.clicked.connect(self.calc_clusterization)

        self.checkbox_auto_run = QCheckBox("Автовыполнение", self)
        self.checkbox_auto_run.move(self.button_cluster.width() + 20,
                                    self.spinbox_cluster_seed.y() + self.spinbox_cluster_seed.height() + 10)
        self.checkbox_auto_run.setChecked(True)

        self.cluster_table = QTableView(self)
        self.cluster_table.move(10, self.button_cluster.y() + self.button_cluster.height() + 10)
        self.cluster_table.resize(self.left_zone - 10, 65)

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
        self.mf.settings.graph_settings.point_size = self.point_size
        self.mf.save_config_app()

    @pyqtSlot(int)
    def set_px_mode(self, _: int) -> None:
        self.px_mode = self.checkbox_px_mode.isChecked()
        self.set_point_size(self.slider_point_size.value())
        self.update_point_data()
        self.mf.settings.graph_settings.px_mode = self.px_mode
        self.mf.save_config_app()

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

    @pyqtSlot(int)
    def set_cluster_threshold(self, value: int) -> None:
        self.cluster_threshold = value / 10
        self.label_cluster_threshold.setText(f"Порог кластеризации ({self.cluster_threshold}): ")
        self.label_cluster_threshold.adjustSize()
        if self.checkbox_auto_run.isChecked():
            self.calc_clusterization()

    @pyqtSlot()
    def calc_clusterization(self) -> None:
        if self.points is not None:
            random_seed: Optional[int] = self.spinbox_cluster_seed.value()
            data_method: ClusterizationDataMethod = list(self.cluster_data_method_dict.keys())[
                list(self.cluster_data_method_dict.values()).index(self.combobox_cluster_data_method.currentText())
            ]

            clusters = clusterization_threshold(self.points, self.cluster_threshold,
                                                data_method=data_method,
                                                random_seed=random_seed if random_seed != -1 else None)
            np.random.seed(None)
            max_colors = clusters.max()
            colors = np.random.rand(max_colors, 4)
            colors[:, 3] = 1.0
            point_colors = colors[clusters - 1]
            self.colors = point_colors
            self.update_point_data()

            model = TableModelNumpy(clusters.reshape((1, -1)))
            self.cluster_table.setModel(model)
            self.cluster_table.resizeColumnsToContents()

    @pyqtSlot()
    def generate_points(self) -> None:
        try:
            text, ok = QInputDialog.getText(self, 'Генерация точек', 'Сколько точек сгенерировать?')
            if ok:
                self.clear_point()
                point_size = int(text)
                self.points = (np.random.rand(point_size, 3) - 0.5) * 40.0
                self.colors = np.random.rand(point_size, 4)
                self.colors[:, 3] = 1.0
                self.sizes = np.zeros((point_size)) + self.point_size  # noqa
                self.update_point_data()
        except Exception as e:
            print_e(e)


