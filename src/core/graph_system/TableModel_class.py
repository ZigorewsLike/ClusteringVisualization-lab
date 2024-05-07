from PyQt6 import QtCore
from PyQt6.QtCore import Qt


class TableModelNumpy(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModelNumpy, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row(), index.column()])

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]
