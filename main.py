"""
Главный файл для запуска приложения
"""
import ctypes
import os
import sys
import traceback
import tracemalloc
from datetime import datetime, timedelta
import pickle

from src.core.log_system import print_e, print_d, OutputBuffer
from src.global_constants import APP_NAME, DEBUG, VERSION, APP_ROAMING_DIR, TRACE, LOG_IN_FILE

from PyQt6.QtCore import QRect, QPoint
from PyQt6.QtGui import QIcon
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QMessageBox
from src.forms import MainForm

import warnings

try:
    from ctypes import windll
    app_id = f'intercoder.labs.cv.{VERSION}'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
except ImportError:
    pass

warnings.filterwarnings('ignore')
if TRACE:
    tracemalloc.start(1)


# Функция перехвата критических ошибок
def except_hook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print_e("Some error:\n", tb, '\033[0m')

    with open('error_log.txt', 'a') as error_f:
        error_f.write(f"\n{datetime.now()} : {tb} \n {'#'*10} \n")

    error_critical_msg = QMessageBox()
    error_critical_msg.setText(f"{tb}")
    error_critical_msg.setIcon(QMessageBox.Icon.Critical)
    error_critical_msg.setWindowTitle(f'Critical Error: {exc_value}.')
    error_critical_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    error_critical_msg.exec()
    if not DEBUG:
        QtWidgets.QApplication.quit()


if __name__ == '__main__':
    # Перехват критических ошибок и stdout
    sys.excepthook = except_hook
    sys.stdout = OutputBuffer()

    # region Создание необходимых для программы директорий и файлов
    for dir_name in ["data/local/"]:
        os.makedirs(dir_name, exist_ok=True)
    os.makedirs(APP_ROAMING_DIR, exist_ok=True)
    if LOG_IN_FILE:
        os.makedirs('logs', exist_ok=True)
    # endregion

    # region Вычисление dpi для больших мониторов или ноутбуков, где масштаб больше 100%
    user32 = ctypes.windll.user32
    w_curr = user32.GetSystemMetrics(0)
    user32.SetProcessDPIAware()
    w_phys = user32.GetSystemMetrics(0)
    curr_dpi = round(w_phys * 96 / w_curr, 0)
    # endregion

    print_d("curr_dpi: ", curr_dpi, w_curr, w_phys)

    # region Инициализация приложения, определение глобальных параметров, стиля
    os.environ["QT_SCALE_FACTOR"] = str(curr_dpi / 96)
    os.environ["QT_FONT_DPI"] = "96"
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon('Icon.ico'))

    app.setStyle("fusion")
    screen = app.primaryScreen()
    size = screen.size()
    # endregion

    # splash_screen = SplashScreen()
    # splash_screen.show()

    # Параметры окна (ширина и высота)
    params_dist: dict = {"size_width": size.width(), "size_height": size.height()}
    # Создание окна авторизации
    mainWin = MainForm(params_dist)
    mainWin.load_ann_models()
    # splash_screen.close()

    mainWin.show()
    # Ожидание завершения приложения
    app.exec()
    # Сброс stdout (метод OutputBuffer класса)
    sys.stdout.reset()
    tracemalloc.stop()
