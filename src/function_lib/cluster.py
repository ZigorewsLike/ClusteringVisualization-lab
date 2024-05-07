from typing import Optional

import numpy as np

from src.enums import ClusterizationDataMethod


def euclid_disp(v_x: np.ndarray, e_cl: np.ndarray) -> np.ndarray:
    """
    Вычисление евклидово расстояния с учётом дисперсии.

    :param v_x: Вектор признаков
    :param e_cl: Кластер
    :return: Евклидово расстояние с учётом дисперсии
    """
    # if e_cl.shape[0] == 1:
    #     return np.dot(v_x, e_cl[0])
    # Расчёт вектора отклонений от средних значений кластера
    delta: np.ndarray = v_x - e_cl.mean(axis=0)
    # Диагональная матрица дисперсий
    d_mat: np.ndarray = np.diag(e_cl.std(axis=0))
    # Обратная матрица дисперсий
    try:
        d_mat_inv: np.ndarray = np.linalg.inv(d_mat)
    except Exception as e:
        d_mat_inv = d_mat

    return np.dot(delta @ d_mat_inv, delta.T)


def clusterization_threshold(input_array: np.ndarray,
                             threshold: float,
                             data_method: ClusterizationDataMethod = ClusterizationDataMethod.FORWARD,
                             random_seed: Optional[int] = None) -> np.ndarray:
    """
    Выполнение кластеризации с использованием порогового метода

    :param input_array: Входной массив
    :param threshold: Порог
    :param data_method: Метод пред-обработки данных
    :param random_seed: Seed для случайного перемешивания точек. По умолчанию отключено
    :return: None
    """
    array_size: int = input_array.shape[0]
    cluster: np.ndarray = np.zeros((array_size), dtype=int)  # noqa

    # Предобработка данных
    indexes: np.ndarray = np.arange(array_size)
    if data_method is ClusterizationDataMethod.SHUFFLE:
        np.random.seed(random_seed)
        np.random.shuffle(indexes)
        input_array = input_array[indexes]
    elif data_method is ClusterizationDataMethod.REVERSE:
        indexes = indexes[::-1]
        input_array = input_array[indexes]

    # region Кластеризация
    cluster[0] = 1
    for elem_index, elem_val in enumerate(input_array[1:]):
        elem_val: np.ndarray
        for cluster_index in range(1, cluster.max() + 1):
            cluster_vals = input_array[np.where(cluster == cluster_index)]
            dist = euclid_disp(elem_val, cluster_vals)
            if (np.abs(dist) <= threshold).all():
                cluster[elem_index + 1] = cluster_index
                break
            else:
                continue
        else:
            cluster[elem_index + 1] = cluster.max() + 1
    # endregion

    # Постобработка данных (возвращение нормальных значений для массива)
    if data_method is ClusterizationDataMethod.SHUFFLE:
        reverse_indexes = [ind[1] for ind in sorted(dict(zip(indexes, np.arange(array_size))).items())]
        cluster = cluster[reverse_indexes]
    elif data_method is ClusterizationDataMethod.REVERSE:
        reverse_indexes = np.arange(array_size)[::-1]
        cluster = cluster[reverse_indexes]
    return cluster


if __name__ == '__main__':
    np.random.seed(69)

    x = np.random.rand(20, 3)
    print(clusterization_threshold(x, 0.5))
    print(x)


