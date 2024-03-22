import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import math as m
import os


SR = 44100


def save_wav(x_wav: np.ndarray, file_name: str, in_dir='out'):

    """
    保存波形數據為 WAV 檔案。

    :param x_wav: 波形數據的 NumPy 數組。
    :param file_name: 保存的檔案名稱。
    :param in_dir: 保存檔案的目錄。
    """

    if np.max(np.abs(x_wav)) > 0:
        wav = np.int16(x_wav / np.max(np.abs(x_wav)) * 10000)
        existing_name = os.listdir(in_dir)
        saving_file_name = f'{file_name}.wav'
        r = 1
        while saving_file_name in existing_name:
            saving_file_name = f'{file_name}{r}.wav'
            r += 1
        write(f'{in_dir}/{saving_file_name}', SR, wav)


def generate_interpolated_log_normal_array(final_length, step, log_std):
    """
    生成一個經過線性插值的對數正態分布數組。

    Parameters:
    final_length (int): 最終數組的長度。
    log_mean (float): 對數正態分布的對數均值。
    log_std (float): 對數正態分布的對數標準差。

    Returns:
    numpy.ndarray: 經過線性插值的數組。
    """
    log_mean = 0
    # 計算初始長度
    initial_length = max(1, int(np.round(final_length / step)))

    # 生成原始對數正態分布數組
    log_normal_array = np.random.normal(log_mean, log_std, initial_length)
    array = np.exp(log_normal_array)

    # 新的插值目標點
    old_indices = np.linspace(0, initial_length - 1, initial_length)
    new_indices = np.linspace(0, initial_length - 1, final_length)

    # 線性插值
    interpolated_array = np.interp(new_indices, old_indices, array)

    return interpolated_array


def point_btw(a, b, k, rd_z):

    """
    生成從 a 到 b 的 k 個平滑過渡點。a, b可為整數、浮點數或NumPy數組。

    :param a: 起始點。
    :param b: 結束點。
    :param k: 過渡點的數量。
    :return: 包含從 a 到 b 的過渡點的 NumPy 數組。
    """

    # 生成所有 r 值的數組
    r_values = np.linspace(0, 1, k, dtype='float64')

    # 計算正弦值
    sin_values = np.sin(r_values * m.pi - m.pi / 2)

    # 調整正弦值範圍並應用於差值
    adjusted = (sin_values + 1) / 2

    # 生成點
    points = a + np.outer(adjusted, b - a)

    # 陣列扁平化，如果 a 和 b 是標量（如 int 或 float）
    if isinstance(a, (int, float)):
        points = np.reshape(points, newshape=-1) * generate_interpolated_log_normal_array(k, 1000, rd_z)

    return points


def smooth_transition(nodes, k, rd_z=0.):

    """
    為一系列節點生成平滑過渡。

    :param nodes: 包含節點的列表，每個節點為 (位置, 值) 格式。
    :param k: 整個過渡的樣本總數。
    :return: 包含平滑過渡的 NumPy 數組。
    """

    # 確保輸入是排序好的
    nodes = sorted(nodes, key=lambda x: x[0])
    total_span = nodes[-1][0] - nodes[0][0]

    # 根據 nodes[0][1] 的類型初始化結果數組
    if isinstance(nodes[0][1], np.ndarray):
        # 如果節點值是 NumPy 數組，則 result 為多維數組
        result = np.zeros(shape=(k, ) + nodes[0][1].shape, dtype='float64')
    else:
        # 如果節點值是標量（int 或 float），則 result 為一維數組
        result = np.zeros(k, dtype='float64')

    c = 0  # 用於跟蹤當前位置

    # 遍歷每一對連續節點
    for i in range(len(nodes) - 1):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i + 1]

        # 計算當前跨度和點的數量
        span = x2 - x1
        points_in_span = round(k * (span / total_span))

        # 使用 point_btw 函數計算
        if len(result) >= c + points_in_span:
            result[c:c + points_in_span] = point_btw(y1, y2, points_in_span, rd_z)
            c += points_in_span
        else:
            result[c:] = point_btw(y1, y2, len(result) - c, rd_z)

    return result


def extract_data_from_wave_unit_by_idx(wave_form: np.ndarray, idx: np.ndarray):

    """
    由頻率索引的從單元波形中提取數據。

    :param wave_form: 單元波形的 NumPy 數組。
    :param idx: 頻率索引的 NumPy 數組。
    :return: 提取的數據。
    """

    c = 2 * m.pi / SR  # 常數
    x = np.arange(1, len(wave_form) + 1)
    return np.sum(wave_form * np.sin(c * x * idx[:, None]), axis=1)


class TimbreFactor:

    def __init__(self, d_spd, od_ev, peaks=None, sigmas=None, gs_amps=None):
        self.d_spd = d_spd
        self.od_ev = od_ev
        self.peaks = peaks if peaks is not None else []
        self.sigmas = sigmas if sigmas is not None else []
        self.gs_amps = gs_amps if gs_amps is not None else []

    def to_overture_factors(self, ovt_n):

        def gaussian_hill(j, pk, sgm, amplitude):

            """
            高斯衰減函數，用於定義泛音因子的衰減。

            :param j: 當前泛音的索引。
            :param pk: 泛音峰值位置。
            :param sgm: 高斯函數的標準差，控制衰減的寬度。
            :param amplitude: 峰值的強度（高度）。
            :return: 在索引 j 處的衰減值。
            """

            return (amplitude ** 1 / 2 if amplitude > 0 else -(-amplitude ** 1 / 2)) * np.exp(
                -((j - pk) ** 2) / (2 * sgm ** 2))

        def handle_param_input(param, reference, default):

            """ 處理輸入參數，確保與參考數組長度一致。"""

            if param is None:
                return np.full(len(reference), default)
            elif isinstance(param, list):
                if len(param) != len(reference):
                    raise ValueError("輸入參數的長度必須與peaks長度一致。")
                return np.array(param)
            else:
                return np.full(len(reference), param)

        # 錯誤檢查
        if self.peaks is None:
            if self.sigmas is not None or self.gs_amps is not None:
                raise ValueError("如果沒有輸入 peaks，sigma 和 gs_amps 也應該留空。")
            peaks, sigmas, gs_amps = [], [], []
        else:
            peaks = np.atleast_1d(self.peaks)
            sigmas = handle_param_input(self.sigmas, peaks, default=10)
            gs_amps = handle_param_input(self.gs_amps, peaks, default=10)

        # 初始化泛音數組
        ovt = np.ones(ovt_n, dtype='float64')

        # 奇偶泛音強度調整
        ovt *= 2 ** (self.od_ev * np.where(np.arange(ovt_n) % 2 == 0, 1., -1.))

        # 應用每個 peak 和對應的 sigma 和 gs_amp
        hill_values = np.zeros(ovt_n)
        for peak, sigma, gs_amp in zip(peaks, sigmas, gs_amps):
            hill_values += np.array([gaussian_hill(i, peak, sigma, gs_amp) for i in range(ovt_n)])

        # 將泛音數組乘以高斯曲線
        ovt *= 1.5 ** hill_values

        # 應用衰減速度
        decay_factors = np.linspace(1, 0, ovt_n) ** (2 ** self.d_spd)
        ovt *= decay_factors

        # 正規化泛音因子
        ovt /= ovt.sum()

        return ovt


if __name__ == '__main__':
    pass
