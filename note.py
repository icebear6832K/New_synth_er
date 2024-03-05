import random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import math as m
import os
from typing import List, Tuple
from int_pair_rank import freq_rsl, score_multiple


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


def _extract_data_from_wave_unit_by_idx(wave_form: np.ndarray, idx: np.ndarray):

    """
    由頻率索引的從單元波形中提取數據。

    :param wave_form: 單元波形的 NumPy 數組。
    :param idx: 頻率索引的 NumPy 數組。
    :return: 提取的數據。
    """

    c = 2 * m.pi / SR  # 常數
    x = np.arange(1, len(wave_form) + 1)
    return np.sum(wave_form * np.sin(c * x * idx[:, None]), axis=1)


def _handle_input(input_val, default, is_tbr=False):

    """
    處理輸入Sound類別初始化方法的參數，確保其格式化正確。

    :param input_val: 輸入的參數值。
    :param default: 預設值，用於輸入為 None 的情況。
    :param is_tbr: 是否為 tbr（音色）參數，用於特殊處理。
    :return: 格式化後的輸入參數。
    """

    # 處理 None 或單一數值
    if input_val is None or isinstance(input_val, (int, float, TimbreFactor)):
        return [(0., input_val if input_val is not None else default),
                (1., input_val if input_val is not None else default)]

    # 處理列表輸入
    if isinstance(input_val, list):
        # 檢查是否為數字或 NumPy 數組列表
        if all(isinstance(x, (int, float, TimbreFactor)) for x in input_val):
            return [(i / (len(input_val) - 1), val) for i, val in enumerate(input_val)]
        # 檢查是否為元組列表
        elif all(isinstance(x, tuple) for x in input_val):
            if not is_tbr and all(len(x) == 2 for x in input_val):
                return input_val
            elif all(isinstance(x[1], TimbreFactor) for x in input_val):
                return input_val

    # 若不符合以上條件，則拋出錯誤
    raise ValueError("輸入值有誤")


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


class Sound(object):

    def __init__(
            self,
            length=1.,
            freq: int | float | List[int | float] | List[Tuple[float, int | float]] = None,
            vol: int | float | List[int | float] | List[Tuple[float, int | float]] = None,
            tbr: TimbreFactor | List[TimbreFactor] | List[Tuple[float, TimbreFactor]] = None,
            attack_amp=1.2,
            attack_time=70.,
            decay_time=50.,
            release_time=1000.,
            starts_at=0.,
            rd_z=0.002,
            fm_changes=None
    ):

        """
        初始化 Sound 對象。

        :param length: 音頻長度，單位為秒。
        :param freq: 頻率的時間序列，可以是單值（持續值）、列表（均分變動值）或元組列表（相對時間節點與值）。
        :param vol: 音量的時間序列，可以是單值、列表或元組列表。
        :param tbr: 音色（timbre）的時間序列，可為TimbreFactor物件或其列表、元組列表。
        """

        # 頻率、音量、音色參數
        self._freq = _handle_input(freq, default=442)
        self._vol = _handle_input(vol, default=30)
        self._tbr = _handle_input(tbr, default=TimbreFactor(d_spd=1, od_ev=0), is_tbr=True)

        # ADSR參數
        self._att_amp = attack_amp
        self._att = attack_time
        self._dcy = decay_time
        self._rsl = release_time

        # 長度、起始點參數
        self._len = round(length * SR)
        self._sta = round(SR * starts_at)
        self._rdz = rd_z

        if fm_changes is None:
            fm_changes = []
        self._fm_changes = fm_changes

    def starts_ends(self):
        return self._sta, self._sta + self._len

    def set_fm_changes(self, amp, freq):
        self._fm_changes.append((amp, freq))

    def _apply_fm_chg(self, freq_arr):

        def mk_a_wv(amp_array, fm_freq, step):
            """
            根据振幅数组和频率生成调制波形。
            amp_array: 振幅的变化数组
            freq: 调制频率
            step: 总步骤数
            """
            times = np.linspace(0, step / SR, step, endpoint=False)
            modulation_wave = np.sin(2 * np.pi * fm_freq * times)
            return np.power(10, amp_array * modulation_wave)

        for amp, freq in self._fm_changes:
            if isinstance(amp, list):
                # 如果amp是列表，使用smooth_transition计算整个振幅变化数组
                fm_amp = smooth_transition(amp, self._len)
            else:
                # 如果amp是单一数值，创建一个全为amp值的数组
                fm_amp = np.full(self._len, amp)

            freq_arr *= mk_a_wv(fm_amp, freq, self._len)

        return freq_arr

    def _frq_to_wv_index(self):

        """ 將頻率轉換為波形索引的NumPy 數組。 """

        # 使用 smooth_transition 函数生成平滑的频率变化数组
        freq_array = smooth_transition(self._freq, self._len, rd_z=self._rdz)
        freq_array = self._apply_fm_chg(freq_array)

        # 計算音頻索引，取整數，確保其值 < SR
        cum_freq_indices = np.round(np.cumsum(freq_array), decimals=1).astype('float64') % SR

        return cum_freq_indices

    def _apply_adsr_envelope(self):

        """
        根据 ADSR 参数生成包络。

        :return: ADSR 包络的 NumPy 数组。
        """

        attack_samples = int(SR * self._att / 1000)
        decay_samples = int(SR * self._dcy / 1000)

        if self._rsl == -1 or SR * (self._att + self._dcy + self._rsl) / 1000 > self._len:
            sustain_samples = 0
            release_samples = self._len - attack_samples - decay_samples
            if release_samples <= 0:
                raise ValueError("長度不足容納 envelope")
        else:
            release_samples = int(SR * self._rsl / 1000)
            sustain_samples = self._len - attack_samples - decay_samples - release_samples

        # 利用log曲線构建 ADSR
        attack_env = (1 - np.logspace(0, 1, attack_samples, base=0.1)) / 0.9 * self._att_amp
        decay_env = (np.logspace(0, 1, decay_samples, base=0.1) - 0.1) / 0.9 * (self._att_amp - 1) + 1
        sustain_env = np.ones(sustain_samples) if sustain_samples > 0 else np.array([])
        release_env = (np.logspace(0, 1, release_samples, base=0.1) - 0.1) / 0.9

        return np.concatenate([attack_env, decay_env, sustain_env, release_env])

    def to_wave(self, step=100):

        """
        根據頻率、音量和音色生成波形。

        :param step: 設定次改變音色的sample間隔，其值越小則音色變動的細緻度越高。
        :return: 無，但會保存生成的波形到檔案。
        """

        fx = self._frq_to_wv_index()  # 計算音頻對單位波形索引

        wv = np.zeros(self._len)  # 初始化波形數組

        ovt_n = round(SR / 2.25 / max(x[1] for x in self._freq))

        # 使用新生成的 tbr 泛音分佈
        tbr_chg = smooth_transition(
            [(t, tbr.to_overture_factors(ovt_n=ovt_n)) for t, tbr in self._tbr],
            k=self._len // step
        )

        # 套用音頻與音色參數
        for i, tbr in enumerate(tbr_chg):
            start = i * step
            end = min(start + step, self._len)
            wv[start:end] = _extract_data_from_wave_unit_by_idx(tbr, fx[start:end])

        # 套用力度參數
        wv *= smooth_transition(self._vol, len(wv), rd_z=self._rdz)

        # 套用ADSR
        wv *= self._apply_adsr_envelope()

        return wv


def output_sound_objs(sound_list: list, f_name='dft'):
    """
    將多個 Sound 物件的波形輸出為單一音頻波形並進行保存。

    :param sound_list: Sound 物件的列表。
    :param f_name: 輸出檔案的名稱。
    :return: 無，但會將合併後的波形保存為檔案。
    """
    end = max(x.starts_ends()[1] for x in sound_list) + 1
    wave = np.zeros(shape=(end,))

    # 計算每個 Sound 物件的波形
    waves = [sound.to_wave() for sound in sound_list]

    # 合併波形
    for i, (note, wave_fragment) in enumerate(zip(sound_list, waves)):
        start, end = note.starts_ends()
        plt.plot(wave_fragment)
        plt.title(f'sound_{i}')
        wave[start:end] += wave_fragment

    plt.plot(wave)
    plt.show()
    save_wav(wave, f_name, in_dir='out')


if __name__ == '__main__':

    go = (95, 132, 263, 441, 665)
    bsm = [-1 + rd.random() for _ in range(10)]
    tsm = [(rd.random()-0.5)*2.3 for _ in range(len(bsm))]
    vol = [[rd.randint(23, 47)] + [rd.random() * 5 for _ in range(len(go) - 1)] for _ in range(len(bsm))]
    rn = [go]
    route = [(), ()]

    for x in range(len(bsm)):
        print(go)
        score_dict = score_multiple(*go)[1]
        pair_find_keys = [kx for kx in score_dict.keys() if kx[0] not in route[-2:] and kx[1] not in route[-3:]]
        go_x = min(pair_find_keys, key=lambda y: score_dict[y])

        route.append(go_x[0])
        route.append(go_x[1])
        b, mt = freq_rsl(*[go[z] for z in go_x], base_mv=bsm[x], top_mv=tsm[x])
        go = sorted([x for k, x in enumerate(go) if k not in go_x] + [b, mt])
        print(score_multiple(*go)[0])
        print(go_x)
        print()
        if x > 0:
            vol[x - 1][go_x[0]] += rd.randint(39, 50) * abs(bsm[x]) + 5
            vol[x - 1][go_x[1]] += rd.randint(39, 50) * abs(tsm[x]) + 2

        vol[x][go_x[0]] += rd.randint(47, 50) * abs(bsm[x]) + 5
        vol[x][go_x[1]] += rd.randint(39, 46) * abs(tsm[x]) + 2
        rn.append(go)
