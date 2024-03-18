from supporting_functions import SR, smooth_transition, extract_data_from_wave_unit_by_idx, save_wav
from supporting_functions import TimbreFactor
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ratio_scoring import find_next_by_mvn, absolute_scale, score_multiple
import random as rd


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
        self._tbr = _handle_input(
            tbr,
            default=TimbreFactor(
                d_spd=rd.random()*2+3.5,
                od_ev=(rd.random()-0.3)*0.3,
                peaks=[rd.randint(3, 9)],
                sigmas=[rd.random()*2],
                gs_amps=[rd.gauss(0, 2)]
            ),
            is_tbr=True
        )

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
            wv[start:end] = extract_data_from_wave_unit_by_idx(tbr, fx[start:end])

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
    pass
