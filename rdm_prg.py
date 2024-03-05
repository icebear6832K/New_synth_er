import random as rd
import math as m
import matplotlib.pyplot as plt
from note import Sound, output_sound_objs, TimbreFactor

# 定義簡單整數比率
SIMPLE_INT_RATIOS = {
    (x, y): round(y / x, 9)
    for x in range(1, 9)
    for y in range(x, 9) if m.gcd(x, y) == 1
}


def find_closest_simple_ratio(num_list):
    """
    計算給定數字列表中每對數字的比率，並找到最接近的簡單整數比率。

    :param num_list: 代表頻率的數字串列
    :return: 一個字典，包含每對數字與最接近的簡單整數比率的詳細資訊，分別為：
        key: 數字對的索引元組
        [0]: 第一個頻率值
        [1]: 第二個頻率值
        [2]: 兩頻率之比值
        [3], [4]: 比值最接近兩頻率之比的簡單整數（最近簡單整數比）
        [5]: 「兩頻率之比值」與「最近簡單整數比」的比值

    輸入範例: [440, 554.37, 659.25]
    輸出範例: {(0, 1): (440, 554.37, 1.25993, 4, 5, 1.007944), (0, 2): (440, 659.25, 1.4983, 2, 3, 0.9988667), (1, 2): (554.37, 659.25, 1.18919, 5, 6, 0.9909917)}
    """

    num_list = sorted(num_list)
    output_dict = {}

    for i, num in enumerate(num_list[:-1]):
        for j, num2 in enumerate(num_list[i + 1:], start=i + 1):
            ratio = round(num2 / num, 5)
            closest_ratio, closest_key, best_cls = min(
                ((abs(ratio / simple_ratio - 1), key, round(ratio / simple_ratio, 7))
                 for key, simple_ratio in SIMPLE_INT_RATIOS.items()),
                key=lambda x: x[0]
            )
            output_dict[(i, j)] = (num, num2, ratio, *closest_key, best_cls)

    return output_dict


def find_best_adjustment(num_list, val=2 ** (1 / 12)):
    """
    在給定的代表頻率的數字串列中，找到其比率與指定值（預設為一個半音的頻率比率）最接近的一對數字。

    :param num_list: 代表頻率的數字串列
    :param val: 指定的值，預設為音樂中一個半音的頻率比率
    :return: 最接近指定值的數字對的索引，以及其「兩頻率之比值」與「最近簡單整數比」的比值

    輸入範例: [440, 554.37, 659.25]
    輸出範例: (0, 2), 0.9909917
    """
    ratio_dict = find_closest_simple_ratio(num_list)
    best_key = min(
        (key for key in ratio_dict if round(ratio_dict[key][5], 2) != 1.),
        key=lambda key: abs((m.log(ratio_dict[key][5] if ratio_dict[key][5] > 1 else 1 / ratio_dict[key][5])) - m.log(val)),
        default=None
    )

    if best_key is not None:
        return best_key, ratio_dict[best_key][5]

    else:
        return None, None


def apply_adjustment(freq_list, key_to_adjust, ratio_adjustment, arrange_factor):
    """
    根據給定的因子，調整頻率列表中特定一對頻率。

    :param freq_list: 頻率列表
    :param key_to_adjust: 需要調整的數字對的鍵
    :param ratio_adjustment: 調整因子，決定調整的程度
    :param arrange_factor: 平衡因子，決定高低兩頻率分別調整的比例
    :return: 調整前後的頻率值

    輸入範例: ([440, 554.37, 659.25], (0, 2), 1.09238, 0.5)
    輸出範例: (440, 459.87, 659.25, 630.76)
    """
    # 提取需要調整的頻率
    k_index, n_index = key_to_adjust
    k, n = freq_list[k_index], freq_list[n_index]

    # 計算調整後的頻率值
    adjusted_k = round(k * (ratio_adjustment ** (1 - arrange_factor)), 2)
    adjusted_n = round(n / (ratio_adjustment ** arrange_factor), 2)

    # 更新頻率列表
    freq_list[k_index], freq_list[n_index] = adjusted_k, adjusted_n

    return k, adjusted_k, n, adjusted_n


def harmony_factor(freq_list):
    fac = 1
    for each_harm_factor in ((x[1][5] if x[1][5] >= 1 else 1/x[1][5]) for x in find_closest_simple_ratio(freq_list).items()):
        fac *= each_harm_factor
    fac **= 1 / len(freq_list)
    return fac


def adjust_val(val_choice, i, g=1.04):
    if val_choice is None:
        val = rd.gauss(g, 0.02)
    elif isinstance(val_choice, list):
        val = val_choice[i % len(val_choice)]
    else:
        val = val_choice
    return val


def adjust_frequencies(freq_list, val_choice=None, arrage=None, change_limit=50):

    """
    調整頻率列表，使其接近簡單整數比率。這個過程在指定的更改次數內進行。

    :param freq_list: 初始頻率列表
    :param val_choice: 指定頻率移動之程度，尋找與「最近簡單整數比」的比最接近該值的頻率數對
    :param change_limit: 最大更改次數
    :return: 調整後的頻率列表和更改詳情

    輸入範例: [440, 554.37, 659.25]
    輸出範例: (調整後的頻率列表, 更改詳情列表)
    """

    #  存放每次結果的列表
    result_list = [tuple(freq_list)]

    #  存放每次改變的音頻之索引
    changes_idx = []

    #  存放每次的隨機參數
    rd_vals = []

    print(''.join([f'{x:11.2f}' for x in freq_list]), end='      ')
    print(harmony_factor(freq_list), end='    ')

    for i in range(change_limit):

        #  無指定val_choice時，隨機選擇
        val = adjust_val(val_choice, i)
        arr = adjust_val(arrage, i, -0.05)

        #  輸入val，尋找頻率數列中「與最近簡單整數比的比值」最接近val值的頻率數對
        key_to_adjust, ratio_adjustment = find_best_adjustment(freq_list, val=val)

        #  當數列中所有數值間與最近之簡單整數比的比值，四捨五入至小數第二位=1.00時，認定已無可解決之頻率對。
        if key_to_adjust is None:
            break

        if ratio_adjustment > 1.09 or ratio_adjustment < 1/1.09:
            arrange_factor = rd.gauss(0.5, 0.1)
        else:
            arrange_factor = (0.5 + rd.gauss(0.5, 0.1)) % 1

        print(f'平衡值：{arrange_factor:.2f}')

        k, adjusted_k, n, adjusted_n = apply_adjustment(freq_list, key_to_adjust, ratio_adjustment, arrange_factor)

        move_base = rd.random() > 0.7  # 决定是否移动基础频率
        base_ori = None
        base_new = None
        if move_base:
            base_index = rd.randint(0, 1)
            direction = rd.choice([-1, 1])
            freq_adjustment = 2 ** (direction * (rd.random() + 0.5) / 12)
            base_ori = freq_list[base_index]
            freq_list[base_index] *= freq_adjustment
            base_new = freq_list[base_index]
            # 记录移动基础频率的决策

        harm_fac = harmony_factor(freq_list)
        result_list.append(tuple(freq_list))

        print(''.join(['        |  ' if i in key_to_adjust else '           ' for i in range(len(freq_list))]), end='     ')
        print(key_to_adjust)
        print(''.join([f'{x:11.2f}' for x in freq_list]), end='      ')
        print(harm_fac, end='   ')

        changes_idx.append(key_to_adjust)
        rd_vals.append((val, arrange_factor))

    return result_list, changes_idx, rd_vals


def create_composition(freq_tuples, changes_idx, length=80., melody_volume_range=(20, 60), accompaniment_volume=0):
    """
    根据频率元组列表创建并合并Sound对象为一段音频，同时为每个Sound对象加入随机的音量变化。

    :param freq_tuples: 频率元组列表，每个元组代表一瞬间的频率设置。
    :param length: 每个Sound对象的长度，单位为秒。
    :param base_volume: 基础音量。
    :param melody_volume_range: 主旋律音量范围，形式为(最小值, 最大值)。
    :param accompaniment_volume: 伴奏部分的音量。
    """

    num_moments = len(freq_tuples)
    num_sounds = len(freq_tuples[0])

    # 转置freq_tuples以分离每个Sound对象的频率参数
    freq_params = list(zip(*freq_tuples))

    # 准备每个Sound对象的音量变化列表
    volume_changes = [[accompaniment_volume * rd.gauss(1, 0.1) for _ in range(num_moments)] for _ in range(num_sounds)]
    for i, change in enumerate(changes_idx):

        change_1 = change[0]
        change_2 = change[1]
        volume_changes[change_1][i] = rd.randint(*melody_volume_range)
        volume_changes[change_2][i] = rd.randint(*melody_volume_range)

        if i > 0:
            volume_changes[change_1][i - 1] = rd.randint(*melody_volume_range) * rd.gauss(0.15, 0.1)
            volume_changes[change_2][i - 1] = rd.randint(*melody_volume_range) * rd.gauss(0.15, 0.1)
        if i < len(changes_idx) - 1:
            volume_changes[change_1][i + 1] = rd.randint(*melody_volume_range) * rd.gauss(0.7, 0.1)
            volume_changes[change_2][i + 1] = rd.randint(*melody_volume_range) * rd.gauss(0.7, 0.1)
        if i < len(changes_idx) - 2:
            volume_changes[change_1][i + 2] = rd.randint(*melody_volume_range) * rd.gauss(0.5, 0.1)
            volume_changes[change_2][i + 2] = rd.randint(*melody_volume_range) * rd.gauss(0.5, 0.1)

    plt.plot([x for x in zip(*volume_changes)])
    plt.show()
    plt.plot(freq_tuples)
    plt.title('freq_prog')
    plt.show()

    for x in volume_changes:
        for y in x:
            print(f'{round(y, 3):10}', end='')
        print()
    # 创建Sound对象列表，并应用音量变化
    sounds = [Sound(
        length=length,
        freq=list(x for y, x in enumerate(freq_params[i]) if not y % 2),
        vol=list(volume_changes[i]),
        tbr=TimbreFactor(
            d_spd=0.5+rd.random()*3,
            od_ev=(-0.4+rd.random())*2,
            peaks=rd.sample([x for x in range(2, 19)], 3),
            sigmas=[rd.random()*5 for _ in range(3)],
            gs_amps=[(rd.random()-0.3)*20 for _ in range(3)]
        ),
        rd_z=rd.random()*0.005,
        fm_changes=[
            (rd.random()*0.001, rd.randint(20, 50)),
            (sorted([(0, 0), (rd.random(), 0), (rd.random(), 0), (rd.random(), 0), (1, 0)] + [(rd.random(), rd.random()*0.007) for _ in range(5)], key=lambda x: x[0]),
             rd.randint(4, 7))
        ]
    ) for i in range(num_sounds)]

    # 合并音频输出
    output_sound_objs(sounds, "composition.wav")


def find_closest_2_power(a):
    for _ in range(100):
        if a/2 < 1:
            break
        else:
            a /= 2
    return a


if __name__ == '__main__':
    a = [104, 164, 228, 266, 310, 408]
    note, cdx, _ = adjust_frequencies(a, arrage=[-0.01, -0.12, -0.05, -0.07, 0.08, 0.03, -0.38])
    create_composition(note, cdx)
