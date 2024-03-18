import math
import matplotlib.pyplot as plt

D = 8

# 生成基於1到D之間的奇數的互質數對比例，並四捨五入至10位小數。這是建立和諧比例基礎的第一步。
SIMPLE_INT_RATIOS = {
    (x, y): round(y / x, 10)
    for x in range(1, D)
    for y in range(x + 1, D) if math.gcd(x, y) == 1 and x % 2 == 1 and y % 2 == 1
}

# 根據整數對的和以及當和相同時較大數的大小進行排序，以確保比較「輕」的比例優先。
key_rank_1 = sorted(SIMPLE_INT_RATIOS.keys(), key=lambda x: (x[0] + x[1], x[1]))


# 簡化整數對，用於連續乘2的過程中保持數對的互質性。
def simplify_pair(x, y, power):
    factor = 2 ** power
    gcd_value = math.gcd(x * factor, y)
    new_x = x * factor // gcd_value
    new_y = y // gcd_value
    return min(new_x, new_y), max(new_x, new_y)


# 對每個基本比例應用倍數過程，生成新的比例對並計算其得分。
sc_score_dict = {}
for i, key in enumerate(key_rank_1):
    shift = sum(key)  # 整數對的和，用於調整得分以反映和諧度。
    simplified_pair = key
    j = 0
    while simplified_pair[1] / simplified_pair[0] <= 16:
        sc_score_dict[simplified_pair] = (j + shift) * 10 + i  # 計算得分，考慮位移和排序位置。
        simplified_pair = simplify_pair(key[0], key[1], j)  # 應用倍數並重新排序。
        j += 1

# 初始化最終得分字典，預設特定比例為最高分。
SC_RANK_DICT = {(1, 2**i): 100 for i in range(5)}  # 對於(1, 1), (1, 2), (1, 4)等，賦予最高分100分。

# 根據前面計算的得分進行排序並分配最終得分。
for i, pair in enumerate(sorted(sc_score_dict.keys(), key=lambda x: sc_score_dict[x], reverse=False)):
    SC_RANK_DICT[pair] = 100 - math.log(i + 2) * 25  # 使用對數函數調整分數以反映排名差異。


# 計算兩個數的絕對比例。
def absolute_scale(a, b):
    return max(a / b, b / a)


# 評分函數，評估給定比例的和諧度。
def score(ratio: float | int):

    while ratio >= 4:
        ratio /= 2

    a, b = 10000, round(10000 * ratio)
    a, b = a // math.gcd(a, b), b // math.gcd(a, b)

    if (a, b) in SC_RANK_DICT.keys():
        return SC_RANK_DICT[(a, b)]
    else:
        # 如果直接比例未找到，則尋找最接近的比例並根據接近程度調整得分。
        p_cls = min(
            SC_RANK_DICT.keys(),
            key=lambda x: absolute_scale(x[1] / x[0], b / a)
        )
        # 考慮約化程度進行扣分。
        return SC_RANK_DICT[p_cls] * 1000 ** (-abs(math.log(p_cls[1] / p_cls[0] / b * a, 1.07)))


def score_multiple(*args):
    num_list = list(args)
    score_dict = dict()
    tt_score = 1
    count = 0
    for i_x in range(len(num_list) - 1):
        for j_x in range(i_x + 1, len(num_list)):
            score_ij = score(absolute_scale(num_list[j_x], num_list[i_x]))
            tt_score *= score_ij
            count += 1
            score_dict[(i_x, j_x)] = score_ij
    tt_score **= 1/count
    return round(tt_score, 5), score_dict


def max_score_in_range(start, end, step=0.001):
    """
    找出指定區間內（含首尾）和諧度評分的最大值。
    :param start: 起始比值
    :param end: 終止比值
    :param step: 檢驗密度
    :return: (最高評分, 最高評分之比值)

    e.g. (1.478, 1.543, 0.001) -> (75, 1.500)
    """

    max_score = -float('inf')  # Initialize with negative infinity for max score
    max_score_position = None  # Initialize with None for position of max score

    current_position = start  # Start from the beginning of the range
    while current_position <= end:
        current_score = score(current_position)  # Calculate score for the current position
        if current_score > max_score:
            max_score = current_score  # Update max score if the current score is greater
            max_score_position = current_position  # Update position of the max score

        current_position += step  # Move to the next position in the range

    return max_score, round(max_score_position, 5)


def min_score_in_range(start, end, step=0.001):
    """
    找出指定區間內（含首尾）和諧度評分的最小值。
    :param start: 起始比值
    :param end: 終止比值
    :param step: 檢驗密度
    :return: (最低評分, 最低評分之比值)
    """
    min_score = float('inf')  # Initialize with negative infinity for max score
    min_score_position = None  # Initialize with None for position of max score

    current_position = start  # Start from the beginning of the range
    while current_position <= end:
        current_score = score(current_position)  # Calculate score for the current position
        if current_score < min_score:
            min_score = current_score  # Update max score if the current score is greater
            min_score_position = current_position  # Update position of the max score

        current_position += step  # Move to the next position in the range

    return min_score, round(min_score_position, 5)


def find_intervals_for_score_range(score_min, score_max=100, start=1., end=2., step=0.001):
    """
    輸入指定的和諧度評分範圍（含首尾）和指定的頻率比例範圍（含首尾）；
    依據輸入之檢驗密度遍歷這段頻率比例範圍內符合指定評分的點，並計算其區間。
    輸出每個區間之起始位置([0], [1])、最大和諧度及其位置([2], [3])、最小和諧度及其位置([4], [5])。
    :param score_min: 指定和諧度評分最小值
    :param score_max: 指定和諧度評分最大值
    :param start: 指定頻率比例最小值
    :param end: 指定頻率比例最大值
    :param step: 檢驗密度
    :return: [(區間起始位置, 區間結束位置, 區間內最大和諧度, 區間內最大和諧度位置, 區間內最小和諧度, 區間內最小和諧度位置), ...]

    e.g. (60, 100, 1, 2, 0.001)
        -> [
        (1, 1.003, 100, 1, 73.6509654952328, 1.003),
        (1.5, 1.5, 72.53469278329726, 1.5, 72.53469278329726, 1.5),
        (1.994, 1.999, 95.0220380215783, 1.999, 73.58332029031524, 1.994)
        ]
    """
    intervals = []
    interval_start = None
    in_interval = False
    current_position = start

    while current_position <= end:
        current_score = score(current_position)

        if score_min <= current_score <= score_max and end - current_position >= step:
            if not in_interval:
                in_interval = True
                interval_start = current_position
        else:
            if in_interval:
                in_interval = False
                interval_end = current_position - step
                intervals.append(
                    (round(interval_start, 5),
                     round(interval_end, 5),
                     *max_score_in_range(interval_start, interval_end, step),
                     *min_score_in_range(interval_start, interval_end, step))
                )

        current_position += step

    return intervals


def handle_target_score(target_score, target_score_tolerance):
    if 0 <= target_score_tolerance < 0.5:
        return target_score - 0.5, target_score + 0.5
    elif target_score_tolerance == -1:
        return target_score, 100
    elif target_score_tolerance == -2:
        return 0, target_score
    else:
        return target_score - target_score_tolerance, target_score + target_score_tolerance


def find_next_by_mvn_n_rsl(
        ratio, target_score=70,
        target_score_tolerance=30,
        ratio_moving_ratio=None,
        ratio_moving_tolerance=1.2,
):
    """
    輸入原頻率比、指定評分和變動率，尋找新的頻率比。
    輸出一個最符合預期且在變動率偏差範圍內的新頻率比。
    :param ratio: 原頻率比
    :param target_score: 目標之和諧度評分
    :param target_score_tolerance: 目標和諧度評分偏差範圍
    :param ratio_moving_ratio: 預期之新舊頻率比的比例
    :param ratio_moving_tolerance: 新舊頻率比的預期偏差範圍
    :return:
    """
    if ratio_moving_ratio is None:
        ratio_moving_ratio = 1
        ratio_moving_tolerance = 1.5

    target_score_lowest, target_score_highest = handle_target_score(target_score, target_score_tolerance)
    expected_ratio = ratio * ratio_moving_ratio
    closest = 999.
    running_code = 0

    while running_code < 10:
        rst = find_intervals_for_score_range(
            target_score_lowest,
            target_score_highest,
            start=ratio * ratio_moving_ratio / ratio_moving_tolerance,
            end=ratio * ratio_moving_ratio * ratio_moving_tolerance
        )
        if len(rst) > 0:
            for dmn in rst:
                if dmn[0] <= expected_ratio <= dmn[1]:
                    return expected_ratio, score(expected_ratio), running_code
                else:
                    closer_bound = min(dmn[:2], key=lambda x: absolute_scale(expected_ratio, x))
                    if absolute_scale(expected_ratio, closer_bound) < absolute_scale(expected_ratio, closest):
                        closest = closer_bound
            return closest, score(closest), running_code + 1
        else:
            target_score_lowest -= 5
            target_score_highest += 5
            ratio_moving_tolerance *= 1.1
            running_code += 1
    print("指定之和諧度或比例移動量過於不切實際")
    return None


def find_next_by_rsl(
        ratio: float, target_score: int | float = 70,
        target_score_tolerance: int | float = -1,
        ratio_moving_ratio: float = 1.2,
):
    """
    輸入原頻率比、指定評分和變動率，尋找新的頻率比。
    輸出一個最符合預期且在變動率偏差範圍內的新頻率比。
    :param ratio: 原頻率比
    :param target_score: 目標之和諧度評分
    :param target_score_tolerance: 目標和諧度評分偏差範圍
    :param ratio_moving_ratio: 預期之新舊頻率比的比例
    :return:
    """

    target_score_lowest, target_score_highest = handle_target_score(target_score, target_score_tolerance)
    expected_ratio = ratio * ratio_moving_ratio
    closest = 999.

    rst = find_intervals_for_score_range(
        target_score_lowest,
        target_score_highest,
        start=ratio * ratio_moving_ratio / 2,
        end=ratio * ratio_moving_ratio * 2
    )

    if len(rst) > 0:
        for dmn in rst:
            if dmn[0] <= expected_ratio <= dmn[1]:
                return expected_ratio, score(expected_ratio)
            else:
                closer_bound = min(dmn[:2], key=lambda x: absolute_scale(expected_ratio, x))
                if absolute_scale(expected_ratio, closer_bound) < absolute_scale(expected_ratio, closest):
                    closest = closer_bound
        return closest, score(closest)

    else:
        print("指定之比例移動量過於不切實際")
        return None


def find_next_by_mvn(
        ratio: float, target_mvn: float = 1.2,
        target_mvn_tolerance: float | int = -1,
        target_score: float | int = 60
):
    if 1 <= target_mvn_tolerance <= 1.01:
        starting_ratio, ending_ratio = ratio * target_mvn / 1.01, ratio * target_mvn * 1.01
    elif target_mvn_tolerance == -1:
        starting_ratio, ending_ratio = sorted(ratio * x for x in [target_mvn, target_mvn * (2 if target_mvn >= 1 else 1 / 2)])
    elif target_mvn_tolerance == -2:
        starting_ratio, ending_ratio = sorted(ratio * x for x in [target_mvn, 1])
    else:
        starting_ratio, ending_ratio = ratio * target_mvn / target_mvn_tolerance, ratio * target_mvn * target_mvn_tolerance
    max_score, max_score_ratio = max_score_in_range(starting_ratio, ending_ratio)
    min_score, min_score_ratio = min_score_in_range(starting_ratio, ending_ratio)
    if max_score >= target_score >= min_score:
        rst = find_intervals_for_score_range(
            target_score-5,
            target_score+5,
            start=starting_ratio,
            end=ending_ratio
        )
        if target_mvn_tolerance == -1:
            rst = rst[0] if target_mvn >= 1 else rst[-1]
        elif target_mvn_tolerance == -2:
            rst = rst[0] if target_mvn < 1 else rst[-1]
        else:
            rst = rst[len(rst) // 2]

        rst_ratio = (rst[0] + rst[1]) / 2
        return rst_ratio, score(rst_ratio)
    elif max_score < target_score:
        return max_score_ratio, max_score
    elif min_score > target_score:
        return min_score_ratio, min_score
    else:
        raise ValueError("something wrong.........")


def freq_rsl(a, b, base_mv=0., top_mv=0., ratio_chg=None, adjusting=0):

    if (base_mv, top_mv) == (0, 0) and ratio_chg is None:
        base_mv = 1
        top_mv = -1
    elif (base_mv, top_mv) != (0, 0):
        pass
    else:
        base_mv = math.log(ratio_chg, 2) * -6
        top_mv = math.log(ratio_chg, 2) * 6

    ratio_chg = 2 ** (1 / 12 * top_mv - 1 / 12 * base_mv)
    exp_new_rt = b / a * ratio_chg
    new_rt = find_next_by_rsl(
        b/a,
        ratio_moving_ratio=ratio_chg,
    )[1]

    if base_mv == 0.:
        b_2 = b * 2 ** (1 / 12 * top_mv) * (new_rt / exp_new_rt)
        a_2 = b_2 / new_rt

    else:
        a_2 = a * 2 ** (1 / 12 * base_mv) * (exp_new_rt / new_rt)
        b_2 = a_2 * new_rt

    return round(a_2, 5), round(b_2, 5)


if __name__ == '__main__':

    frq_1 = 430
    frq_2 = 598
    ratio_o = frq_2/frq_1
    ratio, _ = find_next_by_mvn(ratio_o, 0.88, -2, 100)
    blc = 0  # blc=0時，上下移動量相同、方向相反；blc=1時僅上聲部移動；blc=-1時僅下聲部移動。blc>1時或blc<-1表示兩聲部往同方向移動
    frq_2_new = frq_2 * ((ratio/ratio_o) ** (blc / 2 + 0.5))
    frq_1_new = frq_1 * ((ratio/ratio_o) ** (blc / 2 - 0.5))
    print(frq_1_new)
    print(frq_2_new)
