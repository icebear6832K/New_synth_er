from sound_object import Sound, output_sound_objs
from ratio_scoring import absolute_scale, find_next_by_mvn
import math


MN2 = 2 ** (1/12)


def multi_freq_rsl(
        original_freq_list,
        moving_base_index=0,
        ratio_changing_target=MN2,
        ratio_changing_tolerance=1.02,
        expecting_score=60,
        bass_movement=None,
        moving_base_movement=None,
        average_movement=0
):
    """
    :param original_freq_list: 原始頻率列表
    :param moving_base_index: 作為調整基準的頻率索引
    :param ratio_changing_target: （對於每組新舊頻率之）目標改變比例，默認為十二平均律中的半音步進
    :param ratio_changing_tolerance: 改變比例的容忍度
    :param expecting_score: 期望達到的和諧度評分
    :param bass_movement: 低音移動的步數，影響所有頻率
    :param moving_base_movement: 基準頻率的移動步數，不改變其他頻率相對於基準的比例
    :param average_movement: 平均移動的步數，以保持整體和諧度
    :return:
    """
    rtm_dict = dict()  # 儲存調整後的比例

    # 計算每個頻率與調整基準的頻率(moving_base)之間原始比例(o_rt)，調整後的比例(n_rt)，以及實際改變量(n_rt / o_rt)
    for i in range(len(original_freq_list)):
        if i == moving_base_index:
            continue
        else:
            o_rt = absolute_scale(original_freq_list[i], original_freq_list[moving_base_index])
            n_rt, sc = find_next_by_mvn(o_rt, ratio_changing_target, ratio_changing_tolerance, expecting_score)
            print(o_rt, n_rt, sc)
            rtm_dict[(moving_base_index, i)] = n_rt / o_rt

    print(rtm_dict)
    fct = [1 for _ in original_freq_list]
    for key in rtm_dict:
        key_0 = min(key)
        key_1 = max(key)
        for i in range(len(original_freq_list)):
            if i == key_0:
                pass
            elif i == key_1:
                fct[i] *= rtm_dict[key]
            else:
                fct[i] *= 1 if key_0 == moving_base_index else rtm_dict[key]
    print(fct)
    if bass_movement is not None:
        return [original_freq_list[i] * fct[i] * (MN2 ** bass_movement) for i in range(len(original_freq_list))]

    if moving_base_movement is not None:
        return [original_freq_list[i] * fct[i] * (MN2 ** moving_base_movement) / (fct[moving_base_index]) for i in range(len(original_freq_list))]

    if average_movement is not None:
        avg = 1
        for f in fct:
            avg *= f
        avg **= 1/len(original_freq_list)
        return [original_freq_list[i] * fct[i] / avg * (MN2 ** average_movement) for i in range(len(original_freq_list))]


a = [294, 487, 551, 980, 1222]
b = multi_freq_rsl([294, 487, 551, 980, 1222], average_movement=1)
print([(i, (math.log(a[i], 2) - math.log(b[i], 2)) * 12) for i in range(len(a))])
