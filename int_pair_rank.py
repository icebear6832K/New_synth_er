import math
import matplotlib.pyplot as plt

D = 8

SIMPLE_INT_RATIOS = {
    (x, y): round(y / x, 10)
    for x in range(1, D)
    for y in range(x + 1, D) if math.gcd(x, y) == 1 and x % 2 and y % 2
}


key_rank_1 = sorted(SIMPLE_INT_RATIOS.keys(), key=lambda x: (x[1] + x[0], x[1]))


def simplify_pair(x, y, power):
    factor = 2 ** power
    gcd_value = math.gcd(x * factor, y)
    new_x = x * factor // gcd_value
    new_y = y // gcd_value
    return min(new_x, new_y), max(new_x, new_y)


sc_score_dict = {}
for i, key in enumerate(key_rank_1):
    shift = sum(key)
    simplified_pair = key
    j = 0
    while simplified_pair[1] / simplified_pair[0] <= 16:
        sc_score_dict[simplified_pair] = (j + shift) * 10 + i
        simplified_pair = simplify_pair(key[0], key[1], j)
        j += 1


SC_RANK_DICT = dict()

for i in range(5):
    SC_RANK_DICT[(1, 2**i)] = 100

for i, pair in enumerate(sorted(sc_score_dict.keys(), key=lambda x: sc_score_dict[x], reverse=False)):
    SC_RANK_DICT[pair] = 100 - math.log(i + 2) * 25


def score(ratio: float | int):

    if 0 < ratio < 1:
        ratio = 1/ratio

    while ratio >= 9:
        ratio /= 2

    a, b = round(10000), round(10000*ratio)
    a, b = a // math.gcd(a, b), b // math.gcd(a, b)

    if (a, b) in SC_RANK_DICT.keys():
        return SC_RANK_DICT[(a, b)]

    else:
        p_cls = min(
            SC_RANK_DICT.keys(),
            key=lambda x: (x[1]/x[0])/(b/a) if (x[1]/x[0])/(b/a) >= 1 else (b/a)/(x[1]/x[0])
        )
        return SC_RANK_DICT[p_cls] * 1000 ** (-abs(math.log(p_cls[1]/p_cls[0]/b*a, 1.07)))


def score_multiple(*args):
    num_list = list(args)
    score_dict = dict()
    tt_score = 1
    count = 0
    for i_x in range(len(num_list) - 1):
        for j_x in range(i_x + 1, len(num_list)):
            score_ij = score(num_list[j_x] / num_list[i_x])
            tt_score *= score_ij
            count += 1
            score_dict[(i_x, j_x)] = score_ij
    tt_score **= 1/count
    return round(tt_score, 5), score_dict


def max_score_in_range(start, end, step=0.001):

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

    intervals = []  # Initialize the list to store intervals
    interval_start = None  # Initialize start of the current interval
    in_interval = False  # Flag to track if currently in an interval

    current_position = start
    max_score = 0
    max_score_position = 0
    while current_position <= end:
        current_score = score(current_position)  # Calculate score for the current position
        # Check if the current score is within the desired range
        if score_min <= current_score <= score_max:
            if not in_interval:  # If not already in an interval, start a new one
                in_interval = True
                interval_start = current_position
                max_score = 0
                max_score_position = 0
            if current_score > max_score:
                max_score = current_score
                max_score_position = current_position
        else:
            if in_interval:  # If exiting an interval, end it and add to the list
                in_interval = False
                intervals.append(
                    (round(interval_start, 5),
                     round(current_position - step, 5),
                     max_score,
                     round(max_score_position, 5))
                )

        current_position += step  # Move to the next position

    # Check if ended while still in an interval
    if in_interval:
        intervals.append((round(interval_start, 5), round(end, 5)))

    return intervals


def find_next_by_rsl(ratio, target_hmn=60, movement=None, movement_dmn=1.05):

    if movement is None:
        movement = 1
        movement_dmn = 1.5

    rst = find_intervals_for_score_range(
        target_hmn,
        100,
        start=ratio*movement/movement_dmn,
        end=ratio*movement*movement_dmn
    )
    while len(rst) == 0:
        movement_dmn += 0.05
        rst = find_intervals_for_score_range(
            target_hmn,
            100,
            start=ratio*movement/movement_dmn,
            end=ratio*movement*movement_dmn
        )
    return max(rst, key=lambda x: x[2])[2:]


def find_next_by_mvn(ratio, movement, movement_tlr=1.01, target_rsl=None, target_dmn=15):

    if target_rsl is None:
        target_rsl = 80

    rst = find_intervals_for_score_range(
        target_rsl - target_dmn,
        target_rsl + target_dmn,
        start=ratio * movement / movement_tlr,
        end=ratio * movement * movement_tlr,
    )

    while len(rst) == 0:
        movement_tlr += 0.01
        target_dmn += 5
        rst = find_intervals_for_score_range(
            target_rsl - target_dmn,
            target_rsl + target_dmn,
            start=ratio * movement / movement_tlr,
            end=ratio * movement * movement_tlr,
        )
    return max(rst, key=lambda x: abs(math.log(x[3]/ratio)))[2:]


def freq_rsl(a, b, base_mv=0., top_mv=0., ratio_chg=None):

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
        movement=ratio_chg,
    )[1]

    if base_mv == 0.:
        b_2 = b * 2 ** (1 / 12 * top_mv) * (new_rt / exp_new_rt)
        a_2 = b_2 / new_rt

    else:
        a_2 = a * 2 ** (1 / 12 * base_mv) * (exp_new_rt / new_rt)
        b_2 = a_2 * new_rt

    return round(a_2, 5), round(b_2, 5)


if __name__ == '__main__':
    print(freq_rsl(95, 786))
    print(freq_rsl(528, 796, base_mv=2))
    print(freq_rsl(477, 986, base_mv=-1))
    print(freq_rsl(477, 986, top_mv=-0.7))
    print(freq_rsl(477, 986, ratio_chg=0.78))
