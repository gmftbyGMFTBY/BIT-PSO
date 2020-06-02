from math import *
from copy import deepcopy
import numpy as np
import random
import ipdb

def calculate_time(n, m, item, times, schedule):
    '''
    item: record the sort index for each machine; [0, 1, 2, 2, 1, 0, 1, 0, 2]
    schedule: the using index for each component on the machine
    times: the using times for each pair of (component and machine)
    '''
    processed_id = [0] * n
    machineWorkTime = [0] * m
    startTime = [[0 for _ in range(m)] for _ in range(n)]
    endTime = [[0 for _ in range(m)] for _ in range(n)]
    final_time = 0
    for wId in item:
        pId = processed_id[wId]
        processed_id[wId] += 1
        mId = schedule[wId][pId]
        t = times[wId][mId]
        if pId == 0:
            startTime[wId][pId] = machineWorkTime[mId]
        else:
            startTime[wId][pId] = max(endTime[wId][pId-1], machineWorkTime[mId])
        machineWorkTime[mId] = startTime[wId][pId] + t
        endTime[wId][pId] = machineWorkTime[mId]
        final_time = max(final_time, machineWorkTime[mId])
    return final_time

def init(n, m, size=100):
    '''
    best init is the one that each machine uses the first component
    return one particle
    '''
    particles = []
    init_seq = []
    for i in range(n):
        init_seq.extend([i] * m)
    for _ in range(size):
        seq = deepcopy(init_seq)
        random.shuffle(seq)
        while seq in particles:
            random.shuffle(seq)
        particles.append(seq)
    return particles

def init_pair(length, size):
    rest = []
    while len(rest) < size:
        pair = np.random.choice(length, 2).tolist()
        if pair[0] != pair[1]:
            rest.append(pair)
    return rest

def find_switch_pairs(x, y, n):
    '''
    x: [0, 0, 0, 1, 2, 1, 2, 2, 1]
    y: [0, 1, 2, 2, 1, 0, 1, 2, 0]
    '''
    def find_idx_item(seq, item, nidx):
        # nidx begin from 1
        counter = 0
        for idx, i in enumerate(seq):
            if i == item:
                counter += 1
            if counter == nidx:
                return idx
    current, l, pairs = 0, len(x), []
    y_counter = [1] * n
    while current < l:
        item_x = x[current]
        idx_y = find_idx_item(y, item_x, y_counter[item_x])
        y_counter[item_x] += 1
        if idx_y != current:
            pairs.append((current, idx_y))
        current += 1
    return pairs

def apply_pairs(x, pairs):
    for p_x, p_y in pairs:
        x[p_x], x[p_y] = x[p_y], x[p_x]

if __name__ == "__main__":
    pairs = find_switch_pairs(
            [0, 0, 0, 1, 1, 1, 2, 2, 2], 
            [0, 0, 0, 1, 1, 1, 2, 2, 2], 
           3)
    print(pairs)
