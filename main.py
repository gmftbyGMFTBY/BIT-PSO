from tqdm import tqdm
from pso import *
import numpy as np
import os
from pprint import pprint
import argparse

def prepare_data(path):
    with open(path) as f:
        data, cache = [], []
        for line in f.readlines():
            if '+' in line or 'instance' in line:
                continue
            elif len(line.split()) == 2:
                if cache:
                    data.append(cache)
                cache = [line]
            else:
                cache.append(line)
        if cache:
            data.append(cache)
    for i, s in enumerate(data):
        with open(f'data/test_{i}.txt', 'w') as f:
            for line in s:
                f.write(line)
    print(f'[!] find {len(data)} samples in the {path}')

def read_data(path):
    '''
    read data from input stream
    '''
    path = f'data/test_{path}.txt'
    print(f'[!] load the data from {path}')
    with open(path) as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines]
    n_m = lines[0]
    n, m = list(map(int, n_m.split()))
    schedule = []
    times = {}
    for i in range(n):
        times[i] = {}
        for j in range(m):
            times[i][j] = None
    for i in range(n):
        line = list(map(float, lines[i+1].split()))
        index, p = 0, []
        while index < len(line):
            machine = line[index]
            p.append(int(machine))
            index += 1
            time = line[index]
            index += 1
            times[i][int(machine)]= time
        schedule.append(p)
    return n, m, times, schedule

if __name__ == "__main__":
    # test
    # from utils import *
    # item = [0, 2, 1, 2, 1, 0, 2, 1, 0]
    # print(calculate_time(n, m, item, times, schedule))
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default=0, type=int)
    args = parser.parse_args()
    n, m, times, schedule = read_data(args.item)
    agent = PSO(n, m, times, schedule)
    agent.show_parameters()
    agent.train()
    agent.save_rest(f'rest/rest_{args.item}.pkl')
