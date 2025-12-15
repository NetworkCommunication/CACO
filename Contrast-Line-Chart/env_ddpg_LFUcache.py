"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
@Describtion: gkddpg environment
"""
import numpy as np
from other import *
from other import type_lists
from BloomFilter import BloomFilter

def get_relationship(car, num_scar, num_tcar):
    car_red = dict_slice(car, 0, num_tcar)
    car_blue = dict_slice(car, num_tcar, num_scar+num_tcar+1)

    df_relationship = pd.DataFrame()

    min_distance = float('inf')
    for key_red in car_red.keys():
        for key_blue in car_blue.keys():
            x0 = car_red[key_red].position[0]
            y0 = car_red[key_red].position[0]
            x1 = car_blue[key_blue].position[1]
            y1 = car_blue[key_blue].position[1]

            min_distance = min(pow(pow(x1 - x0, 2) + pow(y1 - y0, 2), 0.5), min_distance)

    for key_red in car_red.keys():
        for key_blue in car_blue.keys():
            x0 = car_red[key_red].position[0]
            y0 = car_red[key_red].position[1]
            x1 = car_blue[key_blue].position[0]
            y1 = car_blue[key_blue].position[1]
            distance = pow(pow(x1 - x0, 2) + pow(y1 - y0, 2), 0.5)
            relative_distance = min_distance / distance

            speed_min = min(car_red[key_red].speed, car_blue[key_blue].speed)
            speed_max = max(car_red[key_red].speed, car_blue[key_blue].speed)
            relative_speed = speed_min / speed_max

            relative_direction = 1 if car_red[key_red].direction == car_blue[key_blue].direction else 0

            df_relationship.loc[
                key_red, key_blue - 14] = 1 / 3 * relative_distance + 1 / 3 * relative_speed + 1 / 3 * relative_direction

    # print(df_relationship)
    row_1 = df_relationship.iloc[0].tolist()
    return row_1

def get_popularity_index():
    df_pht = pd.DataFrame()
    for i in range(3):
        df_pht[str(i)] = func1(100, 8)

    list_pht = df_pht.sum(axis=1).tolist()
    arr = np.array(list_pht)

    list_pht_max_index = arr.argsort()[-3:][::-1].tolist()
    return list_pht_max_index

def get_popularity_type(popularity):
    list_pht_max_index_type = []
    for index in popularity:
        list_pht_max_index_type.append(type_lists[int(index)])
    return list_pht_max_index_type

def add_bloom(popularity_type):
    bloom = BloomFilter(10000, 20)
    for tp in popularity_type:
        bloom.add(tp)
    return bloom

def isin_bloom(bloom, types):
    if types in bloom:
        return True
    else:
        return False

# 得到值在list里的排名
def get_sort_index(num, relationship):
    # 关系值从大到小排序
    relationship.sort(reverse=True)
    return relationship.index(num)+1

# 得到成功情况下的时间排序序号
def get_coefficient_success(sort_index):
    if sort_index == 1:
        coefficient = 1
    else:
        coefficient = 1 + 0.5 * (sort_index-1)
    return coefficient

# 得到失败情况下的时间排序序号
def get_coefficient_unsuccess(sort_index):
    return 4 + 0.5 * sort_index

class ENV_DDPG_LFUCACHE:
    def __init__(self, num_car, num_tcar, num_scar, num_task):
        self.relationship = None
        self.progress = None
        self.num_car = num_car
        # 初始化数据
        self.car = get_car_info(num_car)
        self.task = get_task_info(num_task)
        self.num_tcar = num_tcar
        self.num_scar = num_scar
        # num_task
        self.num_task = num_task

        # dn:任务的输入数据大小
        self.dn = [0] * self.num_task
        # cn:完成任务所需的cpu周期量
        self.cn = [0] * self.num_task
        # fn:速度，每秒cpu
        self.fn = [0] * self.num_tcar
        self.fm = [0] * self.num_scar
        # cpu_remain:cpu剩余量，服务车辆 + 1
        self.cpu_remain = [0] * (self.num_scar + 1)

        self.count_wrong = 0
        self.done = False
        self.t_local = 0
        self.t_calculate = 0
        self.t_offload = 0
        self.reward = 0
        self.i_task = 0

        self.rn = 4e6
        self.car_cpu_frequency = 1.5e9
        self.popularity = []
        self.bloom = BloomFilter(10000, 20)
        self.popularity_type = []
        self.t_all = 0
        self.num_unload = 0
        self.num_unload_success = 0

    def get_init_state(self):
        self.num_unload = 0
        self.num_unload_success = 0
        self.count_wrong = 0
        self.t_local = 0
        self.t_calculate = 0
        self.t_offload = 0
        self.t_all = 0
        self.car = get_car_info(self.num_car)

        self.task = get_task_info(self.num_task)

        self.relationship = get_relationship(self.car, self.num_scar, self.num_tcar)

        self.popularity = get_popularity_index()

        self.popularity_type = get_popularity_type(self.popularity)

        self.bloom = add_bloom(self.popularity_type)

        # dn:任务的输入数据大小
        self.dn = [0] * self.num_task
        # cn:完成任务所需的cpu周期量
        self.cn = [0] * self.num_task
        # fn:速度，每秒cpu
        # cpu_remain:cpu剩余量，服务车辆和基站
        if self.done:
            self.i_task = 0
        i = 0
        # cpu剩余量
        while i < (self.num_scar + 1):
            self.cpu_remain[i] = self.car_cpu_frequency
            i += 1

        # 进度
        self.progress = [0] * self.num_task

        state = np.concatenate((self.dn, self.cn, self.progress, self.cpu_remain, self.relationship, self.popularity))
        return state

    def step(self, action):
        # action:是否卸载，卸载给谁，卸载的百分比，服务价格
        i = 0
        while i < 2:
            if action[i] > 1:
                action[i] = 1
            if action[i] < -1:
                action[i] = -1
            i += 1

        # 从action获得数据`
        get1 = 1 if action[0] > 0 else 0  # 是否卸载
        # 卸载给谁 范围[0,6]
        get2 = int((action[1] + 1) * 3 + 0.5)

        # print('是否卸载： {}'.format(get1))
        # print('卸载给谁 {}'.format(get2))

        i_task = self.i_task
        T = self.task[i_task].delay_constraints
        Cpu_task = self.task[i_task].cn
        # 本地卸载
        if get1 == 0:
            t = Cpu_task/self.car[i_task].fn
            # 成功的情况：时延小于时延约束 所需cpu小于该车的cpu量
            if t <= T and Cpu_task <= self.car[0].cpu_frequency:
                self.dn[i_task] = self.task[i_task].dn
                self.cn[i_task] = self.task[i_task].cn
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                # cpu剩余量变动，本地卸载只使用第一辆车的cpu
                self.cpu_remain[0] = self.car[0].cpu_frequency - Cpu_task

                self.car[0].cpu_frequency -= Cpu_task
                self.reward = -7 * t
                self.t_all += t
            else:
                # 本地卸载失败
                self.dn[i_task] = self.task[i_task].dn
                self.cn[i_task] = self.task[i_task].cn
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                self.count_wrong += 1
                self.reward = -10 * T
                self.t_all += T

        self.i_task += 1
        state = np.concatenate((self.dn, self.cn, self.progress, self.cpu_remain, self.relationship, self.popularity))
        return state, self.reward, self.done

