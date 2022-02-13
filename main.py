from email.mime import base
import math
from msilib.schema import Class
from multiprocessing import allow_connection_pickling
from sys import stdout
from random import sample
import random
import sys
import os

import numpy
import numpy as np
from itertools import combinations
import pygad
import numpy as np
import pandas as pd
from mip import Model, xsum, BINARY, INTEGER
from dataclasses import dataclass

from pyeasyga import pyeasyga

random.seed(1314141)

# -------- FROM HERE ------------------------------------------------------------#
# External parameters:
# activity = [random.choice([True, False]) for x in range(15)]
activity = [True, True, False, False, True, True, False]
sequence = [3, 2, 1, 5, 28, 12, 16, 9]
max_topologies = 10
rows, cols = (10, 10)

Grid = np.zeros((10, 10))

x1 = 0
x2 = 3
y1 = 0
y2 = 9
route_y = 9
Grid[x1:, y1] = 1
Grid[y2:, :x2] = 1


# print(Grid)

# sys.exit()


# print(Grid)

@dataclass
class workstation:
    num: int
    active: bool


@dataclass
class config:
    x: float
    y: float


all_workstations = []
fitness_value = []
# making workstations
# for num_ws, active in zip(range(len(activity)), activity):
#     all_workstations.append(workstation(num=num_ws+1, active=active))

for num_ws in sequence:
    all_workstations.append(workstation(num=num_ws, active=True))

# base_station means base topology , add active ws in list of base station
base_stations = [ws for ws in all_workstations if ws.active]

# print a base station to check empty topology
for bs in base_stations:
    print(bs)

print("all workstations:", all_workstations)
print("Base     sations:", base_stations)

class topology():
    def __init__(self, ws_list, config_list, top_num):
        self.ws_list = ws_list
        self.configs = config_list
        self.num = top_num

    def display(self):
        for ws, cfg in zip(self.ws_list, self.configs):
            print(f"Workstation number: {ws.num} Coordinates x={cfg.x} y={cfg.y} active={ws.active}")
        print(f"The topology number is {self.num}")

    def calculate_distance(self):
        total_ws = len(self.ws_list)  # 5
        dist = 0.0
        for i in range(total_ws - 1):
            dist += math.sqrt(math.pow(self.configs[i + 1].x - self.configs[i].x, 2) + math.pow(
                self.configs[i + 1].y - self.configs[i].y, 2) * 1.0)
        return dist

    def enlist_postions(self):
        total_ws = len(self.ws_list)
        ws_pos_list = []
        ws_seq_list = []
        for ws, cfg in zip(self.ws_list, self.configs):
            # ws_arrays = [ws.num,[cfg.x, cfg.y]]
            ws_arrays = [cfg.x, cfg.y]
            ws_seq_list.append(ws_arrays)
        return ws_seq_list

    def overlap_ws(self):
        no_of_overlaps = 0
        total_ws = len(self.ws_list)  # 5
        for i in range(total_ws - 1):
            if (math.sqrt(math.pow(self.configs[i + 1].x - self.configs[i].x, 2) + math.pow(
                    self.configs[i + 1].y - self.configs[i].y, 2) * 1.0)) <= 3:
                no_of_overlaps += 1

        return no_of_overlaps

    def overlap_routes(self):
        no_of_overlaps = 0
        route_x = 0
        route_y = 0
        Grid = np.zeros((10, 10))
        total_ws = len(self.ws_list)  # 5
        for i in range(total_ws - 1):
            route_x = self.configs[i + 1].x - self.configs[i].x
            route_y = self.configs[i + 1].y - self.configs[i].y
            if route_x <= route_y:
                Grid[route_x:, route_y] = 1
                Grid[route_y, :route_x] = 1
            if route_y < route_x:
                Grid[0:route_x] = 1
                Grid[route_x:route_y] = 1

        return self.configs

    def fitness_calc(self):
        total_ws = len(self.ws_list)  # 5
        dist = 0.0
        fitness_val = 0
        no_of_overlaps = 0
        for i in range(total_ws - 1):
            dist += math.sqrt(math.pow(self.configs[i + 1].x - self.configs[i].x, 2) + math.pow(
                self.configs[i + 1].y - self.configs[i].y, 2) * 1.0)
            if (math.sqrt(math.pow(self.configs[i + 1].x - self.configs[i].x, 2) + math.pow(
                    self.configs[i + 1].y - self.configs[i].y, 2) * 1.0)) <= 3:
                no_of_overlaps += 1
        if no_of_overlaps >= 1:
            fitness_val = 0
        else:
            fitness_val = round(dist)
        return fitness_val


all_topologies = []
config_list = []
total_cost = []
gen_data = []
# x_pos = random.randint(0, 9)
start_ws_value = (0, 0)
end_ws_value = (9, 9)

# Configuration data for topologies is created here
for num in range(max_topologies):
    config_list.append([config(random.randint(0, 9), random.randint(0, 9)) for i in base_stations])

config_list = np.array(config_list)
config_list[:, 0] = config(start_ws_value[0], start_ws_value[1])
config_list[:, -1] = config(end_ws_value[0], end_ws_value[1])

# for num in range(max_topologies):
#      for i in base_stations:
#          if i != 0 and i != len(sequence):
#              config_list.append([random.randint(0, 9),random.randint(0, 9)])

# print(config_list)

print(config_list)

# Create topologies with Configuration data from config list
for num in range(max_topologies):
    all_topologies.append(topology(base_stations, config_list[num], num + 1))

for top in all_topologies:
    top.display()
    total_cost.append(round(top.calculate_distance()))
    print(top.enlist_postions())
    print("Overlapping workstations:", top.overlap_ws())
    print("Topology cost:", top.calculate_distance())
    # print("Fitness value:", top.fitness_calc())
    print("Overlapping routes:", top.overlap_routes())
    gen_data.append(top.enlist_postions())
    fitness_value.append(top.fitness_calc())

print(total_cost)
print("Total No of Topologies:", len(all_topologies))
print("Topology positions:", gen_data)
# print(np.shape(gen_data))
print("Fitness values of the topologies:", fitness_value)
l1 = []
for i in total_cost:
    if i not in l1:
        l1.append(i)
    # else:
    # print(i,end=' ')

# -------- Gentic ALgorithm CODE ------------------------------------------------------------#


# PYGAD METHOD
print("Genetic code starts here")

seed_data = [[0, 0], [3, 9], [4, 5], [4, 3], [5, 8], [0, 1], [2, 2], [9, 9]]
population = gen_data  # Population created from random individual topologies
desired_output = [[0, 0], [6, 0], [6, 3], [2, 3], [2, 6], [6, 6], [6, 8], [9, 9]]

parents = []
new_population = []








def fitness_score(data):
    global populations, best
    new_config = []
    fit_value = []
    fit_score = []
    for i in range(len(data)):
        # print(data[i][0], data[i][1])
        xy_config = config(data[i][0], data[i][1])
        new_config.append(xy_config)
    top1 = topology(base_stations, new_config, 101)

    return round(top1.calculate_distance())




sys.exit()

#F1 = fitness_func(seed_data)

#print("new ftiness function:", F1)


######## select parent function

best=-100000
populations =([[random.randint(0,1) for x in range(6)] for i in range(4)])
print(type(populations))
parents=[]
new_populations = []
print(populations)

def selectparent():
    global parents
    parents = population[0:2]
    print("parents type:", type(parents))
    print(parents)


selectparent()


########### single-point crossover################
def crossover():
    global parents

    cross_point = random.randint(0, 5)
    parents = parents + tuple([(parents[0][0:cross_point + 1] + parents[1][cross_point + 1:6])])
    parents = parents + tuple([(parents[1][0:cross_point + 1] + parents[0][cross_point + 1:6])])

    print(parents)

crossover()

############ mutation function##################

def mutation() :
    global populations, parents
    mute = random.randint(0,49)
    if mute == 20 :
        x=random.randint(0,3)
        y = random.randint(0,5)
        parents[x][y] = 1-parents[x][y]
    populations = parents
    print(populations)

mutation()

########### Execute GA##################

for i in range(1000) :
    fitness_score()
    selectparent()
    crossover()
    mutation()
print("best score :")
print(best)
print("sequence........")
print(populations[0])

sys.exit()

# # print(data_inputs)
#
#
# def fitness_func():
#     fitness_val = 0
#     for i in fitness_value:
#         f = fitness_value[i]
#     return fitness_val
#
# fitness_function = fitness_func
# num_generations = 50
# num_parents_mating = 4
#
# sol_per_pop = 8
# num_genes = len(function_inputs)
#
# init_range_low = -2
# init_range_high = 5
#
# parent_selection_type = "sss"
# keep_parents = 1
#
# crossover_type = "single_point"
#
# mutation_type = "random"
# mutation_percent_genes = 10
#
#
# ga_instance = pygad.GA(num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_function,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=num_genes,
#                        init_range_low=init_range_low,
#                        init_range_high=init_range_high,
#                        parent_selection_type=parent_selection_type,
#                        keep_parents=keep_parents,
#                        crossover_type=crossover_type,
#                        mutation_type=mutation_type,
#                        mutation_percent_genes=mutation_percent_genes)
#
#
# ga_instance.run()
#
#
#
#
#
#
# sys.exit()
# #
# # # initialise the GA
# # ga = pyeasyga.GeneticAlgorithm(seed_data,
# #                             population_size=200,
# #                             generations=100,
# #                             crossover_probability=0.8,
# #                             mutation_probability=0.2,
# #                             elitism=True,
# #                             maximise_fitness=False)
# #
# # def fitness(individual, data)
# #     fitness =0
# #
# #
# # sys.exit()
# nq = 4
# maxFitness = (nq * (nq - 1)) / 2  # 8*7/2 = 28
#
#
#
#
# def random_chromosome(size):  # making random chromosomes
#     return [random.randint(1, nq) for _ in range(nq)]
#
#
# chromosome_1 = [3, 4, 2, 1]
#
# maxFitness = (nq * (nq - 1)) / 2  # 8*7/2 = 28
#
#
# def fitness(chromosome):
#     horizontal_collisions = sum([chromosome.count(queen) - 1 for queen in chromosome]) / 2
#     diagonal_collisions = 0
#
#     n = len(chromosome)
#     left_diagonal = [0] * 2 * n
#     right_diagonal = [0] * 2 * n
#     for i in range(n):
#         left_diagonal[i + chromosome[i] - 1] += 1
#         right_diagonal[len(chromosome) - i + chromosome[i] - 2] += 1
#
#     diagonal_collisions = 0
#     for i in range(2 * n - 1):
#         counter = 0
#         if left_diagonal[i] > 1:
#             counter += left_diagonal[i] - 1
#         if right_diagonal[i] > 1:
#             counter += right_diagonal[i] - 1
#         diagonal_collisions += counter / (n - abs(i - n + 1))
#
#     return int(maxFitness - (horizontal_collisions + diagonal_collisions))  # 28-(2+3)=23
#
#
# # def fitness(chromosome):
#
#
# print(random_chromosome(4))
# print(fitness(chromosome_1))
#
#
# def probability(chromosome, fitness):
#     return fitness(chromosome) / maxFitness
#
#
# def random_pick(population, probabilities):
#     populationWithProbabilty = zip(population, probabilities)
#     total = sum(w for c, w in populationWithProbabilty)
#     r = random.uniform(0, total)
#     upto = 0
#     for c, w in zip(population, probabilities):
#         if upto + w >= r:
#             return c
#         upto += w
#     assert False, "Shouldn't get here"
#
#
# def reproduce(x, y):  # doing cross_over between two chromosomes
#     n = len(x)
#     c = random.randint(0, n - 1)
#     return x[0:c] + y[c:n]
#
#
# def mutate(x):  # randomly changing the value of a random index of a chromosome
#     n = len(x)
#     c = random.randint(0, n - 1)
#     m = random.randint(1, n)
#     x[c] = m
#     return x
#
#
# def genetic_queen(population, fitness):
#     mutation_probability = 0.03
#     new_population = []
#     probabilities = [probability(n, fitness) for n in population]
#     for i in range(len(population)):
#         x = random_pick(population, probabilities)  # best chromosome 1
#         y = random_pick(population, probabilities)  # best chromosome 2
#         child = reproduce(x, y)  # creating two new chromosomes from the best 2 chromosomes
#         if random.random() < mutation_probability:
#             child = mutate(child)
#         print_chromosome(child)
#         new_population.append(child)
#         if fitness(child) == maxFitness: break
#     return new_population
#
#
# def print_chromosome(chrom):
#     print("Chromosome = {},  Fitness = {}"
#           .format(str(chrom), fitness(chrom)))
#
#
# if __name__ == "__main__":
#     nq = int(input("Enter Number of Queens: "))  # say N = 8
#     maxFitness = (nq * (nq - 1)) / 2  # 8*7/2 = 28
#     population = [random_chromosome(nq) for _ in range(100)]
#
#     generation = 1
#
#     while not maxFitness in [fitness(chrom) for chrom in population]:
#         print("=== Generation {} ===".format(generation))
#         population = genetic_queen(population, fitness)
#         print("")
#         print("Maximum Fitness = {}".format(max([fitness(n) for n in population])))
#         generation += 1
#     chrom_out = []
#     print("Solved in Generation {}!".format(generation - 1))
#     for chrom in population:
#         if fitness(chrom) == maxFitness:
#             print("");
#             print("One of the solutions: ")
#             chrom_out = chrom
#             print_chromosome(chrom)
#
#     board = []
#
#     for x in range(nq):
#         board.append(["x"] * nq)
#
#     for i in range(nq):
#         board[nq - chrom_out[i]][i] = "Q"
#
#
#     def print_board(board):
#         for row in board:
#             print(" ".join(row))
#
#
#     print()
#     print_board(board)
