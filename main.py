import copy
import sys
import math
import random
from multiprocessing.dummy import Pool, Lock
from datetime import datetime

# Alunos: Jhonatan Santos e Tiago Prata

# Variables Global
size = 0
x = []
y = []
_type = ''
distance_matrix = []
lock = Lock()


# Struct Problem
class Set(object):
    index = 0
    distance = 0

    def __init__(self, index=0, distance=sys.maxsize):
        self.index = index
        self.distance = distance

    def __str__(self):
        return f'i: {self.index} d: {self.distance}'


class Solution(object):
    sets = []
    total_distance = 0

    def __init__(self, sets=None, total_distance=0):
        if sets is None:
            sets = []
        self.sets = sets
        self.total_distance = total_distance

    def __str__(self):
        return f'total distance: {self.total_distance}, n_candidates: {len(self.sets)}'


def best_first_search(index, visited):
    _set = Set()
    for i in range(size):
        if distance_matrix[index][i] < _set.distance and not visited[i]:
            _set.distance = distance_matrix[index][i]
            _set.index = i
    return _set


# Creating sets of S
def get_solutions():
    global solutions
    for i in range(size):
        count = 1
        visited = [False] * size
        visited[i] = True
        k = i
        # return array with best tour between cities
        while count < size:
            best = best_first_search(k, visited)
            solutions[i].sets.append(best)
            solutions[i].total_distance += best.distance
            k = best.index
            visited[k] = True
            count += 1
        return_city = Set(i, distance_matrix[k][i])
        solutions[i].sets.append(return_city)
        solutions[i].total_distance += return_city.distance


# Tweak - Recalculate new random distance
def update(solution):
    update_tour = 0
    for i in range(size - 1):
        update_tour += distance_matrix[solution.sets[i].index][solution.sets[i + 1].index]
    solution.total_distance = update_tour
    return copy.deepcopy(solution)


def tweak(solution):
    a, b = random.randint(0, size - 2), random.randint(0, size - 2)
    while not a != b:
        b = random.randint(0, size - 2)
    solution.sets[a], solution.sets[b] = solution.sets[b], solution.sets[a]
    return copy.deepcopy(solution), a, b


def simulated_annealing(index):
    global solutions_sa
    # initial variables
    step = 1
    max_steps = size * 50000
    energy = 200
    initial_solution = copy.deepcopy(solutions[index])
    best_solution = copy.deepcopy(initial_solution)
    while step < max_steps:
        step += 1
        if energy <= 0:
            break
        # Tweak
        solution, a, b = tweak(initial_solution)
        new_solution = update(copy.deepcopy(solution))
        if new_solution.total_distance < initial_solution.total_distance:
            initial_solution = copy.deepcopy(new_solution)
        elif random.uniform(0, 1) < math.exp((initial_solution.total_distance - new_solution.total_distance) / energy):
            initial_solution = copy.deepcopy(new_solution)
        energy -= step / (1 + 0.8 * step)  # Cooling
        if best_solution.total_distance > initial_solution.total_distance:
            best_solution = copy.deepcopy(initial_solution)
    solutions_sa[index] = best_solution
    lock.acquire()
    print_tela(index)
    lock.release()


def print_tela(index):
    print(f'Solution {index}:', end=' ')
    for k in range(size):
        print(f'{solutions_sa[index].sets[k].index}', end=' ')
    print(f'total: {solutions_sa[index].total_distance}')


def calculate_tour_distance(_tour):
    dist = 0
    for i in range(size - 1):
        dist += distance_matrix[_tour[i]][_tour[i + 1]]
    dist += distance_matrix[_tour[size - 1]][_tour[0]]
    return dist


def read_file(filename):
    global _type
    global size
    global x
    global y
    global distance_matrix
    with open(filename, 'r') as f:
        content = f.read().splitlines()
        position = 0
        for line in content[:-1]:
            # print(line)
            if line.startswith('EDGE_WEIGHT_TYPE'):
                if line.split()[-1] not in ['EUC_2D', 'ATT', 'CEIL_2D']:
                    raise Exception('ERROR! tsp file is not of type EUC_2D, ATT or CEIL_2D aborting!!')
                _type = line.split()[-1]
            if line.startswith('DIMENSION'):
                size = int(line.split()[-1])
                x = [0] * size
                y = [0] * size
            if line.split()[0].isnumeric():
                if not size or size == 0:
                    raise ValueError('Size not found')
                x[position] = float(line.split()[1])
                y[position] = float(line.split()[2])
                position += 1
    distance_matrix = [[0 for j in range(size)] for i in range(size)]

    if _type == 'EUC_2D':
        for i in range(0, size):
            for j in range(0, size):
                xd = x[i] - x[j]
                yd = y[i] - y[j]
                distance = math.sqrt(xd * xd + yd * yd)
                # calculating the euclidean distance, rounding to int and storing in the distance matrix
                distance_matrix[i][j] = int(distance + 0.5)
    elif _type == 'CEIL_2D':
        for i in range(0, size):
            for j in range(0, size):
                xd = x[i] - x[j]
                yd = y[i] - y[j]
                distance = math.sqrt(xd * xd + yd * yd)
                distance_matrix[i][j] = int(distance + 0.000000001)
    elif _type == 'ATT':
        for i in range(0, size):
            for j in range(0, size):
                xd = x[i] - x[j]
                yd = y[i] - y[j]
                rij = math.sqrt((xd * xd + yd * yd) / 10.0)
                tij = int(rij + 0.5)
                distance_matrix[i][j] = tij + 1 if tij < rij else tij


if len(sys.argv) != 2:
    raise Exception('Inform path to .tsp')
file = sys.argv[1]
read_file(file)

tour = [i for i in range(size)]
print(f'Route length: {calculate_tour_distance(tour)}')
solutions = {i: Solution() for i in range(size)}
get_solutions()

print('Greedy Algorithm')
best_solution_greedy = solutions[0]
for i in range(size):
    distance = solutions[i].total_distance
    if distance < best_solution_greedy.total_distance:
        best_solution_greedy = solutions[i]

random.seed(datetime.now())
print(f'Best Solution {best_solution_greedy.total_distance}: ', end='')
for j in range(size):
    print(f'{best_solution_greedy.sets[j].index}', end=' ')
print()

solutions_sa = {i: Solution() for i in range(size)}
print('Simulated Annealing')
pool = Pool(4)
results = pool.map(simulated_annealing, range(size))
# for i in range(size):
#     simulated_annealing(i)

# results
best_solution_sa = solutions_sa[0]
for i in range(size):
    distance = solutions_sa[i].total_distance
    if distance < best_solution_sa.total_distance:
        best_solution_sa = solutions_sa[i]

print(f'Best Solution {best_solution_sa.total_distance}: ', end='')
for j in range(size):
    print(f'{best_solution_sa.sets[j].index}', end=' ')
print()
