import numpy as np
from numpy.random import uniform

def affinity(p_i):

    return p_i[0] * p_i[1] * np.sin(p_i[0]**2 + p_i[1]**2)

def create_random_cells(population_size, problem_size, b_lo, b_up):
    population = [uniform(low=b_lo, high=b_up, size=problem_size) for x in range(population_size)]
    
    return population

def clone(p_i, clone_rate):
    clone_num = int(clone_rate / p_i[1])
    clones = [(p_i[0], p_i[1]) for x in range(clone_num)]
    
    return clones

def hypermutate(p_i, mutation_rate, b_lo, b_up):
    if uniform() <= p_i[1] / (mutation_rate * 100):
        ind_tmp = []
        for gen in p_i[0]:
            if uniform() <= p_i[1] / (mutation_rate * 100):
                ind_tmp.append(uniform(low=b_lo, high=b_up))
            else:
                ind_tmp.append(gen)
                
        return (np.array(ind_tmp), affinity(ind_tmp))
    else:
        return p_i

def select(pop, pop_clones, pop_size):
    population = pop + pop_clones
    
    population = sorted(population, key=lambda x: x[1])[:pop_size]
    
    return population

def replace(population, population_rand, population_size):
    population = population + population_rand
    population = sorted(population, key=lambda x: x[1])[:population_size]
    
    return population


import matplotlib.pyplot as plt
import seaborn as sns

b_lo, b_up = (-1, 1)

population_size = 100
selection_size = 10
problem_size = 2
random_cells_num = 20
clone_rate = 20
mutation_rate = 0.2
stop_codition = 100

population = create_random_cells(population_size, problem_size, b_lo, b_up)
best_affinity_it = []
stop = 0
while stop != stop_codition:

    population_affinity = [(p_i, affinity(p_i)) for p_i in population]
    populatin_affinity = sorted(population_affinity, key=lambda x: x[1])
    
    best_affinity_it.append(populatin_affinity[:5])

    population_select = populatin_affinity[:selection_size]

    population_clones = []
    for p_i in population_select:
        p_i_clones = clone(p_i, clone_rate)
        population_clones += p_i_clones

    pop_clones_tmp = []
    for p_i in population_clones:
        ind_tmp = hypermutate(p_i, mutation_rate, b_lo, b_up)
        pop_clones_tmp.append(ind_tmp)
    population_clones = pop_clones_tmp
    del pop_clones_tmp

    population = select(populatin_affinity, population_clones, population_size)
    population_rand = create_random_cells(random_cells_num, problem_size, b_lo, b_up)
    population_rand_affinity = [(p_i, affinity(p_i)) for p_i in population_rand]
    population_rand_affinity = sorted(population_rand_affinity, key=lambda x: x[1])
    population = replace(population_affinity, population_rand_affinity, population_size)
    population = [p_i[0] for p_i in population]
    
    stop += 1
    
bests_mean = []

for pop_it in best_affinity_it:
    bests_mean.append(np.mean([p_i[1] for p_i in pop_it]))

bests_mean = list(map(lambda x: abs(-0.9063854609626795 - x), bests_mean)) 

f, axes = plt.subplots(1, figsize=(10, 6))    

sns.lineplot(y=bests_mean, x=range(len(bests_mean)),ax=axes).set_title('Immune  Algorithms') 
plt.xlabel('Number of generations')
plt.ylabel('Absolute error')
