import numpy as np
import pandas as pd

def polynomial_function(x):
    return x**4 -6*x**3 + 5*x**2 -26*x +24

def fitnessFunction(x):
    """This is fitness function the closer the answer of polynomial is to zero the greater the fitness is"""
    return 1 / (1 + abs(polynomial_function(x)))


populationSize = 10
generations = 100
pc = 0.7 # basically pc is the crossover rate taught by sir saad qasim khan
mc = 0.02 #mc is the mutation rate
lower_bound, upper_bound = -10, 10

# np.random.seed(50)
population = np.random.uniform(-10, 10, populationSize)
sol = {'solution': [], 'fitness':[]}

for generation in range(generations):

    fitnessValue = np.array([fitnessFunction(ind) for ind in population])

    # Selection will be based on simple roulette wheel selection based on fitness
    probabilities = fitnessValue / fitnessValue.sum()
    selectedParent = np.random.choice(range(populationSize), size=populationSize, p=probabilities)
    selectedPopulation = population[selectedParent]

    # Crossover
    offspring = []
    for i in range(0, populationSize, 2):
        if np.random.rand() < pc:
            crossover_point = np.random.rand()
            parent1, parent2 = selectedPopulation[i], selectedPopulation[i+1]
            child1 = crossover_point * parent1 + (1 - crossover_point) * parent2
            child2 = crossover_point * parent2 + (1 - crossover_point) * parent1
            offspring.extend([child1, child2])
        else:
            offspring.extend([selectedPopulation[i], selectedPopulation[i+1]])

    for i in range(len(offspring)):
        if np.random.rand() < mc:
            offspring[i] += np.random.uniform(-0.5, 0.5)  # Random mutation step
            offspring[i] = np.clip(offspring[i], -10, 10)

    # Update population with offspring
    population = np.array(offspring)

    bestIndex = np.argmax(fitnessValue)
    solution = population[bestIndex]
    fitness = fitnessValue[bestIndex]
    sol['solution'].append(solution)
    sol['fitness'].append(fitness)
    print(f"Generation {generation}: Best solution = {solution}, Fitness = {fitness}")

sol = pd.DataFrame(sol)
best, fitness = sol[sol['fitness'] == sol['fitness'].max()].iloc[0].values

print(f'\nThe best solution amongst the {generations} generations is: {best} with fitness: {fitness}')