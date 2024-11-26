import streamlit as st
import numpy as np
import pandas as pd

def polynomial_function(x, coefficients):
    """Evaluate the polynomial for given x and coefficients"""
    return sum(c * (x ** i) for i, c in enumerate(coefficients))

def fitnessFunction(x, coefficients):
    """Fitness function: Closer the polynomial value to zero, better the fitness."""
    return 1 / (1 + abs(polynomial_function(x, coefficients)))

def genetic_algorithm(coefficients, populationSize=10, generations=100, pc=0.7, mc=0.02, lower_bound=-10, upper_bound=10):
    """Run the genetic algorithm."""
    population = np.random.uniform(lower_bound, upper_bound, populationSize)
    sol = {'solution': [], 'fitness': []}

    for generation in range(generations):
        fitnessValue = np.array([fitnessFunction(ind, coefficients) for ind in population])
        probabilities = fitnessValue / fitnessValue.sum()
        selectedParent = np.random.choice(range(populationSize), size=populationSize, p=probabilities)
        selectedPopulation = population[selectedParent]

        offspring = []
        for i in range(0, populationSize, 2):
            if np.random.rand() < pc:
                crossover_point = np.random.rand()
                parent1, parent2 = selectedPopulation[i], selectedPopulation[i + 1]
                child1 = crossover_point * parent1 + (1 - crossover_point) * parent2
                child2 = crossover_point * parent2 + (1 - crossover_point) * parent1
                offspring.extend([child1, child2])
            else:
                offspring.extend([selectedPopulation[i], selectedPopulation[i + 1]])

        for i in range(len(offspring)):
            if np.random.rand() < mc:
                offspring[i] += np.random.uniform(-0.5, 0.5)
                offspring[i] = np.clip(offspring[i], lower_bound, upper_bound)

        population = np.array(offspring)
        bestIndex = np.argmax(fitnessValue)
        solution = population[bestIndex]
        fitness = fitnessValue[bestIndex]
        sol['solution'].append(solution)
        sol['fitness'].append(fitness)

    sol = pd.DataFrame(sol)
    best, fitness = sol[sol['fitness'] == sol['fitness'].max()].iloc[0].values
    return best, fitness, sol

st.title("Genetic Algorithm Solver for Polynomial Equations")
st.markdown("This app uses a genetic algorithm to find the roots of quadratic, cubic, or custom polynomial equations.")

degree = st.selectbox("Select the degree of the polynomial equation:", [2, 3, 4], index=0)

st.markdown(f"Enter the coefficients for your degree-{degree} polynomial:")
st.latex(f"a_{degree}x^{degree} + a_{degree-1}x^{degree-1} + ... + a_1x + a_0 = 0")

coefficients = []
for i in range(degree, -1, -1):
    coef = st.number_input(f"Coefficient for x^{i}:", value=0.0)
    coefficients.append(coef)

coefficients = coefficients[::-1]

if st.button("Run Genetic Algorithm"):
    st.write("Polynomial coefficients (highest to lowest degree):", coefficients[::-1])

  
    best_solution, best_fitness, solutions_df = genetic_algorithm(coefficients)

    st.write(f"**Best solution found:** {best_solution}")
    st.write(f"**Best fitness value:** {best_fitness}")

    st.line_chart(solutions_df['fitness'])
