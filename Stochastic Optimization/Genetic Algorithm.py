import os
import time
import numpy as np
import gurobipy as grb
import random
import sys
from itertools import combinations
np.random.seed(42)


def convertMatrixtoVect(X):
    # shape X : (n_items,n_bins)
    (n_items,n_bins) = X.shape
    ret = np.full(shape = (n_items),fill_value = -1)
    for i in range(n_items):
        for j in range(n_bins):
            if(X[i,j] == 1):
                ret[i] = j
    return ret
def convertVectToMatrix(X,n_items,n_bins):
    ret = np.zeros(shape = (n_items,n_bins))
    for i in range(n_items):
        if(X[i] != -1):
            ret[i,X[i]] = 1
def generate_single_value_matrices(original_matrix):
    n_items, n_bins = original_matrix.shape
    single_value_matrices = []

    for row_idx in range(n_items):
        for col_idx in range(n_bins):
            if original_matrix[row_idx, col_idx] == 1:
                single_matrix = np.zeros((n_items, n_bins))
                single_matrix[row_idx, col_idx] = 1
                single_value_matrices.append(single_matrix)

    return single_value_matrices
def generate_summations(matrices, group_size):
    summations = []

    for combo in combinations(matrices, group_size):
        summation = sum(combo)
        summations.append(summation)

    return summations
def informativeFitness(X,C,W,p,w,n_compulsory_items,n_non_compulsory_items,n_bins,U,B):
    value = 0
    spent = 0
    W_occupied = np.zeros(shape = n_bins)  # weight occupied by the items in the bin j
    Y = np.zeros(shape = n_bins) # variable to understand which bin are rented and which are not
    n_items = n_compulsory_items + n_non_compulsory_items
    for i in range(n_items):
        if(X[i] != -1):
            Y[X[i]] = 1
        if(X[i] != -1):
            W_occupied[X[i]] += w[i]     ## this is to check if the volume occupied by items in the bins are not breaking constraints

    for i in range(n_compulsory_items,n_items):
        if(X[i] != -1):
            value += p[i]
    for j in range(n_bins):
        spent += Y[j] * C[j]
    value = value - spent


    # this is to check that every compulsory items has been delivered
    for i in range(n_compulsory_items):
        if(X[i] == -1):
            print("compulsory_items not delivered")
            break

    ## this is to check if the volume occupied by items in the bins are not breaking constraints
    for j in range(n_bins):
        if(W_occupied[j] > W[j]):
            print("bin capacity exceeded")
            break
    ## we check to not spend more than our budget
    if(spent > B):
        print("budget exceeded")
    dic = {}
    for j in range(n_bins):
        if(Y[j] == 1):
            dic[j] = W[j] - W_occupied[j]
    return -1*value, dic
def feasibleFitness(X,C,W,p,w,n_compulsory_items,n_non_compulsory_items,n_bins,U,B):
    value = 0
    spent = 0
    flag = 111
    W_occupied = np.zeros(shape = n_bins)  # weight occupied by the items in the bin j
    Y = np.zeros(shape = n_bins) # variable to understand which bin are rented and which are not
    n_items = n_compulsory_items + n_non_compulsory_items
    for i in range(n_items):
        if(X[i] != -1):
            Y[X[i]] = 1
        if(X[i] != -1):
            W_occupied[X[i]] += w[i]     ## this is to check if the volume occupied by items in the bins are not breaking constraints

    for i in range(n_compulsory_items,n_items):
        if(X[i] != -1):
            value += p[i]
    for j in range(n_bins):
        spent += Y[j] * C[j]
    value = value - spent

    ##CONSTRAINTS

    # this is to check that every compulsory items has been delivered
    for i in range(n_compulsory_items):
        if(X[i] == -1):
            #print("compulsory_items not delivered")
            flag = -222
            break

    ## this is to check if the volume occupied by items in the bins are not breaking constraints
    for j in range(n_bins):
        if(W_occupied[j] > W[j]):
            #print("bin capacity exceeded")
            flag = -444
            break
    ## we check to not spend more than our budget
    if(spent > B):
        #print("budget exceeded")
        flag = -333
    dic = {}
    for j in range(n_bins):
        if(Y[j] == 1):
            dic[j] = W[j] - W_occupied[j]
    return flag #useless function
def kernelSearch(X,n_items,n_bins):
    X_kernel = X.copy()
    X_bucket = X.copy()
    for i in range(n_items):
        for j in range(n_bins):
            if(X[i,j] > 0.95):
                X_kernel[i,j] = 1
            else:
                X_kernel[i,j] = 0
    for i in range(n_items):
        for j in range(n_bins):
            if(X[i,j] < 0.95 and X[i,j] > 0.1):
                X_bucket[i,j] = 1
            else:
                X_bucket[i,j] = 0
    return X_kernel,X_bucket
def instanceClass(n_items,compulsory_percentage,second_length):
    w = np.random.uniform(1,second_length,n_items) # item weights
    p = np.zeros(n_items)
    for i in range(0,n_items):
        p[i] = np.random.uniform(0.5 * w[i],3 * w[i],1)
    # item profits
    n_compulsory_items = int(compulsory_percentage * n_items)
    n_non_compulsory_items = n_items - n_compulsory_items


    n_type = 4
    V_tot = np.sum(w) # sum of item weights

    W_t = [80,120,180,250] # type 0 has Capacity 80, type 1 has capacity 100, etc
    C_t = [80,120,180,250] # type 0 cost 80, type 1 cost 100, etc
    U_t = np.zeros(n_type,dtype=int)
    for i in range(n_type):
        U_t[i] = np.random.choice(range(int((V_tot / W_t[i]) / n_type) + 1, int(V_tot / W_t[i])))
        #U_t[i] = int(V_tot / W_t[i])
        # In Baldi's paper he use this definition for U_t, he instantiate enough bin of any type to cover all the items weight
        # But i prefer to instantiate not enough bin of type t to cover all the items. But at least the sum of all bins
        # must be able to cover all the items
    n_bins = np.sum(U_t)
    U1 = np.full(U_t[0],W_t[0])
    U2 = np.full(U_t[1],W_t[1])
    U3 = np.full(U_t[2],W_t[2])
    U4 = np.full(U_t[3],W_t[3])
    C1 = np.full(U_t[0],C_t[0])
    C2 = np.full(U_t[1], C_t[1])
    C3 = np.full(U_t[2], C_t[2])
    C4 = np.full(U_t[3], C_t[3])

    W = [*U1, *U2, *U3, *U4]
    C = [*C1, *C2, *C3, *C4]


    B = 0.7 * np.sum(C)
    # sum(C) is the total cost that we have to spend to rent all the bins avaiable
    # and we multiplicate it for a penalization term
    return p,w,n_compulsory_items,n_non_compulsory_items,n_bins,W,C,U_t,B
def gurobiRunner(p,w,n_compulsory_items,n_non_compulsory_items,n_bins,W,C,U,B,n_items):
    model = grb.Model('gbpp')

    # Y = 1 if bin j is rented
    Y = model.addVars(
        n_bins,
        vtype=grb.GRB.CONTINUOUS,
        lb = 0,
        ub = 1,
        name='Y'
    )

    X = model.addVars(
        n_items, n_bins,
        vtype=grb.GRB.CONTINUOUS,
        lb = 0,
        ub = 1,
        name='X'
    )

    bins = range(n_bins)
    items = range(n_items)
    # set objective function
    expr = sum(
        C[j] * Y[j] for j in bins
    )
    expr -= grb.quicksum(p[i] * X[i,j] for i in range(n_compulsory_items, n_items) for j in bins)
    model.setObjective(expr, grb.GRB.MINIMIZE)
    model.update()

    # add capacity constraints
    model.addConstrs(
        ( grb.quicksum(w[i]*X[i,j] for i in items) <= W[j]*Y[j] for j in bins),
        name="capacity_constraint"
    )

    # add compulsory item constraints
    model.addConstrs(
        ( grb.quicksum(X[i,j] for j in bins) == 1 for i in range(n_compulsory_items)),
        name="compulsory_item"
    )

    # add compulsory item constraints
    # note that the first items are compulsory: [0,1, ..., n_compulsory_items - 1,n_compulsory_items, ..., n_items]
    model.addConstrs(
        ( grb.quicksum(X[i,j] for j in bins) <= 1 for i in range(n_compulsory_items, n_items)),
        name="non_compulsory_item"
    )
    model.addConstr(
        ( grb.quicksum(C[j] * Y[j] for j in bins) <= B ),
        name = 'budget'
    )
    # We don't need the number of type constraints because of the construction of the problem



    model.setParam('MIPgap', 0.02)
    model.setParam(grb.GRB.Param.TimeLimit, 300)
    model.setParam('OutputFlag', 1)


    start = time.time()
    model.optimize()
    end = time.time()
    comp_time = end - start
    print(f"computational time: {comp_time} s")
    """
    # print variables
    if model.status == grb.GRB.Status.OPTIMAL:
        for j in bins:
            if Y[j].X >=  0:
                print(f"Y[{j}].X: {Y[j].X}")
                print("Compulsory")
                for i in  range(n_compulsory_items):
                    if X[i,j].X > 0:
                        print(f"X[{i},{j}].X: {X[i,j].X}")
                print("Non Compulsory")
                for i in  range(n_compulsory_items, n_items):
                    if X[i,j].X > 0:
                        print(f"X[{i},{j}].X: {X[i,j].X}")
    """
    ## X_new serve giusto a dichiarare un oggetto di tipo numpy e non far più riferimento ad oggetti di tipo gurobipy
    X_new = np.zeros(shape=(n_items, n_bins))
    Y_new = np.zeros(shape=(n_bins))
    for i in range(n_items):
        for j in range(n_bins):
            X_new[i, j] = X[i, j].X
    for j in range(n_bins):
        Y_new[j] = Y[j].X
    return X_new,Y_new
def fitness(X, C, W, p, w, n_compulsory_items, n_non_compulsory_items, n_bins, U, B):
    # X representation: shape = (n_items)
    # if item i is delivered by bin j -> X[i] = j. Otherwise if item i is not delivered X[i] = -1
    # using this problem representation. One item can not be delivered by two bin in the same time breaking the constraints
    value = 0
    spent = 0
    alpha = 0.7
    W_occupied = np.zeros(shape=n_bins)  # weight occupied by the items in the bin j
    Y = np.zeros(shape=n_bins)  # variable to understand which bin are rented and which are not
    n_items = n_compulsory_items + n_non_compulsory_items
    for i in range(n_items):
        if (X[i] != -1):
            Y[X[i]] = 1  # variable to understand which bin are rented and which are not
        if (X[i] != -1):
            W_occupied[X[i]] += w[i]
            ## this is to check if the volume occupied by items in the bins are not breaking constraints

    for i in range(n_compulsory_items, n_items):
        if (X[i] != -1):
            value += p[i]  # profit from non_compulsory_items
    for j in range(n_bins):
        spent += Y[j] * C[j]
    value = value - spent  # objective function inverted. So we need to maximize this value

    # this is to check that every compulsory items has been delivered
    for i in range(n_compulsory_items):
        if (X[i] == -1):
            value = value * 0.95


    ## this is to check if the volume occupied by items in the bins are not breaking constraints
    for j in range(n_bins):
        exceeded = W_occupied[j] - W[j]
        if(exceeded > 0):
            value = value * 0.85

    ## we check to not spend more than our budget
    if (spent > B):
        value = value * 0.7

    unique_values, counts = np.unique(X, return_counts=True)
    repetitions = dict(zip(unique_values, counts))

    for key in repetitions:
        if (repetitions[key] > 5):
            value = value * 0.6

    return value
def stoppingCriteria(array):
    for index, el in enumerate(array):
        if (el < 1.10 * solution and el > 0.90 * solution):
            print(el)
            return index
    return -1
def gurobiRunnerBinary(p,w,n_compulsory_items,n_non_compulsory_items,n_bins,W,C,U,B,n_items):
    model = grb.Model('gbpp')

    # Y = 1 if bin j is rented
    Y = model.addVars(
        n_bins,
        vtype=grb.GRB.BINARY,
        lb = 0,
        ub = 1,
        name='Y'
    )

    X = model.addVars(
        n_items, n_bins,
        vtype=grb.GRB.BINARY,
        lb = 0,
        ub = 1,
        name='X'
    )

    bins = range(n_bins)
    items = range(n_items)
    # set objective function
    expr = sum(
        C[j] * Y[j] for j in bins
    )
    expr -= grb.quicksum(p[i] * X[i,j] for i in range(n_compulsory_items, n_items) for j in bins)
    model.setObjective(expr, grb.GRB.MINIMIZE)
    model.update()

    # add capacity constraints
    model.addConstrs(
        ( grb.quicksum(w[i]*X[i,j] for i in items) <= W[j]*Y[j] for j in bins),
        name="capacity_constraint"
    )

    # add compulsory item constraints
    model.addConstrs(
        ( grb.quicksum(X[i,j] for j in bins) == 1 for i in range(n_compulsory_items)),
        name="compulsory_item"
    )

    # add compulsory item constraints
    # note that the first items are compulsory: [0,1, ..., n_compulsory_items - 1,n_compulsory_items, ..., n_items]
    model.addConstrs(
        ( grb.quicksum(X[i,j] for j in bins) <= 1 for i in range(n_compulsory_items, n_items)),
        name="non_compulsory_item"
    )
    model.addConstr(
        ( grb.quicksum(C[j] * Y[j] for j in bins) <= B ),
        name = 'budget'
    )
    # We don't need the number of type constraints because of the construction of the problem



    model.setParam('MIPgap', 0.04)
    model.setParam(grb.GRB.Param.TimeLimit, 1000)
    model.setParam('OutputFlag', 1)


    start = time.time()
    model.optimize()
    end = time.time()
    comp_time = end - start
    print(f"computational time: {comp_time} s")
    """
    # print variables
    if model.status == grb.GRB.Status.OPTIMAL:
        for j in bins:
            if Y[j].X >=  0:
                print(f"Y[{j}].X: {Y[j].X}")
                print("Compulsory")
                for i in  range(n_compulsory_items):
                    if X[i,j].X > 0:
                        print(f"X[{i},{j}].X: {X[i,j].X}")
                print("Non Compulsory")
                for i in  range(n_compulsory_items, n_items):
                    if X[i,j].X > 0:
                        print(f"X[{i},{j}].X: {X[i,j].X}")
    """
    # X_new is useful to store the item in X changing from gurobipy type to numpy type
    X_new = np.zeros(shape=(n_items, n_bins))
    Y_new = np.zeros(shape=(n_bins))
    for i in range(n_items):
        for j in range(n_bins):
            X_new[i, j] = X[i, j].X
    for j in range(n_bins):
        Y_new[j] = Y[j].X
    return X_new,Y_new
def postOptimization(X,C,W,p,w,n_compulsory_items,n_non_compulsory_items,n_bins,U,B):
    value = 0
    spent = 0
    W_occupied = np.zeros(shape=n_bins)  # weight occupied by the items in the bin j
    W_left = np.zeros(shape=n_bins)
    Y = np.zeros(shape=n_bins)  # variable to understand which bin are rented and which are not
    n_items = n_compulsory_items + n_non_compulsory_items
    for i in range(n_items):
        if (X[i] != -1):
            Y[X[i]] = 1
        if (X[i] != -1):
            W_occupied[X[i]] += w[i]  ## this is to check if the volume occupied by items in the bins are not breaking constraints
    for i in range(n_compulsory_items, n_items):
        if (X[i] != -1):
            value += p[i]
    for j in range(n_bins):
        spent += Y[j] * C[j]
    value = value - spent



    for j in range(n_bins):
        W_left[j] = W[j] - W_occupied[j]

    for j in range(n_bins):  # of course W_left has meaning only if the bin is rented
        if(Y[j] == 0):
            W_left[j] = 0



    for i in reversed(range(n_items)):
        if(X[i] != -1):
            if W_left[X[i]] < 0:
                j = X[i]
                X[i] = -1
                W_left[j] += w[i]
                for k in range(n_items):
                    if(X[k] == -1 and w[k] < W_left[j]):
                        X[k] = j
                        W_left[j] -= w[k]
            if(W_left[j] == W[j]): #if we do not have item inside a bin, it means that we do not need to rent it
                Y[j] = 0


    for i in range(n_compulsory_items):
        if(X[i] == -1):
            for j in range(n_bins):
                if(w[i] < W_left[j]):
                    X[i] = j
                    W_left[j] -= w[i]
                    break
        if(X[i] == -1): #if after searching into W_left, we do not find a feasible bin, we are gonna rent a new bin
            while(1):
                j = np.random.choice(range(0,n_bins))
                if(Y[j] == 0):
                    if(W[j] < w[i]):
                        continue
                    else:
                        Y[j] = 1
                        W_left[j] = W[j] - w[i]
                        X[i] = j
                        break   # not a fancy programming techniques
                else:
                    continue

    return X


def generateResults(n_items,compulsory_percentage,second_length):
    (p,w,n_compulsory_items,n_non_compulsory_items,n_bins,W,C,U,B) = instanceClass(n_items,compulsory_percentage,second_length)
    n_bins = int(n_bins)
    n_items = n_compulsory_items + n_non_compulsory_items

    start = time.time()

    #X,Y = gurobiRunnerBinary(p,w,n_compulsory_items,n_non_compulsory_items,n_bins,W,C,U,B,n_items)
    X,Y = gurobiRunner(p,w,n_compulsory_items,n_non_compulsory_items,n_bins,W,C,U,B,n_items)
    flag = 0
    if(flag == 1):
        #raise KeyboardInterrupt("i am running only gurobi application")
        return






    # Inside X_kernel we have items with value higher than fact = 0.95, so items with high probability to be picked
    # Inside X_bucket we have value that has probability to be taken not too high, but not too low either
    X_kernel,X_bucket = kernelSearch(X,n_items,n_bins)


    # from X_bucket we create (num = 1's inside X_bucket) new matrixes.
    # Each matrix has only one value equal to 1 in different positions
    # In other words, the sum of these matrixes is X_bucket itself
    resulting_matrices = generate_single_value_matrices(X_bucket)
    #Then we create new matrixes, creating matrixes with group_size 1's inside.
    # We create several new matrixes combining the different 1's in group of n = group_size
    group_size = 1  # In order to create groups of different size
    summations = generate_summations(resulting_matrices, group_size)
    # vector encoding : instead of having x[i,j] = 1 for item i delivered by bin j
    # we have shape(vect) = n_items and vect[i] = j for item i delivered by bin j and if item i is not delivered
    populations = [] # our genetic algorithm population
    n_range = int(200 / len(summations)) + 1
    for i in range(n_range):
        for el in summations:
            el = el + X_kernel
            el = convertMatrixtoVect(el) # we are gonna convert each matrix into our vector encoding
            populations.append(el)



    populations = np.array(populations)
    num_generations = 150
    population_size = len(populations)



    mutation_rate = 0.9
    crossover_rate1 = 0.2
    crossover_rate2 = 0.3
    probabilities = np.zeros(n_items)
    probabilities[:n_compulsory_items] = 0.4 ## Higher probability for items within 0 to n_compulsory_items
    probabilities[n_compulsory_items:] = 0.6
    ## Normalize probabilities to sum up to 1
    probabilities /= probabilities.sum()

    for generation in range(num_generations):
        # Evaluate fitness for each solution in the population
        fitness_scores = np.array([fitness(solution,C,W,p,w,n_compulsory_items,n_non_compulsory_items,
                                           n_bins,U,B) for solution in populations])
        # Select parents based on fitness scores
        num_parents = population_size // 2
        sorted_indices = np.argsort(fitness_scores)[::-1]
        parents_indices = sorted_indices[:num_parents]
        parents = populations[parents_indices]


        # we create the next generation using crossover (single-point or double-point crossover)
        children = []
        if np.random.rand() < crossover_rate1:
            for _ in range(population_size - num_parents):
                parent1, parent2 = np.random.choice(parents_indices, size=2, replace=False)
                crossover_point = np.random.randint(n_items)
                child = np.concatenate((populations[parent1, :crossover_point], populations[parent2, crossover_point:]))
                children.append(child)
        elif(np.random.rand() > crossover_rate1 and np.random.rand() < crossover_rate2):
            for _ in range(population_size - num_parents):
                parent1, parent2 = np.random.choice(parents_indices, size=2, replace=False)
                crossover_point1 = np.random.randint(0,n_items)
                crossover_point2 = np.random.randint(crossover_point1, n_items)
                child = np.concatenate((populations[parent1, :crossover_point1],populations[parent2, crossover_point1:crossover_point2],populations[parent1, crossover_point2:]))
                children.append(child)
        else:
            for _ in range(population_size - num_parents):
                parent1, parent2 = np.random.choice(parents_indices, size=2, replace=False)
                crossover_point = np.random.randint(n_items)
                if(fitness_scores[parent1] > fitness_scores[parent2]):
                    child = np.concatenate((populations[parent1, :crossover_point], populations[parent1, crossover_point:]))
                else:
                    child = np.concatenate((populations[parent2, :crossover_point], populations[parent2, crossover_point:]))
                children.append(child)

        # Apply mutation to the children
        for child in children:
            if np.random.rand() < mutation_rate:
                mutation_index = np.random.choice(n_items, p=probabilities)
                if(mutation_index < n_compulsory_items):
                    child[mutation_index] = np.random.randint(0, n_bins)
                else:
                    child[mutation_index] = np.random.randint(-1, n_bins)

        # Replace the old population with the combined population of parents and children
        populations[:num_parents] = parents
        populations[num_parents:] = children

    # Find the best solution in the final population
    best_solution = populations[np.argmax(fitness_scores)]

    print("Best solution:", best_solution)
    print("Best fitness:", informativeFitness(best_solution,C,W,p,w,n_compulsory_items,n_non_compulsory_items,
                                           n_bins,U,B))

    best_solution = postOptimization(best_solution,C,W,p,w,n_compulsory_items,n_non_compulsory_items,n_bins,U,B)

    print("Best solution:", best_solution)
    print("Best fitness:", informativeFitness(best_solution,C,W,p,w,n_compulsory_items,n_non_compulsory_items,
                                           n_bins,U,B))
    end = time.time()
    comp_time = end - start
    print(f"computational time: {comp_time} s")




arr_items = [50,100,150,200]
arr_compulsory = [0.5,1]
arr_second = [60,80,100]

for el in arr_compulsory:
    for el2 in arr_items:
        for el3 in arr_second:
            np.random.seed(42)
            second_length = el3
            n_items = el2
            compulsory_percentage = el
            generateResults(n_items,compulsory_percentage,second_length)
            print("####################################\n")
            print("Parameters:\n")
            print("n_items :" + str(n_items) + "\n")
            print("second_length :" + str(second_length) + "\n")
            print("Compulsory_percentage :" + str(compulsory_percentage) + "\n")






