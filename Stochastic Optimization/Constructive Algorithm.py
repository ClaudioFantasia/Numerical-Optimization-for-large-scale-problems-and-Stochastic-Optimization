import os
import time
import numpy as np
import gurobipy as grb
import random
# fixing seed
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
def ratio_sort_key(pair):
    idx, (p_i, w_i) = pair
    return (p_i / w_i, w_i)
def weight_sort_key(pair):
    idx, (p_i,w_i) = pair
    return w_i
def ratio_sort_key_bin(pair):
    C_i, W_i = pair
    return (C_i / W_i, W_i)

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
def FF(S,item):
    if(len(S) == 0): #in the first iteration S is empty
        return -1
    for bin in S:
        if(item["w"] <= bin[1]["W_left"]):
            bin[1]["W_left"] -= item["w"]
            return bin[0]
    return -1 #if we do not find any bin with enough space
def BF(S,item):
    if(len(S) == 0): #in the first iteration S is empty
        return -1
    min = 1000 # a random big number
    min_index = -1
    for bin in S:
        if(item["w"] <= bin[1]["W_left"]):
             if bin[1]["W_left"] < min:
                min = bin[1]["W_left"]
                min_index = bin[0]

    if(min_index == -1): #if we do not find any bin enough space
        return -1
    else:
        for bin in S:
            if(bin[0] == min_index):
                bin[1]["W_left"] -= item["w"]
                return bin[0]
def algorithm1(SIL,SBL,S,X,Y,n_compulsory_items,n_items):
    for item_index,item in enumerate(SIL):
        bin_index = BF(S,item)
        if(bin_index  != -1): #if the FF/BF algorithm found a bin
               X[item_index,bin_index] = 1 #notare che riduciamo capacità del bin in S già nella funzione FF
        else:
            if(item_index < n_compulsory_items): #this means that we are treating compulsory items because of our data structure
                for bin in SBL:
                    if(item["w"] <= bin[1]["W_left"]):
                        bin[1]["W_left"] -= item["w"]
                        X[item_index,bin[0]] = 1
                        S.append(bin)
                        Y[bin[0]] = 1
                        SBL.remove(bin)
                        break
                    else:
                        continue
            else:
                for bin in SBL:
                    if(profitable(item_index,item,bin,SIL) == True): #cioè se profitable torna True e mi torna un bin_index
                        X[item_index,bin[0]] = 1
                        bin[1]["W_left"] -= item["w"]
                        S.append(bin)
                        Y[bin[0]] = 1
                        SBL.remove(bin)
                        break
                    else:
                        continue
                # se alla fine del ciclo for non è mai entrato nell'if, allora vado avanti e non metto l'item i da nessuna parte. Quindi è come se lo rejecto
    #postOptimization(X, Y, S, SBL, SIL)
    return X,Y
def profitable(item_index,item,bin,SIL):
    sub_SIL = SIL[item_index+1:]
    if(item["w"] > bin[1]["W_left"]):
        return False
    Cb = bin[1]["C"]
    Wb = bin[1]["W_left"] - item["w"]
    Pb = item["p"]
    for i in sub_SIL:
        if i["w"] < Wb:
            Wb = Wb - i["w"]
            Pb = Pb + i["p"]
    if Pb > Cb:
        return True
    else:
        return False
def postOptimization(X,Y,S,SBL,SIL):
    items_weight = []
    for i,el in enumerate(SIL):
        items_weight.append(SIL[i]["w"])
    items_weight = np.array(items_weight)
    for j in S: # stiamo iterando sui bin in S
        tmp = X[:, j[0]] # i take the columns of items in bin j
        tmp = tmp * items_weight # columns of items that is 1 if they are delivered and 0 if not delivered
        Uj = np.sum(tmp) # it's gonna be a columns of items weights (if they are delivered)
        for k in SBL: # #we are gonna iterate on the bin that we have not rented
            if(k[1]["W"] > Uj and k[1]["C"] < j[1]["C"]): #i can change the location of my items
                # in my bin j with bin k that has a smaller cost
                X[:,k[0]] = X[:,j[0]]
                X[:,j[0]] = 0
                k[1]["W_left"] = k[1]["W_left"] - Uj
                j[1]["W_left"] = j[1]["W"] # restoro la capacità iniziale
                SBL.append(j)
                S.remove(j)
                S.append(k)
                SBL.remove(k)
                break # se troviamo il bin, smettiamo di iterare e passiamo alla prossima iterazione di j
def generateResult(n_items,compulsory_percentage,second_length):
    (p,w,n_compulsory_items,n_non_compulsory_items,n_bins,W,C,U,B) = instanceClass(n_items,compulsory_percentage,second_length)
    n_bins = int(n_bins)
    n_items = n_compulsory_items + n_non_compulsory_items
    p_compulsory = p[0:n_compulsory_items]
    p_non_compulsory = p[n_compulsory_items:n_items]
    w_compulsory = w[0:n_compulsory_items]
    w_non_compulsory = w[n_compulsory_items:n_items]


    # Compulsory items are sorted at the top of the item list by non-increasing volume
    # Non-compulsory items are then sorted by non decreasing p_i / w_i,
    # if the ratio between two elements is equal the highest w_i is chosen

    indexed_pairs1 = list(enumerate(zip(p_compulsory, w_compulsory)))
    indexed_pairs2 = list(enumerate(zip(p_non_compulsory,w_non_compulsory)))


    # Sort based on the ratio using the ratio_sort_key() function for non_compulsory_items
    # and sort by weight_sort_key for compulsory items
    sorted_pairs1 = sorted(indexed_pairs1, key=weight_sort_key, reverse=True)
    sorted_pairs2 = sorted(indexed_pairs2, key=ratio_sort_key, reverse=True)
    # Extract the sorted indexes from the sorted pairs
    sorted_indexes1 = [idx for idx, _ in sorted_pairs1]
    sorted_indexes2 = [idx for idx, _ in sorted_pairs2]


    # Use the sorted indexes to get the sorted lists
    sorted_p_compulsory = [p_compulsory[idx] for idx in sorted_indexes1]
    sorted_w_compulsory = [w_compulsory[idx] for idx in sorted_indexes1]
    sorted_p_non_compulsory = [p_non_compulsory[idx] for idx in sorted_indexes2]
    sorted_w_non_compulsory = [w_non_compulsory[idx] for idx in sorted_indexes2]

    sorted_p = [*sorted_p_compulsory, *sorted_p_non_compulsory]
    sorted_w = [*sorted_w_compulsory, *sorted_w_non_compulsory]

    #####
    # We sort our bin by C_j / W_j non decreasing, if two ratio between two bins are equal we take the smallest W_j
    sorted_pairs = sorted(zip(C, W), key=ratio_sort_key_bin)

    sorted_C, sorted_W = zip(*sorted_pairs)
    sorted_C = list(sorted_C)
    sorted_W = list(sorted_W)


    #SIL = {"p" : sorted_p.copy(), "w" : sorted_w.copy()}
    #SBL = {"P" : sorted_C.copy(), "W" : sorted_W.copy()}
    S = []
    SIL = []
    for i in range(0,n_items):
        SIL.append({"p" : sorted_p[i],"w" : sorted_w[i]})
    SBL = []
    for i in range(0,n_bins):
        SBL.append((i,{"C" : sorted_C[i], "W" : sorted_W[i], "W_left" : sorted_W[i]}))

    X = np.zeros(shape = (n_items,n_bins),dtype = int)
    # if item i is delivered by bin j : x[i,j] = 1, otherwise 0
    # The same item cannot be delivered by two or more bins at same time, this means that every row of matrix X must have
    # one value of 1 per row
    Y = np.zeros(n_bins,dtype = int) # if we rent the bin j : Y[j] = 1, otherwise 0

    #S,SBL structure : (bin_index , {P : int, W : int, W_left : int})
    # bin_index will be useful because i am gonna use two different structure
    # and when we access to S dictonary, i need to take into account the index that bin has in SBL dictonary

    #SIL structure : [   {"p" : int , "w" : int}   ]
    # Note that in the for cycle into S or SBL. (e.g for el in S)
    # bin[0] stands for bin_index, bin[1] stats of the bin







    start = time.time()
    X,Y = algorithm1(SIL,SBL,S,X,Y,n_compulsory_items,n_items)
    end = time.time()
    comp_time = end - start
    print(f"computational cost: {comp_time}")
    X = convertMatrixtoVect(X)
    print(informativeFitness(X,sorted_C,sorted_W,sorted_p,sorted_w,n_compulsory_items,n_non_compulsory_items,n_bins,U,B))

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
            generateResult(n_items,compulsory_percentage,second_length)
            print("####################################\n")
            print("Parameters:\n")
            print("n_items :" + str(n_items) + "\n")
            print("second_length :" + str(second_length) + "\n")
            print("Compulsory_percentage :" + str(compulsory_percentage) + "\n")

