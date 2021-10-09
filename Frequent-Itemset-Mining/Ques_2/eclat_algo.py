"""
2018CSB1069 - Aman Bilaiya
Implementation of Eclat Algorithm with D-Eclat optimization similar to as given in Zaki's Book.
"""

from time import time # used to calculate program execution time
from itertools import combinations # used to generate subset combinations of given set
import dataset_info # lib implemented for parsing data

## SOME VARIABLE NAMES ARE TAKEN AS GIVEN IN D-ECLAT PSEUDO ALGO IN ZAKI'S BOOK


# Global Variables
DATASET = list()  # vertical dataset on which eclat algo will be applied
N  = int() # No. of transactions
min_support_cnt = int()
diffset = dict() # to reduce search time
Freq = list()


def DEclat(P, P_dash, printIT=False):
    global min_support_cnt
    global diffset
    global Freq

    d = diffset # shorthand

    for Xa in P_dash:
        temp = dict()
        temp[Xa] = P[Xa][1]
        Freq.append(temp)
        P_temp = dict()
        P_dash_temp = list()
        for Xb in P_dash:
            if P_dash.index(Xb) <= P_dash.index(Xa):
                continue
            else:
                temp = set(Xa).union(set(Xb))
                Xab = list(combinations(temp, len(temp)))[0]
                d[Xab] = (d[Xb]).difference(d[Xa])
                sup_Xab = P[Xa][1] - abs(len(d[Xab]))
                if sup_Xab >= min_support_cnt:
                    P_temp[Xab] = [d[Xab], sup_Xab]
                    P_dash_temp.append(Xab)
        if len(P_dash_temp) != 0:
            if(printIT): print("P_temp: ",P_temp," and P_dash_temp : ",P_dash_temp)
            DEclat(P_temp, P_dash_temp, printIT)


def Eclat_Algo(data, printIT=False):
    # P base
    global min_support_cnt
    global diffset

    d = diffset # shorthand

    P = dict()
    P_dash = list()
    Transaction = set()
    for i in data.keys():
        Transaction = Transaction.union(data[i])
    for i in data.keys():
        sup_i = len(data[i])
        if sup_i >= min_support_cnt: 
            Xa = list(combinations([i], len([i])))[0]
            P_dash.append(Xa)
            d[Xa] = Transaction.difference(data[i])
            P[Xa] = [d[Xa], len(data[i])]
    if(printIT):
        print("Diffset: ",d)
        print("P: ",P," and P_dash : ",P_dash)
    DEclat(P, P_dash, printIT) # core D-Eclat Algorithm

    itemset_length = dict()
    for i in Freq:
        itemset = list(i.keys())[0]
        length = len(list(itemset))
        if length not in itemset_length.keys():
            itemset_length[length] = [i]
        else:
            itemset_length[length].append(i)
    
    Freq_itemsets = list()
    for i in sorted(itemset_length.keys()):
        Freq_itemsets.append(itemset_length[i])
    return Freq_itemsets
        
def memory_usage_psutil():
    # return the memory usage in bytes
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

# For Testing Purpose
def main(dataset_path):

    print("..................ECLAT ALGORITHM STARTED.................")
    
    global DATASET
    global N
    global min_support_cnt
    getDataInfo = dataset_info.parse_transaction_dataset(dataset_path)

    DATASET = getDataInfo[1]  # Vertical Dataset table [Txn x Items]
    N = len(DATASET) # vertical [rows count]
    M = len(getDataInfo[0])

    minSup = 0.08 # enter this in terms of ratio
    # minSup = float(input())
    min_support_cnt = float(minSup*M) # min sup count 

    start_clock = time() # algo started
    freqItemSets = Eclat_Algo(DATASET, False)

    # uncomment below line to print frequent items 
    # print(freqItemSets)
    
    print("Dataset Taken :", dataset_path)
    print("Total Transactions :", len(getDataInfo[0]))
    print("Min Support ratio :",minSup)


    temp = list()
    for i in freqItemSets:
        for j in i:
            temp.append(j)

    kfreq = dict()
    lengths = set()
    for item in temp:
        Size = len(list(list(item.keys())[0]))
        lengths.add(Size)
        kfreq.setdefault(Size, list()).append(list(list(item.keys())[0]))
        # kfreq.setdefault(Size, list()).append(list(list(item.keys()))) # uncomment to append itemsets as well as supCount   

    # # uncomment below lines to print frequent items k-wise 
    for k in lengths:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
        print(kfreq[k])
        print()

    finish_clock = time() # algo started
    print("Time Taken: " + "%.4f" % (finish_clock - start_clock) + " seconds")

    mem_usage = memory_usage_psutil()
    print("Memory used: ",float(mem_usage/(1024*1024))," MB")

# For Testing Purpose
if __name__ == "__main__":
    datasets_dirs = ["datasets/test.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[2])