"""
2018CSB1069 - Aman Bilaiya
Implementation of Apriori Algorithm with some pruning.
"""

from collections import Counter
from time import time
import dataset_info # lib implemented for parsing data

class Apriori:
    def __init__(self, dataset, total_txns, minSup, isRatio=False):
        if(isRatio):
            self.minSup = minSup # min support required
        else:
            self.minSup = float(minSup*total_txns)
        self.dataset = dataset # horizontal dataset
        self.total_txns = total_txns # length of txn

    def findC_1(self): # find C1 frequent sets
        items = [item for txn in self.dataset for item in txn]

        C1Dict = dict()
        C1List = list()
        a = Counter(sorted(items))
        for e in a:
            if((a[e]/ float(self.total_txns)) >= self.minSup):
                C1Dict.update({e:a[e]})
                C1List.append(set([e]))
        return C1List, C1Dict

    def get_K_plus_1_sets(self, Klist): 
        candidate = list()
        for e in Klist:
            for f in Klist:
                a = e | f
                if len(a)-1 == len(e):
                    candidate.append(a)

        resultList = list()
        for i in candidate:
            if i in resultList:
                continue
            else:
                resultList.append(i)

        candidate = resultList

        # pruning itemsets
        resultList = list()
        for e in candidate:
            counter = 0
            for f in self.dataset:
                if(set(f).issuperset(set(e))):
                    counter+=1
            if((counter/float(self.total_txns)) >= self.minSup):
                resultList.append(e)
            else:
                continue
        return resultList, candidate

    def Apriori_Algo(self):

        CandItemsets, CandItemsetsWithSup = self.findC_1()
        freqItemSets = [itemset for itemset in CandItemsets]

        while(CandItemsets != list()):
            CandItemsets, candidate = self.get_K_plus_1_sets(CandItemsets)
            #print(CandItemsets, candidate)
            for itemset in CandItemsets:
                freqItemSets.append(itemset)
        return freqItemSets, len(freqItemSets)
        
def memory_usage_psutil():
    # return the memory usage in bytes
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

# For Testing Purpose
def main(dataset_path):

    print("..................APRIORI ALGORITHM STARTED.................")

    getDataInfo = dataset_info.parse_transaction_dataset(dataset_path)
    DATASET = getDataInfo[0]  # Horizonatal Dataset table [Txn x Items]

    N = len(DATASET)

    minSup = 0.08  # enter this in terms of ratio
    # minSup = float(input())

    start_clock = time() # algo started
    AprioriInst = Apriori(DATASET, N, minSup, True)

    freqItemSets, lvl = AprioriInst.Apriori_Algo()
    
    # uncomment below line to print frequent items 
    # print(freqItemSets)

    # print("Max k explored or valid : ",lvl-1)
    print("Dataset Taken :", dataset_path)
    print("Total Transactions :", N)
    print("Min Support ratio :",minSup)

    kfreq = dict()
    lengths = set()
    for item in freqItemSets:
        lengths.add(len(item))
        kfreq.setdefault(len(item), list()).append(item)

    # # uncomment below lines to print frequent items k-wise 
    for k in lengths:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
        print(kfreq[k])
        print()

    finish_clock = time()
    print("Time Taken: " + "%.4f" % (finish_clock - start_clock) + " seconds")

    mem_usage = memory_usage_psutil()
    print("Memory used: ",float(mem_usage/(1024*1024))," MB")

# For Testing Purpose
if __name__ == "__main__":
    datasets_dirs = ["datasets/test.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[2])