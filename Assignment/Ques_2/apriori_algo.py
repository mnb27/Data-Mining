"""
2018CSB1069 - Aman Bilaiya
Implementation of Apriori Algorithm with extended prefix tree optimization as given in Zaki's Book
"""

from itertools import combinations # used to generate subset combinations of given set
import math
from datetime import datetime # used to calculate program execution time
import dataset_info # lib implemented for parsing data
from time import time

class Apriori:
  def __init__(self, dataset, total_txns, min_support_cnt):
    self.dataset = dataset # horizontal dataset
    self.min_support_cnt = min_support_cnt # min support required
    self.total_txns = total_txns # length of txn

  def getSupportCount(self, itemList, k, printIT=False): # returns support count of items
    supArr = dict()
    total = 0
    for transaction in self.dataset:
        subsets = list(combinations(transaction, k))
        K_subsets = list( set(subsets).intersection(set(itemList)) ) # generated k subsets and will find Ck for k+1 using this
        SupArrdictKeys = supArr.keys()
        for st in K_subsets:
            total = total + 1
            if(printIT):
                print("In getSup func", st)
            if st not in SupArrdictKeys:
                supArr[st] = 1
            else:
                supArr[st] = supArr[st] + 1
    
    for key in SupArrdictKeys:
        supArr[key] = supArr[key]/self.total_txns # updating support count as ratio
    return supArr, total

  def buildPrefixTree(self, Candidate, Parent, leafNodes, printTree=False):
    candidate_k1List = list()
    for nodeX in leafNodes:
        if(printTree):
            print("X: ",nodeX)
        par = Parent[nodeX]
        siblings_nodes = Candidate[par]
        for nodeY in siblings_nodes:
            if(printTree):
                print("Y: ",nodeY)
            if siblings_nodes.index(nodeY) > siblings_nodes.index(nodeX):
               
                nodeXY = list( set(nodeX).union(set(nodeY)) )  # combine elements of nodeX and nodeY
                nodeXY.sort()
                if(printTree):
                    print("X: ",nodeXY)
                subsets_only = list(combinations(nodeXY, len(nodeXY)-1)) # pruning stage
                subsets = list( set(subsets_only).intersection(set(leafNodes)) ) # k-1 len subsets of nodeXY intersection leaf_nodes

                Ckeys = Candidate.keys()
                if len(subsets_only) == len(subsets):
                    nodeXY_lst = list(combinations(nodeXY, len(nodeXY)))
                    if nodeX not in Ckeys:
                        Candidate[nodeX] = nodeXY_lst
                    else:
                        Candidate[nodeX] += nodeXY_lst
                    Parent[nodeXY_lst[0]] = nodeX
                    candidate_k1List.append(nodeXY_lst[0])
    return candidate_k1List, True

  def Apriori_Algo(self):
    minSup = float(self.min_support_cnt/self.total_txns)
    freqentItemSets = list()
    leaf_nodes = list()

    for txnRow in self.dataset:
        leaf_nodes = leaf_nodes + txnRow
    leaf_nodes = list(set(leaf_nodes))
    leaf_nodes.sort()
    leaf_nodes = list(combinations(leaf_nodes, 1))

    parent = dict()
    parent['#'] = 'null'
    for leaf in leaf_nodes: parent[leaf] = '#'

    candidate = {'#':list()} # initial prefix tree Ck
    candidate['#'] = leaf_nodes # C1 ------- [(a,), (b,), so on] 
    k = 1 # denotes the level pf prefix tree

    while len(leaf_nodes) != 0:
        itemList = list()
        for leaf in leaf_nodes:
            temp = list(leaf)
            temp.sort()
            lenn = len(temp)
            leaf = list(combinations(temp, lenn))
            itemList += leaf
            leaf = set(leaf)
        dash_leaf_nodes = list()
        shaded_leaf_nodes = dict()

        sup, total = self.getSupportCount(itemList, k, False) # returns dictionary {itemset:support}
        # print(total)

        Freq_K = list()
        for leaf in leaf_nodes:
            if leaf not in sup.keys():
                sup[leaf] = 0
            if sup[leaf] < minSup: # not satisfying min sup rule so remove from tree
                shaded_leaf_nodes[leaf] = parent[leaf]
                parent[leaf] = 'null'
            else:
                F_k_dict = dict()
                # F_k_dict[leaf]=sup[leaf]
                F_k_dict[leaf] = math.ceil(sup[leaf]*(self.total_txns)) # added in Frequent itemset list
                Freq_K.append(F_k_dict)
                dash_leaf_nodes.append(leaf) 
        leaf_nodes = dash_leaf_nodes
        for leaf in shaded_leaf_nodes.keys():
            candidate[shaded_leaf_nodes[leaf]].remove(leaf)
        leaf_nodes, status = self.buildPrefixTree(candidate, parent, leaf_nodes, False) # find new candidate set
        if len(Freq_K) != 0:
            freqentItemSets.append(Freq_K)
        k = k + 1

    return freqentItemSets, k


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

    min_support_cnt = 4170.48   # enter this in terms of count not ratio or give ratio*N
    # min_support_cnt = int(input())

    # start_clock = datetime.now() # algo started
    start_clock = time() # algo started
    AprioriInst = Apriori(DATASET, N, min_support_cnt)

    freqItemSets, lvl = AprioriInst.Apriori_Algo()
    
    # uncomment below line to print frequent items 
    # print(freqItemSets)

    # print("Max k explored or valid : ",lvl-1)
    print("Dataset Taken :", dataset_path)
    print("Total Transactions :", N)
    print("Support Count Taken :",min_support_cnt)

    # uncomment below lines to print frequent items k-wise 
    k = 1
    for k_freq in freqItemSets:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(k_freq), "---> ")
        k = k + 1
        # print(k_freq)
        print()

    # finish_clock = datetime.now()
    # print("Time taken: ",round((finish_clock - start_clock).total_seconds(), 2), " seconds")

    finish_clock = time()
    print("Time Taken: " + "%.4f" % (finish_clock - start_clock) + " seconds")

    mem_usage = memory_usage_psutil()
    print("Memory used: ",float(mem_usage/(1024*1024))," MB")

# For Testing Purpose
if __name__ == "__main__":
    datasets_dirs = ["datasets/test.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[2])