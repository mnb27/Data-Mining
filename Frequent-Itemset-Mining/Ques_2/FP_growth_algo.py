"""
2018CSB1069 - Aman Bilaiya
Implementation of FP Growth Algorithm similar to as given in Zaki's Book with some basic code optimizations.
"""

import dataset_info # lib implemented for parsing data
from time import time # used to calculate program execution time

class helper:
    def __init__(self):
        pass

    def findC_1(self, transactions_db, min_sup): # find C1 frequent sets
        # 1st scan of database
        itemSet = dict()
        delItemSets = list()
        for transaction in transactions_db:
            for item in transaction:
                if item not in itemSet:
                    itemSet[item] = 1
                else:
                    itemSet[item] += 1

        for key, val in list(itemSet.items()):
            if float(itemSet[key]) >= float(min_sup):
                continue
            else:
                delItemSets.append(itemSet[key])
                del itemSet[key]
        return itemSet, delItemSets

    def getParent(self, node, path):
        parNode = node.par
        while parNode.par:
            path.append(parNode.val)
            parNode = parNode.par
            path = path[::-1]
        return parNode, path

    def findPrefixPaths(self, node):
        if node.par is None:  # reached root
            return None
        paths = list()
        while node is not None:
            parNode, path = self.getParent(node, list())
            for i in range(node.counter):
                paths.append(path)
            node = node.nodeLink

        return paths


class freqC1:
    def __init__(self, transactions_db, min_sup, helperInst):
        Candt1, deletedItems = helperInst.findC_1(transactions_db, min_sup)

        # sorting 1-candidate itemsets
        self.Candt1 = self.sortC1(Candt1)
        self._sorted_items = list(self.Candt1.keys())
        self.sortedTxns = self.sortTxns(transactions_db)

    def sortC1(self, Candt1):
        orderSet = sorted(Candt1.items(), key=lambda item: item[1])
        orderSet = orderSet[::-1]
        sortedC1 = {key: value for key, value in orderSet}
        return sortedC1

    def sortTxns(self, transactions_db): # sorted transactions in desc order of their support count
        sortedTxns = list()
        for transaction in transactions_db:
            sortedTxn = [item for item in self._sorted_items if item in transaction]
            sortedTxns.append(sortedTxn)
        return sortedTxns

class NodeStruct:
    def __init__(self, parNode, val, isChild):
        counter = int()
        self.par = parNode
        self.val = val
        counter = 1 if isChild else 0
        self.counter = counter
        self.subNodes = dict()
        self.nodeLink = None  # linked list kind of links of same items


class FPTree:
    def __init__(self, transactions_db, min_sup, helperInst):
        self.min_sup = min_sup
        self.root = NodeStruct(None, "NULL", False)  # each FPTree is having root None
        self.freqC1Inst = freqC1(transactions_db, min_sup, helperInst)
        
        # ptrTable =  list({value, counts, and a pointer to the first occurrance of that item NodeStruct})
        self.ptrTable = self.createPtrTable()

        self.sortedTxns = self.freqC1Inst.sortedTxns
        self.genrateTrie(self.sortedTxns)

    def createPtrTable(self):
        table = list()
        temp = set()
        for key in self.freqC1Inst.Candt1:
            temp = {"val": key,"counter": self.freqC1Inst.Candt1[key],"head": None}
            table.append(temp)
        return table


    def genrateTrie(self, sortedTxns):
        for sortedTxn in sortedTxns:  # 2nd scan of database
            self.buildTrie(sortedTxn, self.root, 0)

    def buildTrie(self, sortedTxn, currNode, insertions=0): # insert item into trie
        # base case
        tSize = len(sortedTxn)
        if tSize == 0:
            return insertions
        insertions += 1
        itr = 0 # iterator for curr item
        
        currNodeItem = list()
        for i, table in enumerate(self.ptrTable):
            if(table["val"] == sortedTxn[itr]):
                currNodeItem.append([table, i])
        currNodeItem = currNodeItem[itr]

        currVal = currNodeItem[0]["val"]
        currItemIndex = currNodeItem[1]

        childrens = currNode.subNodes

        if currVal not in childrens.keys():  # node does not exist
            children_name=currVal
            children_node=NodeStruct(currNode, currVal, True)
            childrens[children_name] = children_node

            # update ptrTable
            if currNodeItem[0]["head"] is not None:
                currNodeent_link_node = self.ptrTable[currItemIndex]["head"]
                target_node = childrens[currVal]

                # the same procedure as "pointing to the next" in C!
                while currNodeent_link_node.nodeLink is not None:
                    currNodeent_link_node = currNodeent_link_node.nodeLink
                currNodeent_link_node.nodeLink = target_node
            else:
                self.ptrTable[currItemIndex]["head"] = childrens[currVal]
        else:
            childrens[currVal].counter+=1


        if tSize > 1:  # insert next item
            subTxn = sortedTxn[1:]
            self.buildTrie(subTxn, childrens[currVal], insertions)

    def getFreqItemsets(self, helperInst, parNode=None):
        countChildrens = self.root.subNodes
        results = list()
        if countChildrens == 0:
            return results
        
        for item in self.ptrTable[::-1]:
            baseSet = {item["val"], }

            freqItemsets = [set(), 0]
            freqItemsets[1] = item["counter"]
            if parNode is not None:
                freqItemsets[0] = baseSet | (parNode[0])  # union with it's parent node
            else:
                freqItemsets[0] = baseSet

            results.append(freqItemsets)
            condFPTreeTxns = helperInst.findPrefixPaths(item["head"]) # conditional FP Tree txns

            condFPTree = FPTree(condFPTreeTxns, self.min_sup, helperInst)
            condFPTree_words = condFPTree.getFreqItemsets(helperInst, freqItemsets)
            if condFPTree_words is not None:
                for word in condFPTree_words:
                    results.append(word)

        return results

def memory_usage_psutil():
    # return the memory usage in bytes
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def main(dataset):
    print("..................FP GROWTH ALGORITHM STARTED.................")

    getData = dataset_info.parse_transaction_dataset(dataset)
    transactions = getData[0]
    N = len(transactions)

    minSup = 0.08 # enter this in terms of ratio
    # minSup = float(input())
    min_sup = float(minSup*N) # min sup count 

    print("Dataset Taken :", dataset)
    print("Total Transactions :", N)
    print("Min Support ratio :",minSup)


    start_clock = time()
    helperInst = helper()
    frequent_itemsets = FPTree(transactions, min_sup, helperInst).getFreqItemsets(helperInst)
    kfreq = dict()
    lengths = set()
    for item in frequent_itemsets:
        lengths.add(len(item[0]))
        # kfreq.setdefault(len(item[0]), list()).append(item)   # uncomment to append itemsets as well as supCount   
        kfreq.setdefault(len(item[0]), list()).append(item[0])

    # uncomment below line to print frequent items 
    # print(kfreq)

    # # uncomment below lines to print frequent items k-wise 
    for k in lengths:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
        print(kfreq[k])
        print()

    finish_clock = time()
    print("Time Taken: " + "%.4f" % (finish_clock - start_clock) + " seconds")

    mem_usage = memory_usage_psutil()
    print("Memory used: ",float(mem_usage/(1024*1024))," MB")

if __name__ == "__main__":
    datasets_dirs = ["datasets/test.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[2])