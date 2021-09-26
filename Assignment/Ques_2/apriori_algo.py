from itertools import combinations
import dataset_info
import math

# Global Variables
DATASET = list()  # dataset on which apriori algo will be applied
N  = int() # No. of transactions

def getSupportCount(itemList, n):
    global DATASET
    global N
    supArr = dict()
    for transaction in DATASET:
        subsets = list(combinations(transaction, n))
        KSubsets = list( set(subsets).intersection(set(itemList)) ) # generated k subsets and will find Ck for k+1 using this
        SupArrdictKeys = supArr.keys()
        for subset in KSubsets:
            if subset not in SupArrdictKeys:
                supArr[subset] = 1
            else:
                supArr[subset] += 1
    
    for key in SupArrdictKeys:
        supArr[key] = supArr[key]/N # updating sup as ratio
    return supArr


def prefixTree(Candidate, Parent, leaves):
    global DATASET
    global N
    Candidate_k_1 = list()
    for leaf1 in leaves:
        par = Parent[leaf1]
        siblings = Candidate[par]
        for leaf2 in siblings:
            if siblings.index(leaf2) > siblings.index(leaf1):
                # combine elements of leaf1 and leaf2
                leaf12 = list( set(leaf1).union(set(leaf2)) )
                leaf12.sort()
                # pruning
                subsets_only = list(combinations(leaf12, len(leaf12)-1))
                subsets = list( set(subsets_only).intersection(set(leaves)) ) # k-1 len subsets of leaf12 intersection leaves

                Ckeys = Candidate.keys()
                if len(subsets_only) == len(subsets):
                    leaf12_lst = list(combinations(leaf12, len(leaf12)))
                    if leaf1 not in Ckeys:
                        Candidate[leaf1] = leaf12_lst
                    else:
                        Candidate[leaf1] += leaf12_lst
                    Parent[leaf12_lst[0]] = leaf1
                    Candidate_k_1.append(leaf12_lst[0])
    leaves = Candidate_k_1
    return leaves


def Apriori_Algo(minSup):
    '''
    Parameters :
    miSup - Minimum Support required
    Output:
    freqentItemSets : List
    '''
    global DATASET
    global N
    freqentItemSets = list()
    leaves = list()

    for txnRow in DATASET:
        leaves = leaves + txnRow
    leaves = list(set(leaves))
    leaves.sort()
    leaves = list(combinations(leaves, 1))

    parent = dict()
    parent['#'] = 'null'
    for leaf in leaves: parent[leaf] = '#'

    candidate = {'#':list()} # initial prefix tree Ck
    candidate['#'] = leaves # C1 ------- [(a,), (b,), so on] 
    k = 1 # denotes the level pf prefix tree

    while len(leaves) != 0: # no more itemset can be added to tree
        itemList = list()
        for leaf in leaves:
            temp = list(leaf)
            temp.sort()
            lenn = len(temp)
            leaf = list(combinations(temp, lenn))
            itemList += leaf
            leaf = set(leaf)
        new_leaves = list()
        mark_leaves = dict()

        sup = getSupportCount(itemList, k) # returns dictionary {itemset:support}
        Freq_K = list()
        for leaf in leaves:
            if leaf not in sup.keys():
                sup[leaf] = 0
            if sup[leaf] >= minSup:
                F_k_dict = dict()
                # F_k_dict[leaf]=sup[leaf]
                F_k_dict[leaf] = math.ceil(sup[leaf]*N) # added in Frequent itemset list
                Freq_K.append(F_k_dict)
                new_leaves.append(leaf) 
            else: # remove from tree
                mark_leaves[leaf] = parent[leaf]
                parent[leaf] = 'null'
        leaves = new_leaves
        for leaf in mark_leaves.keys():
            candidate[mark_leaves[leaf]].remove(leaf)
        leaves = prefixTree(candidate, parent, leaves) # find new candidate set
        if len(Freq_K) != 0:
            freqentItemSets.append(Freq_K)
        k = k + 1

    return freqentItemSets


# For Testing Purpose
def main(dataset_path):
    global DATASET
    global N
    getDataInfo = dataset_info.parse_transaction_dataset(dataset_path)
    # dataset_info.print_dataset_info(dataset3)
    DATASET = getDataInfo[0]  # Horizonatal Dataset table [Txn x Items]
    N = len(DATASET)
    min_support_cnt = 2
    freqItemSets = Apriori_Algo(min_support_cnt/N)
    print(freqItemSets)
    print("SUPPORT COUNT TAKEN :",min_support_cnt)
    k = 1
    for k_freq in freqItemSets:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(k_freq), "---> ")
        print(k_freq)
        k = k + 1
        print()

# For Testing Purpose
if __name__ == "__main__":
    datasets_dirs = ["datasets/te.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[0])
