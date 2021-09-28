from itertools import combinations
import dataset_info
import math

class Apriori:
  def __init__(self, dataset, total_txns, min_support_cnt):
    self.dataset = dataset # horizontal dataset
    self.min_support_cnt = min_support_cnt # min support required
    self.total_txns = total_txns # length of txn

  def getSupportCount(self, itemList, k):
    supArr = dict()
    for transaction in self.dataset:
        subsets = list(combinations(transaction, k))
        KSubsets = list( set(subsets).intersection(set(itemList)) ) # generated k subsets and will find Ck for k+1 using this
        SupArrdictKeys = supArr.keys()
        for subset in KSubsets:
            if subset not in SupArrdictKeys:
                supArr[subset] = 1
            else:
                supArr[subset] += 1
    
    for key in SupArrdictKeys:
        supArr[key] = supArr[key]/self.total_txns # updating sup as ratio
    return supArr

  def buildPrefixTree(self, Candidate, Parent, leaves):
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

  def Apriori_Algo(self):
    '''
    Parameters :
    miSup - Minimum Support required
    Output:
    freqentItemSets : List
    '''
    minSup = self.min_support_cnt/self.total_txns
    freqentItemSets = list()
    leaves = list()

    for txnRow in self.dataset:
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

        sup = self.getSupportCount(itemList, k) # returns dictionary {itemset:support}
        Freq_K = list()
        for leaf in leaves:
            if leaf not in sup.keys():
                sup[leaf] = 0
            if sup[leaf] < minSup: # not satisfying min sup rule so remove from tree
                mark_leaves[leaf] = parent[leaf]
                parent[leaf] = 'null'
            else:
                F_k_dict = dict()
                # F_k_dict[leaf]=sup[leaf]
                F_k_dict[leaf] = math.ceil(sup[leaf]*(self.total_txns)) # added in Frequent itemset list
                Freq_K.append(F_k_dict)
                new_leaves.append(leaf) 
        leaves = new_leaves
        for leaf in mark_leaves.keys():
            candidate[mark_leaves[leaf]].remove(leaf)
        leaves = self.buildPrefixTree(candidate, parent, leaves) # find new candidate set
        if len(Freq_K) != 0:
            freqentItemSets.append(Freq_K)
        k = k + 1

    return freqentItemSets, k


# For Testing Purpose
def main(dataset_path):
    getDataInfo = dataset_info.parse_transaction_dataset(dataset_path)
    DATASET = getDataInfo[0]  # Horizonatal Dataset table [Txn x Items]

    N = len(DATASET)

    min_support_cnt = 2
    # min_support_cnt = int(input())
    AprioriInst = Apriori(DATASET, N, min_support_cnt)

    freqItemSets, lvl = AprioriInst.Apriori_Algo()
    # print(freqItemSets)
    # print("Max k explored or valid : ",lvl-1)
    print("Dataset Taken :", dataset_path)
    print("Total Transactions :", N)
    print("Support Count Taken :",min_support_cnt)
    k = 1
    for k_freq in freqItemSets:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(k_freq), "---> ")
        k = k + 1
        print(k_freq)
        print()

# For Testing Purpose
if __name__ == "__main__":
    datasets_dirs = ["datasets/te.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[0])
