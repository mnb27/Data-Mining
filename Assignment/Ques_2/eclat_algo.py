from datetime import datetime
from itertools import combinations
import dataset_info

# Global Variables
DATASET = list()  # dataset on which apriori algo will be applied
N  = int() # No. of transactions
min_support_cnt = int()
d = dict() # diffset (it reduces search time)
Freq = list()

        
def base(data):
    global min_support_cnt
    global d
    P = dict()
    P_help = list()
    Transaction = set()
    for i in data.keys():
        Transaction = Transaction.union(data[i])
    for i in data.keys():
        if len(data[i]) >= min_support_cnt: 
            a = list(combinations([i], len([i])))[0]
            d[a] = Transaction.difference(data[i])
            P[a] = [d[a], len(data[i])]
            P_help.append(a)
    return P, P_help


def recursive_eclat(P, P_help):
    global min_support_cnt
    global d
    global Freq
    #print(self.Freq)
    for i in P_help:
        temp = dict()
        temp[i] = P[i][1]
        Freq.append(temp)
        P_temp = dict()
        P_help_temp = list()
        for j in P_help:
            if P_help.index(j) > P_help.index(i):
                temp = set(i).union(set(j))
                ij = list(combinations(temp, len(temp)))[0]
                d[ij] = (d[j]).difference(d[i])
                sup_ij = P[i][1] - len(d[ij])
                if sup_ij >= min_support_cnt:
                    P_temp[ij] = [d[ij], sup_ij]
                    P_help_temp.append(ij)
        if len(P_help_temp) > 0:
            recursive_eclat(P_temp, P_help_temp)


def Eclat_Algo(data):
    P, P_help = base(data)
    recursive_eclat(P, P_help)
    time_end = datetime.now()

    itemset_length = dict()
    for i in Freq:
        itemset = list(i.keys())[0]
        if len(list(itemset)) in itemset_length.keys():
            itemset_length[len(list(itemset))].append(i)
        else:
            itemset_length[len(list(itemset))] = [i]
    
    Freq_itemsets = list()
    for i in sorted(itemset_length.keys()):
        Freq_itemsets.append(itemset_length[i])
    return Freq_itemsets, time_end
        

# For Testing Purpose
def main(dataset_path):
    global DATASET
    global N
    global min_support_cnt
    horizontal_database, vertical_database = dataset_info.parse_transaction_dataset(dataset_path)
    # dataset_info.print_dataset_info(dataset3)
    min_support_cnt = 2
    DATASET = vertical_database  # Vertical Dataset table [Txn x Items]
    N = len(DATASET)
    freqItemSets, time_end = Eclat_Algo(DATASET)
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
    main("datasets/te.txt")