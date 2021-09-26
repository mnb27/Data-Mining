from datetime import datetime
from itertools import combinations
import dataset_info

## SOME VARIABLE NAMES ARE TAKEN AS GIVEN IN D-ECLAT PSEUDO ALGO IN ZAKI'S BOOK


# Global Variables
DATASET = list()  # vertical dataset on which eclat algo will be applied
N  = int() # No. of transactions
min_support_cnt = int()
d = dict() # diffset (it reduces search time)
Freq = list()


def DEclat(P, P_dash):
    global min_support_cnt
    global d
    global Freq
    for Xa in P_dash:
        temp = dict()
        temp[Xa] = P[Xa][1]
        Freq.append(temp)
        P_temp = dict()
        P_dash_temp = list()
        for Xb in P_dash:
            if P_dash.index(Xb) > P_dash.index(Xa):
                temp = set(Xa).union(set(Xb))
                Xab = list(combinations(temp, len(temp)))[0]
                d[Xab] = (d[Xb]).difference(d[Xa])
                sup_Xab = P[Xa][1] - abs(len(d[Xab]))
                if sup_Xab >= min_support_cnt:
                    P_temp[Xab] = [d[Xab], sup_Xab]
                    P_dash_temp.append(Xab)
        if len(P_dash_temp) != 0:
            DEclat(P_temp, P_dash_temp)


def Eclat_Algo(data):

    # P base
    global min_support_cnt
    global d
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
            
    DEclat(P, P_dash) # Main D-Eclat algo
    time_end = datetime.now()

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
    return Freq_itemsets, time_end
        

# For Testing Purpose
def main(dataset_path):
    global DATASET
    global N
    global min_support_cnt
    getDataInfo = dataset_info.parse_transaction_dataset(dataset_path)
    # dataset_info.print_dataset_info(dataset3)
    min_support_cnt = 2
    DATASET = getDataInfo[1]  # Vertical Dataset table [Txn x Items]
    N = len(DATASET) # transactions
    freqItemSets, time_end = Eclat_Algo(DATASET)
    # print(freqItemSets)
    print("Dataset Taken :", dataset_path)
    print("Total Transactions :", N)
    print("Support Count Taken :",min_support_cnt)
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