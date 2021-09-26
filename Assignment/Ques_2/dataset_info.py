import numpy as np
from datetime import datetime

def parse_transaction_dataset(dataset_path):
    Lines = list()
    fh = open(dataset_path, "r")
    Lines = fh.readlines()

    horizontal_database = list() # this will be used in apriori and fp growth algo
    vertical_database = dict() # this will be used in eclat algo

    txn_index = 1
    for line in Lines:
        txnRow = list()
        for item in line.split(' '):
            # print("itm",item)
            item = item.strip() # to remove newline char
            if len(item)>0:
                txnRow.append(item)
                if item not in vertical_database.keys():
                    vertical_database[item] = set([txn_index])
                else:
                    vertical_database[item] = (vertical_database[item]).union(set([txn_index]))
        horizontal_database.append(txnRow)
        txn_index = txn_index + 1

    # print(horizontal_database)
    # print(vertical_database)
    
    Htxn_widths = [len(txn_row) for txn_row in horizontal_database] # transaction widths
    # print("widths",w_s)
    sup_i = dict()
    max_sup_items = list()
    min_sup_items = list()
    max_sup = 0
    min_sup = len(horizontal_database)

    vertical_row_w = [len(vertical_database[key]) for key in vertical_database.keys()]
    # print(vertical_row_w)

    max_sup = max(vertical_row_w)
    min_sup = min(vertical_row_w)

    for key in vertical_database.keys():
        if len(vertical_database[key]) in sup_i.keys():
            sup_i[len(vertical_database[key])].append(key)
        else:
            sup_i[len(vertical_database[key])] = [key]
    # print("hola",sup_i)

    max_sup_items = sup_i[max_sup]
    min_sup_items = sup_i[min_sup]


    # round(max_sup/n, 5)
    # print("Horizontal Txns --> ",horizontal_database)
    # print("Vertical Txns --> ",vertical_database)
    # print('Total transactions : ', len(horizontal_database))
    # print('Total unique items : ', len(vertical_database))
    # print('Transaction width Mean:', round(np.mean(Htxn_widths),2))
    # print('For 1 itemset support ranges form : ' + str(min_sup) + ' to ' + str(max_sup))
    # print('Most 1-Frequent itemset(s) :', max_sup_items)
    # print('Least 1-Frequent itemset(s) :', min_sup_items)
    
    return horizontal_database, vertical_database, min_sup, max_sup, min_sup_items, max_sup_items, Htxn_widths

def main(dataset_path):
    # Get Txn data {Tid vs Itemsets} and {itemId vs Txns}
    getDataInfo = parse_transaction_dataset(dataset_path)

    horizontal_database = getDataInfo[0]
    vertical_database = getDataInfo[1]
    print("Horizontal Txns --> ",horizontal_database)
    print("Vertical Txns --> ", vertical_database)
    print('Total transactions : ', len(horizontal_database))
    print('Total unique items : ', len(vertical_database))
    print('Transaction width Mean:', round(np.mean(getDataInfo[6]),2))
    print('For 1 itemset support ranges form : ' + str(getDataInfo[2]) + ' to ' + str(getDataInfo[3]))
    print('Most 1-Frequent itemset(s) :', getDataInfo[5])
    print('Least 1-Frequent itemset(s) :', getDataInfo[4])

# For Testing Purpose
if __name__ == "__main__":
    main("datasets/te.txt")