from time import time

def get_transactions_db_from_dataset(file_path):
    """
    Get data from csv file with a specific length. And redirect names to indices for less computing in the algorithm.
    :param csv_file_path: string
    :param data_length: int. no more than 9835.
    :return: name2index, index2name, transactions
    """
    transactions_strings = []
    transactions = []
    name2index = dict()
    current_count = 0

    fh = open(file_path, "r")
    Lines = fh.readlines()
    for st in Lines:
        st = st.strip()
        transactions_strings.append(st)

    transactions_data = [transaction_string.split(" ") for transaction_string in transactions_strings]

    for transaction in transactions_data:
        for item in transaction:
            if item not in name2index.keys():
                name2index[item] = current_count
                current_count += 1
    index2name = {value: key for key, value in name2index.items()}  # reverse

    for transaction in transactions_data:
        itemset = [name2index[item] for item in transaction]
        transactions.append(itemset)

    return name2index, index2name, transactions


def get_frequent_one_itemsets_and_counts(transactions_db, min_sup):
    """
    Get frequent 1-itemsets and their occurrences from a transactions database.
    This will be the "first-scan" of database in FP_Growth.
    :param transactions_db: all transactions in a list of list of numbers.
    :return: frequent 1-itemsets (in dictionary showing their occurrences).
    """
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


class FrequentItemsList:
    """
    An object records f-list which sort frequency items in frequency descending order.
    """
    def __init__(self, transactions_db, min_sup):
        frequent_items_name_count_dict, deletedItems = get_frequent_one_itemsets_and_counts(transactions_db, min_sup)
        self._sorted_frequent_items_name_count_dict = {k: v for k, v in sorted(frequent_items_name_count_dict.items(),
                                                                               key=lambda item: item[1], reverse=True)}
        self._sorted_items = list(self._sorted_frequent_items_name_count_dict.keys())
        self._sorted_transactions_db = self._sort_transactions_db(transactions_db)

    def _sort_transactions_db(self, transactions_db):
        """
        Descending sort frequent items.
        :param transactions_db:
        :return:
        """
        _sorted_transactions_db = []
        for transaction in transactions_db:
            sorted_transaction = [item for item in self._sorted_items if item in transaction]
            _sorted_transactions_db.append(sorted_transaction)

        return _sorted_transactions_db

class TreeNode:
    """
    This class defines the node information in a FP tree.
    """
    def __init__(self, item_name, item_count, parent_node):
        self._item_name = item_name
        self._item_count = item_count
        self._parent = parent_node
        self._children = {}  # This will be an "item name - TreeNode" pairs.
        self._node_link = None  # Link other nodes that have the same item name (for header table use)

def get_prefix_paths(node):
    """
    This method is used to find prefix paths of a specific node in the tree.
    :param node: TreeNode
    :return: list
    """
    if node._parent is None:  # reach top
        return None

    paths = []
    while node is not None:
        route = []
        parent_node = node._parent
        while parent_node._parent is not None:  # iteratively find parent
            route.append(parent_node._item_name)
            parent_node = parent_node._parent

        route = route[::-1]  # reverse
        for i in range(node._item_count):
            paths.append(route)

        node = node._node_link

    return paths


class FPTree:
    """
    This class defines an FPTree object used in FP Growth algorithm.
    Will also be called to generate conditional FPTree from conditional pattern base.
    """
    def __init__(self, transactions_db, min_sup):
        self._min_sup = min_sup
        self._root = TreeNode("ROOT (NONE)", 0, None)  # Every tree should have a root of None
        self._frequent_item_list_object = FrequentItemsList(transactions_db, min_sup)  # Get specific transactions

        # _header_table is a list of 3-items dictionary.
        # Each dictionary records frequent item name, their counts, and a pointer to the first occurrence TreeNode.
        self._header_table = [{"item_name": key,
                               "frequency": self._frequent_item_list_object._sorted_frequent_items_name_count_dict[key],
                               "head": None}
                              for key in self._frequent_item_list_object._sorted_frequent_items_name_count_dict]

        self._sorted_transactions_db = self._frequent_item_list_object._sorted_transactions_db
        self._create_tree(self._sorted_transactions_db)  # Driver to the construction process

    def _create_tree(self, sorted_transactions_db):
        for sorted_transaction in sorted_transactions_db:  # This is the "second-scan" of database.
            self._insert_tree(sorted_transaction, self._root)

    def _insert_tree(self, sorted_transaction, current_tree_node):
        """
        This will be a recursive method. We will insert item-by-item to the FPTree.
        :param sorted_transaction: in a type of list.
        :param current_tree_node: a ref to a TreeNode Object.
        :return:
        """
        if len(sorted_transaction) == 0:  # finish
            return

        # Record current item info.
        # Including index for a specific item for easily locating when we update _header_table.
        current_item_info = [[_item_table, index] for index, _item_table in enumerate(self._header_table)
                             if _item_table["item_name"] == sorted_transaction[0]][0]
        current_item_name = current_item_info[0]["item_name"]
        current_item_index = current_item_info[1]

        if current_item_name in current_tree_node._children.keys():  # check if node exists already
            current_tree_node._children[current_item_name]._item_count+=1
        else:
            children_name=current_item_name
            children_node=TreeNode(item_name=current_item_name, item_count=1, parent_node=current_tree_node)
            current_tree_node._children[children_name] = children_node

            # update _header_table
            if current_item_info[0]["head"] is None:
                self._header_table[current_item_index]["head"] = current_tree_node._children[current_item_name]
            else:
                current_link_node = self._header_table[current_item_index]["head"]
                target_node = current_tree_node._children[current_item_name]

                # the same procedure as "pointing to the next" in C!
                while current_link_node._node_link is not None:
                    current_link_node = current_link_node._node_link
                current_link_node._node_link = target_node

        if len(sorted_transaction) > 1:  # insert next item
            self._insert_tree(sorted_transaction=sorted_transaction[1:],
                              current_tree_node=current_tree_node._children[current_item_name])

    def get_header_table(self):
        return self._header_table

    def mine_frequent_itemsets(self, parent_node=None):
        """
        The general method for finding frequent itemsets from FPTree.
        :param parent_node: last node info for recursive use.
        :return: mined results
        """
        if len(self._root._children) == 0:  # reach bottom
            return None
        results = []
        min_sup = self._min_sup
        reversed_header_table = self._header_table[::-1]
        for item in reversed_header_table:
            frequent_itemset_and_count = [set(), 0]
            if parent_node is None:
                frequent_itemset_and_count[0] = {item["item_name"], }  # comma means a set
            else:
                frequent_itemset_and_count[0] = {item["item_name"], }.union(parent_node[0])  # unite set with its parent

            frequent_itemset_and_count[1] = item["frequency"]
            results.append(frequent_itemset_and_count)
            cond_tree_transactions = get_prefix_paths(item["head"])

            cond_tree = FPTree(cond_tree_transactions, min_sup)
            cond_tree_words = cond_tree.mine_frequent_itemsets(frequent_itemset_and_count)
            if cond_tree_words is not None:
                for word in cond_tree_words:
                    results.append(word)

        return results


def post_process_frequent_itemsets(frequent_itemsets, index2name):
    """
    index back to name. Store into a dictionary.
    :param frequent_itemsets:
    :param index2name:
    :return:
    """
    final_results = dict()
    for frequent_itemset in frequent_itemsets:
        item_names = [index2name[key] for key in frequent_itemset[0]]
        item_names_string = ','.join(item_names)
        final_results[item_names_string] = frequent_itemset[1]

    final_results = {k: v for k, v in sorted(final_results.items(), key=lambda item: item[1], reverse=True)}
    return final_results

def memory_usage_psutil():
    # return the memory usage in bytes
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def main(dataset, min_sup = 319.6):
    """
    Driver to the program.
    """
    print("..................FP GROWTH ALGORITHM STARTED.................")
    
    start_clock = time()
    # get data and use the index for computing
    # dataset_path = "datasets/liquor_11frequent.txt"
    name2index, index2name, transactions = get_transactions_db_from_dataset(file_path=dataset)

    print("Dataset Taken :", dataset)
    # print("Total Transactions :", len(getDataInfo[0]))
    print("Support Count Taken :",min_sup)

    # run
    frequent_itemsets = FPTree(transactions, min_sup).mine_frequent_itemsets()
    freqItemSets = post_process_frequent_itemsets(frequent_itemsets, index2name)
    # print(freqItemSets)

    # for item, sup in freqItemSets.items():
        # print(item,"---",sup)

    kfreq = dict()
    lengths = set()
    for key in sorted(freqItemSets.keys()):
        lengths.add(len(key.split(',')))
        kfreq.setdefault(len(key.split(',')), list()).append({key,freqItemSets[key]})

    # uncomment below line to print frequent items 
    # print(kfreq)

    # uncomment below lines to print frequent items k-wise 
    for k in lengths:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
        # print(kfreq[k])
        print()

    finish_clock = time()
    print("Time Taken: " + "%.4f" % (finish_clock - start_clock) + " seconds")

    mem_usage = memory_usage_psutil()
    print("Memory used: ",float(mem_usage/(1024*1024))," MB")

if __name__ == "__main__":
    datasets_dirs = ["datasets/test.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[1])