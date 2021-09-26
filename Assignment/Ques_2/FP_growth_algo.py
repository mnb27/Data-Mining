from itertools import combinations
import dataset_info

class NodeStructure():

    def __init__(self, value, count, parent):
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = list()

    def getChild(self, value):
        childNode = None
        flag = False
        for node in self.children:
            if node.value == value:
                childNode = node
                flag = True
                break
        return flag, childNode

    def add_child(self, value):
        newChild = NodeStructure(value, 1, self)
        self.children.append(newChild)
        return newChild

## HELPER FUNCTIONS ######################
def find_frequent_items(transactions, minSup):
    itemSet = dict()
    for transaction in transactions:
        for item in transaction:
            if item not in itemSet:
                itemSet[item] = 1
            else:
                itemSet[item] += 1

    for key, val in list(itemSet.items()):
        if itemSet[key] >= minSup:
            continue
        else:
            del itemSet[key]
    return itemSet

def build_header_table(frequent):
    headers = dict()
    for key in frequent.keys():
        headers[key] = None
    return headers
####################################

class FPTreeStructure():
 
    def __init__(self, transactions, threshold, root_value, root_count):
        
        self.transactions = transactions
        self.frequent = find_frequent_items(transactions, threshold)
        self.headers = build_header_table(self.frequent)
        self.root = self.build_fptree(root_value,root_count, self.frequent, self.headers)

    def build_fptree(self, root_value,root_count, frequent, headers):

        root = NodeStructure(root_value, root_count, None)

        for transaction in self.transactions:
            sorted_items = list()
            for item in transaction:
                if item not in frequent:
                    continue
                else:
                    sorted_items.append(item)
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            if len(sorted_items) != 0:
                self.insert_tree(sorted_items, root, headers)

        return root

    def insert_tree(self, items, node, headers):

        value = items[0]
        flag, child = node.getChild(value)
        if flag==False: 
            # Add new child.
            child = node.add_child(value)

            # Link it to header structure [linked list table of pointers]
            if headers[value] is not None:
                current = headers[value]
                while current.link:
                    current = current.link
                current.link = child
            else:
                headers[value] = child
        else: # child not none --> increment counter
            child.count += 1

        left_items = items[1:]
        if len(left_items) != 0:
            self.insert_tree(left_items, child, headers)

    def isOnePathTree(self, node):

        num_children = len(node.children)
        if num_children <= 1:
            return True
        elif num_children > 1:
            return False
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold):

        if (len(self.root.children) <=1):
            return self.generate_pattern_list()
        elif self.isOnePathTree(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(self, patterns):

        suffix = self.root.value
        if suffix is None:
            return patterns, len(patterns)
        else:
            # We are in a conditional tree.
            new_patterns = dict()
            for key, val in patterns.items():
                ConcatKey = list(key) + [suffix]
                ConcatKey = sorted(ConcatKey)
                new_patterns[tuple(ConcatKey)] = patterns[key]
            return new_patterns, len(new_patterns)

        return None,0

    def generate_pattern_list(self):

        patterns = dict()
        items = self.frequent.keys()
        suffix = self.root.value
        suffix_value = list()
        # If we are in a conditional tree, the suffix is a pattern on its own.
        if suffix is not None:
            suffix_value = [suffix]
            patterns[tuple([suffix])] = self.root.count

        for i in range(len(items)):
            for subset in combinations(items, i+1):
                pattern = tuple(sorted(list(subset) + suffix_value))
                res = int()
                for x in subset:
                    res = min(res, self.frequent[x])
                patterns[pattern] = res

        return patterns, len(patterns)

    def mine_sub_trees(self, threshold):

        patterns = dict()
        mining_order = sorted(self.frequent.keys(), key=lambda x: self.frequent[x])

        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes = list()
            conditional_tree_input = list()
            node = self.headers[item]

            # Follow node links to get a list of all occurrences of a certain item.
            while node:
                # print("Debug")
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item, trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = list()
                parent = suffix.parent

                while parent.parent:
                    # print("Debug")
                    path.append(parent.value)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # Now we have the input for a subtree, so construct it and grab the patterns.
            subtree = FPTreeStructure(conditional_tree_input, threshold, item, self.frequent[item])
            subtree_patterns,lenn = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern not in patterns:
                    patterns[pattern] = subtree_patterns[pattern]
                else:
                    patterns[pattern] += subtree_patterns[pattern]

        return patterns


def find_frequent_patterns(transactions, support_threshold):
    # Find the frequent paterns
    tree = FPTreeStructure(transactions, support_threshold, None, None)
    return tree.mine_patterns(support_threshold)

def main(dataset_path):
    min_support_cnt = 2
    getDataInfo = dataset_info.parse_transaction_dataset(dataset_path)
    DATASET = getDataInfo[0] # Horizonatal Dataset table [Txn x Items]
    print("Dataset Taken :", dataset_path)
    print("Total Transactions :", len(DATASET))
    print("Support Count Taken :",min_support_cnt)
    # Frequent Itemset and Association Rules
    freqItemSets, totalFreqItemS = find_frequent_patterns(DATASET, min_support_cnt)
    # print(freqItemSets)

    sorted_itemsets = dict()
    for k in sorted(freqItemSets, key=len, reverse=True):
        sorted_itemsets[k] = freqItemSets[k]
    print(sorted_itemsets)

    kfreq = dict()
    lengths = set()
    for key in sorted(freqItemSets.keys()):
        lengths.add(len(key))
        kfreq.setdefault(len(key), list()).append({key,freqItemSets[key]})

    # print(kfreq)
    for k in lengths:
        print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
        # print(kfreq[k])
        print(kfreq[k])
        print()
        

# For Testing Purpose
if __name__=="__main__":
    datasets_dirs = ["datasets/te.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[0])