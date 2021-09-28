from itertools import combinations
import dataset_info

class NodeStructure:

    def __init__(self, val, counter, parent, link, children, printIT=False):
        self.val = val
        self.counter = counter
        self.parent = parent
        self.link = link
        self.children = children
        self.printIT = printIT

    def getChild(self, val, resChildNode):
        # childNode = None
        childNode = resChildNode
        flag = False
        for node in self.children:
            if node.val == val:
                childNode = node
                flag = True
                break
        if(self.printIT): print("Inside getChild debug")
        return flag, childNode

    def addChild(self, val, counter):
        newChild = NodeStructure(val, counter=counter, parent=self, link=None, children=list())
        self.children.append(newChild)
        return newChild, True


class FPTreeStructure:
 
    def __init__(self, dataset, threshold, root_val, root_counter):
        
        self.dataset = dataset
        self.frequent, delItemSets = self.find_frequent_items(dataset, threshold, itemSet=dict())
        # print(delItemSets)

        #initialize header table
        headers = dict()
        for key in (self.frequent).keys():
            headers[key] = None
        self.headers = headers

        self.root = self.build_fptree(root_val,root_counter, self.frequent, self.headers)

    def find_frequent_items(self, dataset, minSup, itemSet):
        # itemSet = dict()
        delItemSets = list()
        for transaction in dataset:
            for item in transaction:
                if item not in itemSet:
                    itemSet[item] = 1
                else:
                    itemSet[item] += 1

        for key, val in list(itemSet.items()):
            if itemSet[key] >= minSup:
                continue
            else:
                delItemSets.append(itemSet[key])
                del itemSet[key]
        return itemSet, delItemSets

    def build_fptree(self, root_val, root_counter, frequent, headers):

        root = NodeStructure(val=root_val, counter=root_counter, parent=None, link=None, children=list())
        # root = NodeStructure(root_val, root_counter, parent)

        for transaction in self.dataset:
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

        val = items[0]
        flag, child = node.getChild(val=val, resChildNode=None)
        if flag==False: 
            # Add new child.
            child, boolFlag = node.addChild(val=val, counter=1)

            # Link it to header structure [linked list table of pointers]
            if headers[val] is not None:
                current = headers[val]
                while current.link:
                    current = current.link
                current.link = child
            else:
                headers[val] = child
        else: # child not none --> increment counter
            child.counter += 1

        left_items = items[1:]
        if len(left_items) != 0:
            self.insert_tree(left_items, child, headers)

        return "successfully inserted"

    def isOnePathTree(self, node):

        num_children = len(node.children)
        if num_children <= 1:
            return True
        elif num_children > 1:
            return False
        else:
            return True and self.isOnePathTree(node.children[0])
        return "end of return"

    def zip_patterns(self, patterns):

        suffix = self.root.val
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
        suffix = self.root.val
        suffix_val = list()
        # If we are in a conditional tree, the suffix is a pattern on its own.
        if suffix is not None:
            suffix_val = [suffix]
            patterns[tuple([suffix])] = self.root.counter

        for i in range(1, len(items) + 1):
            for subset in combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_val))
                # patterns[pattern] = min([self.frequent[x] for x in subset])
                res = float('inf')
                # res = int()
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
                frequency = suffix.counter
                path = list()
                parent = suffix.parent

                while parent.parent:
                    # print("Debug")
                    path.append(parent.val)
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

    def mine_patterns(self, threshold):

        if (len(self.root.children) <=1):
            return self.generate_pattern_list()
        elif self.isOnePathTree(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))
        return "end of return"

class FP_Growth_Algo:
    def __init__(self, dataset, min_sup_count):
        self.dataset = dataset
        self.min_sup_count = min_sup_count

    def fpg_Algo(self):
        # Find the frequent paterns
        root_val = None
        root_counter = None
        tree = FPTreeStructure(self.dataset, self.min_sup_count, root_val, root_counter)
        return tree.mine_patterns(self.min_sup_count)

def main(dataset_path):
    min_support_cnt = 2
    # min_support_cnt = int(input())
    
    getDataInfo = dataset_info.parse_transaction_dataset(dataset_path)
    DATASET = getDataInfo[0] # Horizonatal Dataset table [Txn x Items]
    print("Dataset Taken :", dataset_path)
    print("Total dataset :", len(DATASET))
    print("Support counter Taken :",min_support_cnt)

    FPGrowthInst = FP_Growth_Algo(DATASET, min_support_cnt)
    freqItemSets, totalFreqItemS = FPGrowthInst.fpg_Algo()
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
        print("counter of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
        # print(kfreq[k])
        print(kfreq[k])
        print()
        

# For Testing Purpose
if __name__=="__main__":
    datasets_dirs = ["datasets/te.txt", "datasets/chess.txt", "datasets/liquor_11frequent.txt", 
                 "datasets/t20i6d100k.txt", "datasets/BMS2.txt"]
    main(datasets_dirs[0])