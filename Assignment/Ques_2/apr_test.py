from collections import Counter

globNumberOfTransactions = 0.0
globOriginalList = None
globMinSup = 0

def readFile(filename):
    """ 
    Function read the file 
    return a list of sets which contains the information of the transaction
    """
    originalList = list()
    file = open(filename, 'r')
    c = 0
    for line in file:
        c = c+1
        line = line.strip()
        record = set(line.split(' '))
        originalList.append(record)
    global globNumberOfTransactions 
    globNumberOfTransactions = c
    global globOriginalList
    globOriginalList = originalList
    # print(globOriginalList)
    # print(globNumberOfTransactions)

def getSizeOneItemSet(originalList):
    """this Function generate all the size 1-itemset candidate"""
    Cone = list()
    for s in originalList:
        for e in s:
            Cone.append(e)       
    return sorted(Cone)

def pruneForSizeOne(objectList):
    """this function take in a candidate itemset and filter by support """
    """ K is the result frequent itemset for its size"""
    kDict = dict()
    kList = list()
    a = Counter(objectList)
    for e in a:
        if((a[e]/ float(globNumberOfTransactions)) >= globMinSup):
            kDict.update({e:a[e]})
            c = set([e])
            kList.append(c)
    return kList

def getSizePlueOneItemSet(Klist): 
    """ my way of this doing is super lazy, I just union it and check for size
        and I put the result itemset into the list 
        at the end of the function, I check for its minsup"""
    candidate = list()
    for e in Klist:
        for f in Klist:
            a = e.union(f)
            if len(a) == len(e)+1:
                candidate.append(a)
    #print(candidate)
    #print(len(candidate))
    newlist = []
    for i in candidate:
        if i not in newlist:
            newlist.append(i)
    candidate = newlist
    #print(candidate)
    """ here is the normal pruning process """
    newlist = []
    for e in candidate:
        counter = 0
        for f in globOriginalList:
            if(f.issuperset(e)):
                counter = counter+ 1
        if((counter/float(globNumberOfTransactions)) >= globMinSup):
            newlist.append(e)
    #print(len(candidate))
    return newlist

def apriori(fName):
    candidateList = list()   
    allfrequentitemSet = list()
    readFile(fName)
    Cone = getSizeOneItemSet(globOriginalList)
    candidateList = pruneForSizeOne(Cone)
    for e in candidateList:
        allfrequentitemSet.append(e)
    #print(candidateList)
    while(candidateList !=[]):
        candidateList = getSizePlueOneItemSet(candidateList)
        #print(candidateList)
        for e in candidateList:
             allfrequentitemSet.append(e)
    print("frequent itemsets")          
    print(allfrequentitemSet)
    print("________________________________")
    
        
if __name__ == "__main__":
    
    fName = "datasets/liquor_11frequent.txt"
    
    globMinSup = 0.08
    apriori(fName)