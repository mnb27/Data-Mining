import a
import eclat_algo
import FP_growth_algo
from datetime import datetime
import matplotlib.pyplot as plt


def compare_algos_given_min_sup(dataset, min_sup, print_result):


    N = len(dataset)


    time_list = []
    algorithms = ['Apriori', 'FP_growth', 'Eclat']
    minSup = 2
    if print_result == True:
        print('min_sup =', min_sup)
    for algo in algorithms:
        if print_result == True:
            print('Algorithm : ', algo)
        time_start = datetime.now()
        if algo == 'Apriori':
            AprioriInst = a.Apriori(dataset[0], N, min_sup)
            freqItemSets = AprioriInst.Apriori_Algo(min_sup/N)
        elif algo == 'FP_growth':
            freqItemSets, totalFreqItemS = FP_growth_algo.find_frequent_patterns(dataset[0], minSup)
        elif algo == 'Eclat':
            freqItemSets, time_end = eclat_algo.Eclat_Algo(dataset[1])
        time_end = datetime.now()
        if print_result == True:
            print('time taken : ' + str( round((time_end-time_start).total_seconds(), 2) ) + ' seconds' )
        else:
            time_list.append(round((time_end-time_start).total_seconds(), 2))
    if print_result == True:
        print('Note : Data Preparation time is not included....... As I assumed we have prepared data\n')
        k = 1
        for F_k in freqItemSets:
            print('length of F_'+str(k)+' =',len(F_k))
            print(F_k)
            k += 1
    else:
        return time_list, len(freqItemSets)


def compare_algos_on_dataset(dataset, min_sup_lower=0.01, min_sup_upper=0.1, step=0.01):
    fig = plt.figure(figsize=(20,5))
    plt1 = fig.add_subplot(121) 
    plt2 = fig.add_subplot(122) 

    min_sup_list = []
    Apriori_list = []
    FP_list = []
    Eclat_list = []
    l_list = []

    min_sup = min_sup_lower
    while(min_sup <= min_sup_upper):
        time_list, l = compare_algos_given_min_sup(dataset, min_sup, False)
        min_sup_list.append(min_sup)
        Apriori_list.append(time_list[0])
        FP_list.append(time_list[1])
        Eclat_list.append(time_list[2])
        l_list.append(l)
        min_sup += step
    plt1.plot(min_sup_list, Apriori_list, label = "Apriori")
    plt1.plot(min_sup_list, FP_list, label = "FP_Growth")
    plt1.plot(min_sup_list, Eclat_list, label = "Eclat")
    plt1.set_xlabel('min_sup') 
    plt1.set_ylabel('time taken in seconds') 
    plt1.set_title('Time taken by different algorithms vs min_sup') 
    plt1.legend() 
    plt2.plot(min_sup_list, l_list)
    plt2.set_xlabel('min_sup') 
    plt2.set_ylabel('length of largest frequent itemset') 
    plt2.set_title('length of largest frequent itemset vs min_sup')
    plt.show() 