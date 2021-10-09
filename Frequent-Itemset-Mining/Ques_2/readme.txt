## 2018CSB1069 - Aman Bilaiya
## README DOC FOR QUES 2 - Frequent Itemset Mining

Install dependencies if not :- 
$ pip install numpy
$ pip install psutil
$ pip install matplotlib

Directory contains :-
1) a) apriori_algo.py --> Implementation of Apriori Algorithm with some pruning.
   b) aprioriOpt.py is Optimized version of apriori using extended prefix tree similar to as given in Zaki's book but somehow 
   it is taking too much time for large datasets, so I have used apriori_algo.py for obtaining outputs and results.

2) eclat_algo.py --> Implementation of Eclat Algorithm with D-Eclat optimization similar to as given in Zaki's Book.

3) fpgrowth_algo.py --> Implementation of FP Growth Algorithm similar to as given in Zaki's Book with some basic code optimizations.

4) dataset_info.py--> Script to parse dataset and extract useful information

5) Datasets folder containing chess, liquor,BMS2 and t20i6d100k(TD) datasets

6) “plottingComparisions.ipynb” --> To plot graphs minSupport vs Execution Time

_________________________________HOW TO RUN CODE___________________________________
$ python filename.py

To write output in a file:
$ python filename.py > output.txt 

One can set the min Support value and dataset file by changing the variable in the main function of
each of the algorithm.

NOTE : Ensure that working directory is same as path of "Ques_2 folder"


OUTPUTS :
Refer "outputs" folder where you can find outputs[Frequent Itemsets, Time Taken, Memory Used] for all 3 algorithms for different
datasets and min Support value being used. Naming Convention Used : <algo>_<datasetName>_<minSupport>.txt

The graphs are plotted using “plottingComparisions.ipynb”.

For more info : "Refer Report_Ques2.pdf"