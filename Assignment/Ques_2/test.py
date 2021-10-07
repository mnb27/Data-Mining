# import f

# # f.get_transactions_db_from_dataset("datasets/test.txt")
# flist = [[{'4'}, 2], [{'4', '2'}, 2], [{'5'}, 2], [{'1', '5'}, 2], [{'1', '5', '2'}, 2], [{'5', '2'}, 2], [{'3'}, 6], [{'1', '3'}, 4], [{'3', '1', '2'}, 2], [{'3', '2'}, 4], [{'1'}, 6], [{'1', '2'}, 4], [{'2'}, 7]]

# # kfreq = dict()
# # lengths = set()
# # for item in flist:
# #     lengths.add(len(item[0]))
# #     # kfreq.setdefault(len(item[0]), list()).append(item)
# #     kfreq.setdefault(len(item[0]), list()).append(item[0])

# # # uncomment below line to print frequent items 
# # # print(kfreq)

# # # # uncomment below lines to print frequent items k-wise 
# # for k in lengths:
# #     print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
# #     print(kfreq[k])
# #     print()


flist = [{'1'}, {'2'}, {'3'}, {'4'}, {'5'}, {'1', '2'}, {'1', '3'}, {'1', '5'}, {'3', '2'}, {'4', '2'}, {'5', '2'}, {'1', '3', '2'}, {'1', '5', '2'}]

kfreq = dict()
lengths = set()
for item in flist:
    lengths.add(len(item))
    kfreq.setdefault(len(item), list()).append(item)

# # uncomment below lines to print frequent items k-wise 
for k in lengths:
    print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
    print(kfreq[k])
    print()




















