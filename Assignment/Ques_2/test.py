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


# flist = [{'1'}, {'2'}, {'3'}, {'4'}, {'5'}, {'1', '2'}, {'1', '3'}, {'1', '5'}, {'3', '2'}, {'4', '2'}, {'5', '2'}, {'1', '3', '2'}, {'1', '5', '2'}]

# kfreq = dict()
# lengths = set()
# for item in flist:
#     lengths.add(len(item))
#     kfreq.setdefault(len(item), list()).append(item)

# # # uncomment below lines to print frequent items k-wise 
# for k in lengths:
#     print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
#     print(kfreq[k])
#     print()

flist = [[{('11788',): 17599}, {('36308',): 13885}, {('27102',): 4230}, {('43338',): 4257}, {('36306',): 7610}, {('35918',): 9104}, {('11776',): 9551}, {('64866',): 4547}, {('37996',): 5315}, {('43336',): 6818}, {('11774',): 4511}, {('36904',): 5313}, {('64858',): 5374}], [{('36308', '11788'): 6523}, {('35918', '11788'): 4912}, {('11776', '11788'): 5469}, {('36308', '36306'): 4461}]]

temp = list()
for i in flist:
    for j in i:
        temp.append(j)
print(temp)

# for i in temp:
#     print(len(list(list(i.keys())[0])))

kfreq = dict()
lengths = set()
for item in temp:
    Size = len(list(list(item.keys())[0]))
    lengths.add(Size)
    kfreq.setdefault(Size, list()).append(list(list(item.keys())[0]))

# # uncomment below lines to print frequent items k-wise 
for k in lengths:
    print("Count of " + str(k)+"-Frequent Itemsets"+': ',len(kfreq[k]), "---> ")
    print(kfreq[k])
    print()



















