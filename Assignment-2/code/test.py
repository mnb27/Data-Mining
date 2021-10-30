# import numpy as np

# def centeroidnp(arr):
#     length, dim = arr.shape
#     return np.array([np.sum(arr[:, i])/length for i in range(dim)])

# data = [[0,1],[1,2]]
# print(centeroidnp(np.array(data)))









# label = [4, 0, 2, 3, 4, 4, 1, 3, 4, 4, 2, 4, 4]
# gender = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# # for i in range(len(label)):
# #     print(label[i]," -- ",gender[i])

# freqMale = {}
# freqFemale = {}
# for item in label:
#     freqMale[item] = 0
#     freqFemale[item] = 0

# for i in range(len(label)):
#     item = label[i]
#     if gender[i]==1.0:
#         freqMale[item] += 1
#     else :
#         freqFemale[item] += 1

# ratioInEachCluster = list()
# for id in list(set(label)):
#     a = freqMale[id]
#     b = freqFemale[id]
#     if(b!=0): ratioInEachCluster.append(a/b)
#     else: ratioInEachCluster.append(float('inf'))
#     # print(id," --- ",freqFemale[id]," --- ",freqMale[id])
#     # print()
# print(ratioInEachCluster)
# balance = min(ratioInEachCluster)
# print("Balance: ",balance)
