# import pandas as pd
# df = pd.read_csv('spiral_old.csv', delimiter=',')
# # User list comprehension to create a list of lists from Dataframe rows
# list_of_rows = [tuple(row[0:2]) for row in df.values]
# print(list_of_rows[:5])


# Extract data pnts from the file and place in 2D list P
P=[[]]
del(P[0])
with open ('jain.txt') as myfile:
    for line in myfile:
        a=list(map(float, line.split()))
        P.append(a)
a=[]
for row in P:
    a.append(row[0:2])

print(a[:5])