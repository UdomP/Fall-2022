A = {}
print(type(A))
A['Yu'] = 123456
print(A)
B = dict()
print(type(B))
B['Yu'] = 123456
print(B)
C = list()
print(type(C))
C.append(123456)
print(C)

person = {'James':{'2018':201,'2019':202,'2020':203}, 'Tom':{'2018':103,'2019':102,'2020':101}, 'Jack':{'2018':11,'2019':22,'2020':33}}

print(person)
print(person['James'])
print(person['James']['2019'])
print(person['Tom']['2020'])

# budget = {'':{'2018': ,'2019': ,'2020':}, '':{'2018': ,'2019': ,'2020':}, '':{'2018': ,'2019': ,'2020':}}
infile = open('bodget.csv', 'r')
budget = {}
listW = []
for w in infile.read().split():
    listW.append(w.split(','))
print(listW)
print(listW[0])

for i in range(1, len(listW)):
    temp = {}
    temp[listW[0][1]] = listW[i][1]
    temp[listW[0][2]] = listW[i][2]
    temp[listW[0][3]] = listW[i][3]
    budget[listW[i][0]] = temp

for name in budget:
    sum = 0
    n = 0
    for year in budget[name]:
        sum += int(budget[name][year])
        n += 1
    sum *= (1/n)
    print("average budget for " + name + " is " + str(sum))