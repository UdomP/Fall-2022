# data = [[1,2,3,4], [5,6,7,8], [8,10,11,12], [13,14,15,16]]

# outfile = open('table.txt', 'w')
# for row in data:
#     for col in row:
#         outfile.write('%18.8f' % col)
#     outfile.write('\n')
# outfile.close()

dic = {'key1': 1, 'key2': 2, 'key3': 3}

dic['key4'] = 4

for k in dic:
    print(k, dic[k])

print('....................')

for k in sorted(dic):
    value = dic[k]
    print(k, value)

print('....................')
print('key1' in dic.keys())
print('key1' in dic.values())
print(1 in dic.keys())
print(1 in dic.values())



if 'key2' in dic:
    print(dic)



t1 = dic
t1['key5'] = 5
print(dic)

t2 = dic.copy()
t2['key2'] = 8
print(t1['key2'])


t1 = {}
t1[0] = -5
t1[1] = 10.5
print(t1)
t2 = [0, 0]
t2[0] = -5
t2[1] = 10.5
print(t2)