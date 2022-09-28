def fives(n):
    i = 1
    s = []
    while(n >= i):
        if((i % 5) == 0):
            # print(i, end=', ')
            s.append(i)
        i += 1
    print(*s, sep=', ')

print('X = ' + str(15))
fives(15)
print('X = ' + str(26))
fives(26)