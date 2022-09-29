def findUnique(A):
    u = 0
    l = []
    for nI, i in enumerate(A):
        for nJ, j in enumerate(A):
            if (nI != nJ):
                if i == j:
                    u = 1
                    break
        if u == 0:
            l.append(i)
        u = 0
    return l

A = [10, 3, 2, 8, 10, 3, 10, 10, 99]

print(A)
print("Unique number in List")
print(findUnique(A))