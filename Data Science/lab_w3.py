# def s(n):
#     i = 1
#     result = 0
#     while i <= n:
#         result =+ i**2
#         i+=1
#     print(result)


Cdegrees = range(-20, 41, 5)
Fdegrees = [(9.0/5)*C + 32 for C in Cdegrees]
for i in Cdegrees:
    print(i)
table1 = [Cdegrees, Fdegrees]  # list of two lists
# table2 = zip(Cdegrees, Fdegrees)

# print(type(table2))
# for x in table2:
#     print(x)

# print(type(Cdegrees))
# for x in Cdegrees:
#     print(x)
# print(Cdegrees)
# print(Fdegrees)
# print (table1[0])     # the Cdegrees list
# print (table1[1])     # the Fdegrees list
# print(table1[0][3])
# print (table1[1][2])  # the 3rd element in Fdegree)

# A = [2, 3.5, 8, 10]

# print(A[2:])
# print(A[1:3])
# print(A[:3])
# print(A[1:-1])
# print(A[:])


table2 = []
for C, F in zip(Cdegrees, Fdegrees):
    row = [C, F]
    table2.append(row)

# more compact with list comprehension:
table2 = [[C, F] for C, F in zip(Cdegrees, Fdegrees)]
for C, F in table2[Cdegrees.index(10): Cdegrees.index(35)]:
    print('%5.0f %5.1f' % (C,F))