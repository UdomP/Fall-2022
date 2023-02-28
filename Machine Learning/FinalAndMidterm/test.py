A = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

B = []
for i in range(len(A[0])):
    temp = []
    for j in range(len(A)):
        temp.append(A[j][i])
    B.append(temp)

print(B)