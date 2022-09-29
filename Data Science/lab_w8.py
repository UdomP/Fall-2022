data = [[1,2,3,4], [5,6,7,8], [8,10,11,12], [13,14,15,16]]

outfile = open('table.txt', 'w')
for row in data:
    for col in row:
        outfile.write('%18.8f' % col)
    outfile.write('\n')
outfile.close()