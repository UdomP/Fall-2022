from tkinter.tix import COLUMN


infile=open('averageRain.txt','r+')

for w in infile.readlines():
    print(w.split('\n')[0])
infile.close()

infile=open('averageRain.txt','r+')
for w in infile.read().split('\n'):
    print(w)

infile.close()

data = [[1,2,3,4], [5,6,7,8], [8,10,11,12], [13,14,15,16]]

outfile = open('table.txt', 'w')
for row in data:
    for col in row:
        outfile.write('%18.8f' % col)
    outfile.write('\n')
outfile.close()