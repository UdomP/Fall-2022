import sys
print(sys.argv)

infile=open(sys.argv[1],'r')
wdcnt = {}
totalwd = 0
diffwd = 0

for w in infile.read().split():
	w.lower()
	if w.lower() not in wdcnt:
		wdcnt[w.lower()] = 1
		diffwd += 1
	else:
		wdcnt[w.lower()] += 1
	totalwd += 1

for k in  wdcnt:
	print(k,wdcnt[k])

print("Total Words = " , totalwd)
print("Total Different words = " , diffwd)
print('Word which has more than 4 characters and ends with the suffix "in"')
for wdEnds in sorted(wdcnt, key = wdcnt.get, reverse = True):
	if len(wdEnds) > 4 and wdEnds.endswith('in'):
		print(wdEnds, wdcnt[wdEnds])

