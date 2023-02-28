
def readFile(fileName, argDic):
    infile = open(fileName, 'r')
    dic = argDic
    keyList = []
    for w in infile.read().split('\n'):
        ww = w.split(',')
        if 'CSU ID' in ww:
            keyList = ww
        else:
            if ww[0] == '':
                break
            if ww[0] not in dic:
                dic[ww[0]] = {}
            for keyIndex in range(1, len(keyList)):
                try:
                    dic[ww[0]][keyList[keyIndex]] = ww[keyIndex]
                except:
                    continue
    return dic

newDic = {}

studentDict = readFile('student.csv', newDic)
gpaDict = readFile('gpa.csv', studentDict)
balanceDict = readFile('balance.csv', gpaDict)
hometownDict = readFile('hometown.csv', balanceDict)

newNewDict = {}
studentDictNew = readFile('student.csv', newNewDict)

resultFile = open('result.csv', 'w')

topStr = 'CSU ID'
for key in studentDictNew:
    for keykey in hometownDict[key]:
        topStr += ',' + keykey
    break
resultFile.write(topStr)

for key in studentDictNew:
    tempString = '\n' + key
    for keykey in hometownDict[key]:
        tempString += ',' + hometownDict[key][keykey]
    resultFile.write(tempString)