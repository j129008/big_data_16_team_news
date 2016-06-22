train_file = './trainLite'
test_file = './testLite'

train_out = './data/trainOut'
test_out = './data/testOut'


index = dict()
counter = dict()
for i in range(2, 24):
    index[i] = dict()
    counter[i] = dict()

print('build feature counter...')
# counter
for line in open(train_file,'r'):
    fields = line.split(',')
    fields[-1] = int(fields[-1].strip())
    fields[2] = int(fields[2][6:8])
    for featureID in range(2,len(fields)):
        featureType = fields[featureID]
        try:
            counter[featureID][featureType] += 1
        except:
            counter[featureID][featureType] = 1

# train
print('indexing training data...')
fOut = open(train_out, 'w')
for line in open(train_file,'r'):
    fields = line.split(',')
    fields[-1] = int(fields[-1].strip())
    fields[2] = int(fields[2][6:8])
    countAll = 0
    data = ''
    for featureID in range(2,len(fields)):
        try:
            data += str(featureID-1) + ':'+str(index[featureID][fields[featureID]]) + ' '
        except:
            index[featureID][fields[featureID]] = len(index[featureID])
            data += str(featureID-1) + ':'+str(index[featureID][fields[featureID]]) + ' '
        if featureID == 2 or featureID == 11 or featureID ==12:
            countAll += counter[featureID][fields[featureID]]
    fOut.write( str(fields[1]) + ' ' + data + '23:'+str(countAll) + '\n' )
fOut.close()

print('indexing testing data...')
# test
fOut = open(test_out, 'w')
for line in open(test_file):
    fields = line.split(',')
    fields[-1] = int(fields[-1].strip())
    fields[1] = int(fields[1][6:8])
    countAll = 0
    data = ''
    for featureID in range(1,len(fields)):
        try:
            data += str(featureID)+':'+str(index[featureID+1][fields[featureID]]) + ' '
        except:
            index[featureID+1][fields[featureID]] = len(index[featureID+1])
            data += str(featureID) + ':' + str(index[featureID+1][fields[featureID]]) + ' '
        if featureID == 1 or featureID == 10 or featureID ==11:
            try:
                countAll += counter[featureID+1][fields[featureID]]
            except:
                pass
    fOut.write( data + ' ' + '23:' + str(countAll) + '\n' )
fOut.close()
