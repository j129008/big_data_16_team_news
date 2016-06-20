from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import pickle

# My modules
from sklearn.externals.six.moves import zip

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier

# ----------------------- prepare data -------------- #
# # FIXME change your data path/folder here

train_file = './train' # folder
test_file = './test' # folder

pred_fname = './submission_SGDClassifier_all.csv' # predicitons

# ##########################################

tr_ans = []
tr_vec = []
te_vec = []
te_ids = []

print('load training data ...')

train_file = open(train_file,'r')
train_file.readline()
for line in train_file:
    fields = line.split(',')
    tr_ans.append(fields[1])
    # get hours
    data = [int(fields[2][4:6])]
    for x in range(3,len(fields)):
        data.append(fields[x])
    # remove \n
    data[-1] = data[-1].strip()
    tr_vec.append(data)

print('load testing data ...')

test_file = open(test_file,'r')
test_file.readline()
for line in test_file:
    fields = line.split(',')
    te_ids.append(fields[0])
    data = [int(fields[1][4:6])]
    for y in range(2,len(fields)):
        data.append(fields[y])
    data[-1] = data[-1].strip()
    te_vec.append(data)

print('transform data to index ...')

indexDict = dict()
for i in range(0,len(tr_vec)):
        for j in range(1,len(tr_vec[i])):
                if not(str(j) in indexDict):
                        indexDict[str(j)] = dict()
                        indexDict[str(j)]['all_count'] = 0
                if not(tr_vec[i][j] in indexDict[str(j)]):
                        indexDict[str(j)][tr_vec[i][j]] = indexDict[str(j)]['all_count']
                        indexDict[str(j)]['all_count'] = indexDict[str(j)]['all_count'] + 1

for i in range(0,len(te_vec)):
        for j in range(1,len(te_vec[i])):
                if not(te_vec[i][j] in indexDict[str(j)]):
                        indexDict[str(j)][te_vec[i][j]] = indexDict[str(j)]['all_count']
                        indexDict[str(j)]['all_count'] = indexDict[str(j)]['all_count'] + 1

print('put index in data ...')

for x in range(0,len(tr_vec)):
        for y in range(1,len(tr_vec[x])):
                tr_vec[x][y] = indexDict[str(y)][tr_vec[x][y]]

for x in range(0,len(te_vec)):
        for y in range(1,len(te_vec[x])):
                te_vec[x][y] = indexDict[str(y)][te_vec[x][y]]

# print(tr_vec)
# train_file = open('train_vec.pkl','wb');
# pickle.dump(tr_vec,train_file)
# train_file.close();
# trn_ans = open('train_ans.pkl','wb');
# pickle.dump(tr_ans,trn_ans)
# train_file.close();
# test_file = open('test_vec.pkl','wb');
# pickle.dump(te_vec,test_file)
# train_file.close();
# te_ids_file = open('te_ids.pkl','wb')
# pickle.dump(te_ids,te_ids_file)
# te_ids_file.close()

print('ok')
# ----------------------- just load this time -------------- #

# print('load training data ...')
# train_file = open('train_vec.pkl','r');
# tr_vec = pickle.load(train_file)
# train_file.close();
# print('load labels data ...')
# trn_ans = open('train_ans.pkl','r');
# tr_ans = pickle.load(trn_ans)
# trn_ans.close();
# print('load testing data ...')
# test_file = open('test_vec.pkl','r');
# te_vec = pickle.load(test_file)
# test_file.close();
# print('load testing data ...')
# te_ids_file = open('te_ids.pkl','r');
# te_ids = pickle.load(te_ids_file)
# te_ids_file.close();

print('build SGD classifier ...')
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(tr_vec, tr_ans)

print('make predictions ...')
clf_predictions = clf.predict(te_vec)

print('store predictions in <%s>', pred_fname)
# print(clf_predictions)
result = ''
for x in range(0,len(clf_predictions)):
    result = result  +  te_ids[x] + ',' + str(clf_predictions[x]) + '\n'
with open(pred_fname, 'w') as f:
    f.write(result)
