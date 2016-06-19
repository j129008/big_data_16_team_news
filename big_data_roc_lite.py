from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import pickle
from pprint import pprint
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc

# My modules
from sklearn.externals.six.moves import zip
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import matplotlib
import numpy
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ----------------------- prepare data -------------- #
# # FIXME change your data path/folder here

train_file = './trainLite' # folder
test_file = './testLite' # folder

pred_fname = './submission_SGDClassifier.csv' # predicitons

# ##########################################

tr_ans = []
tr_vec = []
te_vec = []
te_ids = []

print('load training data ...')

index = dict()
for i in range(3, 24):
    index[i] = dict()

for line in open(train_file,'r'):
    fields = line.split(',')
    fields[-1] = int(fields[-1].strip())
    tr_ans.append(fields[1])
    data = [int(fields[2][4:6])]
    for featureID in range(3,len(fields)):
        try:
            data.append(index[featureID][fields[featureID]])
        except:
            index[featureID][fields[featureID]] = len(index[featureID])
            data.append(index[featureID][fields[featureID]])
    tr_vec.append(data)

print('load testing data ...')

for line in open(test_file):
    fields = line.split(',')
    fields[-1] = int(fields[-1].strip())
    te_ids.append(fields[0])
    data = [int(fields[1][4:6])]
    for featureID in range(2,len(fields)):
        try:
            data.append(index[featureID+1][fields[featureID]])
        except:
            index[featureID+1][fields[featureID]] = len(index[featureID+1])
            data.append(index[featureID+1][fields[featureID]])
    te_vec.append(data)

# pprint(tr_vec)
# pprint(index)
# pprint(te_vec)
# input()
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
# clf = SGDClassifier(loss="log", penalty="l2", n_jobs="4")
clf = RandomForestClassifier(n_jobs=4)
clf.fit(tr_vec[1000000:], tr_ans[1000000:])

# print(cross_validation.cross_val_score(clf, tr_vec, tr_ans, scoring='log_loss'))
# clf_predictions = clf.predict_proba(tr_vec)
# print([x[1] for x in clf_predictions])

# sgd_file = open('sgd_model.pkl','wb')
# pickle.dump(clf,sgd_file)
# sgd_file.close()
# print('save model done')


print('make predictions ...')
# clf_predictions = clf.predict_proba(te_vec)

# print('store predictions in ', pred_fname)
# print(clf_predictions)
# result = ''
# f = open(pred_fname, 'w')
# f.write('id,click\n')
# for x in range(0,len(clf_predictions)):
    # f.write(te_ids[x] + ',' + str(clf_predictions[x][1])+'\n')

clf_predictions = clf.predict_proba(tr_vec[:1000000])
fpr, tpr, thresholds = roc_curve(tr_ans[:1000000], [x[1] for x in clf_predictions], pos_label='1')
fpr_new = []
tpr_new = []
# get_through = 10
# i = int(numpy.random.rand()*get_through)
# while i < len(fpr):
    # fpr_new.append(fpr[i])
    # tpr_new.append(tpr[i])
    # i += get_through

# roc_auc = auc(fpr_new, tpr_new)
roc_auc = auc(fpr, tpr)
plt.xlabel("FPR", fontsize=14)
plt.ylabel("TPR", fontsize=14)
plt.title("ROC Curve / "+'AUC = %0.2f'% roc_auc, fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.plot(fpr,tpr,'b',label='AUC = %0.2f'% roc_auc)
plt.savefig('roc_auc.png')

# print(fpr)
# print(tpr)
