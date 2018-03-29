#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pickle
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans



sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### 数据集背景
i = 0
for key, values in data_dict.items():
    if i < 2:
        print key
        pprint.pprint(values)
        i += 1

my_data = data_dict
my_df = pd.DataFrame.from_dict(my_data, orient = "index")
print my_df.head()

# 数据点总数及特征数量(包括'poi'在内)
print len(my_df)
print len(my_df.columns)
print my_df.info()

# POI数和非POI数
print len(my_df[my_df['poi'] == True])
print len(my_df[my_df['poi'] == False])

# 哪些特征含有缺失值
print my_df.info()
my_df = my_df.replace('NaN',np.nan)
print my_df.info()

# 将特征按财务（payment、stock）数据、邮箱数据排序
pm_list = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', \
           'deferral_payments','loan_advances', 'other', 'expenses', 'director_fees', 'total_payments']
st_list = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
em_list = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']
my_df = my_df.loc[:, ['poi']+pm_list+st_list+em_list]
print my_df.info()

### 探索数据问题
## 填充财务数据缺失值(原始财务数据中missing value视为0)
fn_df = my_df[pm_list+st_list].fillna(0)
print fn_df.info()

em_df = my_df[em_list]

my_df = pd.concat([my_df['poi'],fn_df, em_df], axis = 1)
my_df.head()


## 填充邮件数据缺失值
# poi和非poi分开，可视化看看分布，决定用什么填充各列的缺失值比较合适
em_poi = my_df[my_df['poi'] == True][em_list]
em_notpoi = my_df[my_df['poi'] == False][em_list]

plt.hist(em_poi['to_messages'], 50, range = (em_poi['to_messages'].min(), em_poi['to_messages'].max()))
plt.show()
plt.hist(em_notpoi['to_messages'], 50, range = (em_notpoi['to_messages'].min(), em_notpoi['to_messages'].max()))
plt.show()

plt.hist(em_poi['to_messages'], 50, range = (em_poi['to_messages'].min(), em_poi['to_messages'].max()))
plt.show()
plt.hist(em_notpoi['to_messages'], 50, range = (em_notpoi['to_messages'].min(), em_notpoi['to_messages'].max()))
plt.show()

plt.hist(em_poi['shared_receipt_with_poi'], 50, range = (em_poi['shared_receipt_with_poi'].min(), em_poi['shared_receipt_with_poi'].max()))
plt.show()
plt.hist(em_notpoi['shared_receipt_with_poi'], 50, range = (em_notpoi['shared_receipt_with_poi'].min(), em_notpoi['shared_receipt_with_poi'].max()))
plt.show()

# 上面三组图可以看出，poi和非poi，邮件数据分布有较大差异，由于分布都不均匀，所以用各自的中位数进行填充
em_poi = em_poi.fillna(em_poi[em_list].median())
em_notpoi = em_notpoi.fillna(em_notpoi[em_list].median())

em_df = em_poi.append(em_notpoi)
print em_df.info()

## 原始财务数据是否有问题？
# 找出财务数据图中异常值点
plt.scatter(my_df['salary'],my_df['bonus'])
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

print my_df[my_df['salary'] > 2.5 * (10**7)]

# 去除财务数据异常值点
my_df = my_df.drop('TOTAL')

plt.scatter(my_df['salary'],my_df['bonus'])
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

# 财务数据表格中，payments和stock总数是否正确？
def total(my_df, my_list):
    total = 0
    for i in range(len(my_list)-1):
        total = total + my_df[my_list[i]]
    return total

print my_df[pm_list][my_df['total_payments'] != total(my_df, pm_list)]
print my_df[st_list][my_df['total_stock_value'] != total(my_df, st_list)]

# 对比原始财务数据，发现字典中有两个人的值人工输入错误。下面将其调整正确：
print my_df[pm_list+st_list].loc['BELFER ROBERT',:]

v1 = my_df[pm_list+st_list].loc['BELFER ROBERT',:].values
n_v1 = list(v1)[1:] + [0.0]
my_df.loc['BELFER ROBERT','salary':'total_stock_value'] = n_v1

v2 = my_df[pm_list+st_list].loc['BHATNAGAR SANJAY',:].values
n_v2 = [0.0] + list(v2)[:-1]
my_df.loc['BHATNAGAR SANJAY','salary':'total_stock_value'] = n_v2

print my_df[pm_list+st_list].loc['BELFER ROBERT',:]
print my_df[pm_list+st_list].loc['BHATNAGAR SANJAY',:]

# 检查是否已经修正完毕
print my_df[my_df['total_payments'] != total(my_df, pm_list)]

# 原始财务数据中的非人名：“THE TRAVEL AGENCY IN THE PARK”，需要删除
my_df = my_df.drop('THE TRAVEL AGENCY IN THE PARK')

# 原始财务数据中值全为NaN的行'LOCKHART EUGENE E',需要删除
print my_df.loc['LOCKHART EUGENE E',:]
my_df = my_df.drop('LOCKHART EUGENE E')

# 找出邮件数据中的异常值点
plt.scatter(my_df['to_messages'],my_df['from_messages'])
plt.xlabel('to_messages')
plt.ylabel('from_messages')
plt.show()

print my_df[(my_df['to_messages'] > 10000) | (my_df['from_messages'] > 10000)]

# 以上三个人名可能是异常值，但这里不太好确定，可以先忽略
# 清除了明显的问题数据,之后再对异常值进行探索。



###  Remove outliers
### 假设非poi中所有数值低于5%和高于95%的，为异常值。
# 寻找非poi里的异常值，避免非poi里的异常数据影响机器学习算法的决策
print my_df['poi'].value_counts()
notpoi_df = my_df[my_df['poi'] == False]
qt1 = notpoi_df.quantile(q=0.05, axis=0)
print qt1
qt2 = notpoi_df.quantile(q=0.95, axis=0)
print qt2
outliers = notpoi_df[(notpoi_df < qt1) | (notpoi_df > qt2)].count(axis=1)
print outliers.sort_values(axis=0, ascending=False).head(10)

# 删除前四个异常值较多的非poi，以降低他们其他非poi数据的影响
my_df.drop(['FREVERT MARK A', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G', 'BAXTER JOHN C'],\
           axis=0, inplace=True)

# 最终的df信息
print my_df.info()
print my_df['poi'].value_counts()


### Task 3: Create new feature(s) 特征选择与优化

# 运用直觉，手动创建新特征
# 直觉1：奖金比薪水的比例更高的，倾向于是poi
# 直觉2：收到的邮件中，来自poi的比例更高的，倾向于是poi
# 直觉3：发出的邮件中，发给poi的比例更高的，倾向于是poi
my_df['bonus_to_salary'] = my_df['bonus']/my_df['salary']
my_df['to_poi_ratio'] = my_df['from_poi_to_this_person'] / my_df['to_messages']
my_df['from_poi_ratio'] = my_df['from_this_person_to_poi'] / my_df['from_messages']

# 继续填充空值
my_df = my_df.fillna(0)
print my_df.info()

# 目前为止，将我的数据框装换成数据字典，查看数据是否正常
my_dataset = my_df.to_dict(orient = 'index')

i = 0
for key, values in my_dataset.iteritems():
    if i < 2:
        print key
        print values
        i += 1

# 特征缩放，使用最大最小归一化，将所有财务数据值控制在0到1的范围内
from sklearn.preprocessing import MinMaxScaler
sc_df = my_df.copy()
scaler = MinMaxScaler()
sc_df[pm_list+st_list] = scaler.fit_transform(sc_df[pm_list+st_list])

### Store to my_dataset for easy export below.将缩放后的特征存入数据字典，查看新的数据是否正常
my_dataset = sc_df.to_dict(orient = 'index')

i = 0
for key, values in my_dataset.iteritems():
    if i < 2:
        print key
        print values
        i += 1

### 初步人工选择特征
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','from_poi_to_this_person','from_this_person_to_poi']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### 初步特征建模
import tester

clf = GaussianNB()
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()

clf = DecisionTreeClassifier()
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()


clf = SVC(kernel="rbf", C = 10000)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()


clf = AdaBoostClassifier(n_estimators=10)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()


clf = KNeighborsClassifier(n_neighbors=2)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()


clf = KMeans(n_clusters=2)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()

### 使用手动创建的特征再次建模
features_list = ['poi','bonus_to_salary','to_poi_ratio','from_poi_ratio']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf = GaussianNB()
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()


clf = DecisionTreeClassifier()
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()


clf = SVC(kernel="rbf", C = 10000)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()


clf = AdaBoostClassifier(n_estimators=10)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()

clf = KNeighborsClassifier(n_neighbors=2)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()

clf = KMeans(n_clusters=2)
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print tester.main()

### 算法辅助选择特征 ———— 使用所有特征，创建决策树和AdaBoost模型(上面两次批量创建模型，结果都显示出决策树和adaboost的方法最有效)
features_list = features_list+pm_list+st_list+em_list

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf_tree = DecisionTreeClassifier()
tester.dump_classifier_and_data(clf_tree, my_dataset, features_list)
tester.main()


clf_ada = AdaBoostClassifier(n_estimators=10)
tester.dump_classifier_and_data(clf_ada, my_dataset, features_list)
tester.main()

# 从决策树模型中查看最重要的特征
clf_tree.fit(features, labels)
tree_ft_importances = clf_tree.feature_importances_

tree_features = zip(tree_ft_importances, features_list[1:])

tree_ft_rank = sorted(tree_features, key= lambda x:x[0], reverse=True)
print tree_ft_rank

# 从adaboost模型中查看最重要的特征
clf_ada.fit(features, labels)
tree_ft_importances = clf_ada.feature_importances_

tree_features = zip(tree_ft_importances, features_list[1:])

tree_ft_rank = sorted(tree_features, key= lambda x:x[0], reverse=True)
print tree_ft_rank

# 用全部特征建模，两个建模方法都获得了超过0.6的F1分数。
# 然而两种模型得出的重要特征排序却不是完全一致的，通过对比，发现：
# 'from_poi_ratio'，'shared_receipt_with_poi'，'expenses'，'to_poi_ratio'，'to_messages'是在两个模型中都比较重要。
# 为了防止过拟合问题，对于决策树模型，可以选择5个特征；对于adaboost，可以选择7个特征。

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

### 决策树的特征选择
tree_pipe = Pipeline([
    ('select_features', SelectKBest(k=5)),
    ('clf', DecisionTreeClassifier()),
])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
### 5、参数调整，6、模型验证

# 决策树的参数调整：这里使用F1分数为评估分数，因为f1分数同事考虑了精度和召回率指标，下同。
param_grid = dict(clf__criterion = ['gini', 'entropy'],
                  clf__min_samples_split = [2,4,6,8,10],
                  clf__max_depth = [None,5,10,15,20],
                  clf__max_features = [None, 'sqrt', 'log2', 'auto'],
                  clf__splitter = ['best','random'])

tree_clf = GridSearchCV(tree_pipe, param_grid = param_grid, scoring='f1', cv=10)
tree_clf.fit(features, labels)

print tree_clf.best_score_
print tree_clf.best_params_

# 决策树模型的最终验证
tree_clf = Pipeline([('select_features', SelectKBest(k=5)),
                     ('clf', DecisionTreeClassifier(criterion='entropy',
                                                         max_depth=15,
                                                         max_features=None,
                                                         min_samples_split=2,
                                                         splitter ='best'))])
tester.dump_classifier_and_data(tree_clf, my_dataset, features_list)
print tester.main()



### adaboost的特征选择
ada_pipe = Pipeline([
    ('select_features', SelectKBest(k=7)),
    ('clf', AdaBoostClassifier()),
])


# adaboost方法的参数调整：
param_grid = dict(clf__base_estimator = [DecisionTreeClassifier(), RandomForestClassifier(), GaussianNB()],
                  clf__n_estimators = [60,80,100,110,120],
                  clf__learning_rate = [0.4,0.6,0.8,0.9,1.0])

ada_clf = GridSearchCV(ada_pipe, param_grid = param_grid, scoring='f1', cv=10)
ada_clf.fit(features, labels)

print ada_clf.best_score_
print ada_clf.best_params_

# adaboost方法的最终验证
ada_clf = Pipeline([('select_features', SelectKBest(k=7)),
                    ('clf', AdaBoostClassifier(
                         base_estimator = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                                                 max_features=None, max_leaf_nodes=None,
                                                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                                                 min_samples_leaf=1, min_samples_split=2,
                                                                 min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                                                 splitter='best'),
                         learning_rate = 0.8,
                         n_estimators = 60))])

tester.dump_classifier_and_data(ada_clf, my_dataset, features_list)
print tester.main()
