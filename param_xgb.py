#--coding=utf-8--
import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':

    # 读入测试数据
    filename = open('train_dataset.csv', encoding='utf-8')
    train = pd.read_csv(filename)

    # 分割数据
    traint, testst = train_test_split(train, test_size=0.4, random_state=1)

    # 读入测试数据
    filename = open('test_dataset.csv', encoding='utf-8')
    result = pd.read_csv(filename)
    result = result.drop(['用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店'], 1)

    # 处理数据，删除没用的特征
    y = traint['信用分']
    x = traint.drop(['信用分', '用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)

    y_test = testst['信用分']
    x_test = testst.drop(['信用分', '用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)

    #定义初始的模型参数
    model = xgb.XGBRegressor(silent=False, objective='reg:linear')

    #定义准备修改的模型参数，此处修改learning_rate，n_estimators两项
    param_test = {
        'learning_rate': [0.01, 0.03, 0.1, 0.3, 1, 10],
        'n_estimators': [800, 1000, 2000, 3000]
    }

    #网格寻优优化参数
    gsearch1 = GridSearchCV(estimator=model, param_grid=param_test, n_jobs=4, iid=False, cv=5)
    gsearch1.fit(x, y)

    #输出最优参数的值，并进行修改
    print('The parameters of the best model are: ')
    print(gsearch1.best_params_)






