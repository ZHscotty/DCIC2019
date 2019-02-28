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

    #读入测试数据
    filename = open('train_dataset.csv', encoding='utf-8')
    train = pd.read_csv(filename)
    
    #分割数据
    traint, testst = train_test_split(train, test_size=0.4, random_state=1)

    #读入测试数据
    filename = open('test_dataset.csv', encoding='utf-8')
    result = pd.read_csv(filename)
    result = result.drop(['用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店'，'用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)

    #处理数据，删除没用的特征
    y = traint['信用分']
    x = traint.drop(['信用分', '用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)

    y_test = testst['信用分']
    x_test = testst.drop(['信用分', '用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)

    #最后用所有数据进行训练
    y_t = train['信用分']
    x_t = train.drop(['信用分', '用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)

    #使用xbgboost的线性回归模型（参数已调，调参代码见）
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.03, n_estimators=1000, silent=False, objective='reg:linear',
                             min_child_weight=6, gamma=0, subsample=0.7, colsample_bytree=0.8, scale_pos_weight=1, seed=27, reg_alpha=110, reg_lambda=15)

    #模型的训练
    model.fit(x_t, y_t, eval_metric='mae')

    #进行预测
    ans = model.predict(result)

    #结果处理
    ans = [int(i) for i in ans]

    #结果输出
    filename = open('test_dataset.csv', encoding='utf-8')
    user = pd.read_csv(filename)
    user1 = user['用户编码']
    
    columns = ['用户编码', '信用分']
    dataframe = pd.DataFrame({'用户编码': user1, '信用分': ans})
    dataframe.to_csv("result.csv", index=False, sep=',', encoding='utf-8', columns=columns)
    
    print('ok!')


