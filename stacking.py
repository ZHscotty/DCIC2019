##基模型的参数还不完善

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import svm
np.set_printoptions(suppress=True)


def Load_Data():
    train = pd.read_csv('./data/train_dataset.csv', encoding='utf8')
    test = pd.read_csv('./data/test_dataset.csv', encoding='utf8')
    return train, test


def Data_Pro(dataset, features, flag):
    data_x = dataset.drop(features, 1)
    if flag == 0:
        data_y = dataset['信用分']
        data_x = data_x.drop(['信用分'], 1)
        # regularit(data_x)
        return data_x, data_y
    if flag == 1:
        # regularit(data_x)
        return data_x


def Model_Base():
    model1 = xgb.XGBRegressor(max_depth=5, learning_rate=0.02, n_estimators=1500, silent=False, objective='reg:linear', min_child_weight=6, gamma=0,
                              subsample=0.7, colsample_bytree=0.8, scale_pos_weight=1, seed=27, reg_alpha=110, reg_lambda=15)
    model2 = svm.SVR()
    model3 = RandomForestRegressor()
    model4 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5))
    model5 = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=5))

    return [model1, model2, model3, model4, model5]


def Five_Fold(Train, model, features):
    train1 = Train[:10000]
    train2 = Train[10000:20000]
    train3 = Train[20000:30000]
    train4 = Train[30000:40000]
    train5 = Train[40000:50000]

    #1
    x1 = pd.concat([train2, train3, train4, train5], axis=0, ignore_index=True)
    x1, Label = Data_Pro(x1, f, 0)
    validation1 = train1
    validation1, no_need = Data_Pro(validation1, f, 0)
    model.fit(x1, Label)
    predict1 = model.predict(validation1)
    predict1 = pd.DataFrame(predict1)
    test_pre1 = model.predict(testset)
    test_pre1 = pd.DataFrame(test_pre1)

    #2
    x2 = pd.concat([train1, train3, train4, train5], axis=0, ignore_index=True)
    x2, Label = Data_Pro(x2, f, 0)
    validation2 = train2
    validation2, no_need = Data_Pro(validation2, f, 0)
    model.fit(x2, Label)
    predict2 = model.predict(validation2)
    predict2 = pd.DataFrame(predict2)
    test_pre2 = model.predict(testset)
    test_pre2 = pd.DataFrame(test_pre2)

    #3
    x3 = pd.concat([train1, train2, train4, train5], axis=0, ignore_index=True)
    x3, Label = Data_Pro(x3, f, 0)
    validation3 = train3
    validation3, no_need = Data_Pro(validation3, f, 0)
    model.fit(x3, Label)
    predict3 = model.predict(validation3)
    predict3 = pd.DataFrame(predict3)
    test_pre3 = model.predict(testset)
    test_pre3 = pd.DataFrame(test_pre3)

    #4
    x4 = pd.concat([train1, train2, train3, train5], axis=0, ignore_index=True)
    x4, Label = Data_Pro(x4, f, 0)
    validation4 = train4
    validation4, no_need = Data_Pro(validation4, f, 0)
    model.fit(x4, Label)
    predict4 = model.predict(validation4)
    predict4 = pd.DataFrame(predict4)
    test_pre4 = model.predict(testset)
    test_pre4 = pd.DataFrame(test_pre4)

    #5
    x5 = pd.concat([train1, train2, train3, train4], axis=0, ignore_index=True)
    x5, Label = Data_Pro(x5, features, 0)
    validation5 = train5
    validation5, no_need = Data_Pro(validation5, f, 0)
    model.fit(x5, Label)
    predict5 = model.predict(validation5)
    predict5 = pd.DataFrame(predict5)
    test_pre5 = model.predict(testset)
    test_pre5 = pd.DataFrame(test_pre5)


    predict = pd.concat([predict1, predict2, predict3, predict4, predict5], axis=0, ignore_index=True)
    test_pre = (test_pre1+test_pre2+test_pre3+test_pre4+test_pre5)/5

    return predict, test_pre

if __name__ == '__main__':
    #加载数据
    Train, Test = Load_Data()

    #数据处理
    f = ['用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）',
                '当月是否逛过福州仓山万达', '是否黑名单客户', '当月飞机类应用使用次数']
    # Train, Label = Data_Pro(Train, features, 0)
    T, L = Data_Pro(Train, f, 0)
    testset = Data_Pro(Test, f, 1)
    models = Model_Base()
    # kf = KFold(n_splits=5, shuffle=False)
    A1, B1 = Five_Fold(Train, models[0], f)
    A2, B2 = Five_Fold(Train, models[1], f)
    A3, B3 = Five_Fold(Train, models[2], f)
    A4, B4 = Five_Fold(Train, models[3], f)
    A5, B5 = Five_Fold(Train, models[4], f)

    A = pd.concat([A1, A2, A3, A4, A5], axis=1, ignore_index=True)
    B = pd.concat([B1, B2, B3, B4, B5], axis=1, ignore_index=True)

    print(A.shape)
    print(B.shape)

    model = LinearRegression()
    model.fit(A, L)
    ans = model.predict(B)
    ans = [int(i) for i in ans]

    filename = open('./data/test_dataset.csv', encoding='utf-8')
    user = pd.read_csv(filename)
    user1 = user['用户编码']
    columns = ['用户编码', '信用分']
    dataframe = pd.DataFrame({'用户编码': user1, '信用分': ans})
    dataframe.to_csv("./data/result.csv", index=False, sep=',', encoding='utf-8', columns=columns)
    print('ok!')



