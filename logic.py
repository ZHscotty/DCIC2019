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

    # filename = open('feature_importance.csv', encoding='utf-8')
    # features = pd.read_csv(filename)
    # features = features['feature']
    # features = np.array(features)  # np.ndarray()
    # features = features.tolist()  # list
    # features.append('信用分')


    filename = open('train_dataset.csv', encoding='utf-8')
    # train = pd.read_csv(filename, usecols=features)
    train = pd.read_csv(filename)
    traint, testst = train_test_split(train, test_size=0.4, random_state=1)

    print('read success!')

    # features.remove('信用分')
    filename = open('test_dataset.csv', encoding='utf-8')
    # result = pd.read_csv(filename, usecols=features)

    result = pd.read_csv(filename)
    result = result.drop(['用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店'], 1)


    y = traint['信用分']
    # x = traint.drop(['信用分'], 1)
    x = traint.drop(['信用分', '用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)

    y_test = testst['信用分']
    # x_test = testst.drop(['信用分'], 1)
    x_test = testst.drop(['信用分', '用户编码', '用户实名制是否通过核实', '是否大学生客户', '当月是否到过福州山姆会员店', '用户最近一次缴费距今时长（月）', '当月是否逛过福州仓山万达'], 1)




    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.03, n_estimators=1000, silent=False, objective='reg:linear',
                             min_child_weight=6, gamma=0, subsample=0.7, colsample_bytree=0.8, scale_pos_weight=1, seed=27, reg_alpha=110, reg_lambda=15)

    # param_test = {
    #     'learning_rate': [0.01, 0.03, 0.1, 0.3, 1, 10],
    #     'n_estimators': [800, 1000, 2000, 3000]
    # }
    # gsearch1 = GridSearchCV(estimator=model, param_grid=param_test, n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(x, y)

    # cv_result = pd.DataFrame.from_dict(gsearch1.cv_results_)

    # with open('cv_result.csv', 'w') as f:
    #     cv_result.to_csv(f)

    # print('The parameters of the best model are: ')
    # print(gsearch1.best_params_)
    #
    # ans = gsearch1.predict(x_test)
    #
    # ans = [int(i) for i in ans]
    #
    # MAE = (np.abs(ans-y_test)).sum()/len(ans)
    # score = 1/(1+MAE)
    # print(score)

    # model = XGBClassifier(
    #     learning_rate=0.03,
    #     n_estimators=1000,
    #     max_depth=5,
    #     min_child_weight=1,
    #     gamma=0,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     objective='reg:linear',
    #     scale_pos_weight=1,
    #     seed=27)

    # model_param = model.get_xgb_params()
    # xgtrain = xgb.DMatrix(x, label=y)
    # cvresult = xgb.cv(model_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=5,
    #                   metrics='mae', early_stopping_rounds=50)
    # print(cvresult.shape[0])
    # model.set_params(n_estimators=cvresult.shape[0])

    model.fit(x, y, eval_metric='mae')

    ans = model.predict(x_test)

    # ans = model.predict(result)

    ans = [int(i) for i in ans]

    MAE = (np.abs(ans-y_test)).sum()/len(ans)
    score = 1/(1+MAE)
    print(score)
    # print(cvresult.shape[0])

    # filename = open('test_dataset.csv', encoding='utf-8')
    # user = pd.read_csv(filename)
    # user1 = user['用户编码']
    #
    # columns = ['用户编码', '信用分']
    # dataframe = pd.DataFrame({'用户编码': user1, '信用分': ans})
    # dataframe.to_csv("result.csv", index=False, sep=',', encoding='utf-8', columns=columns)
    #
    # print('ok!')


