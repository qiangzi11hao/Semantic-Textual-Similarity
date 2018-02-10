from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import  train_test_split
from sklearn.metrics import  mean_squared_error
from sklearn.svm import  SVR
from scipy.stats import pearsonr
import pickle
import numpy as np
import time
from FeatureExtract import extract

"""
选用了三种model,random forest,XGboost和Gradient Boosting
"""
def model_random_forest(Xtrain,Xtest,y_train,add):
    X_train = Xtrain
    # print(len(X_train[0]))
    rfr = RandomForestRegressor(n_jobs=-1, random_state=10,oob_score=True)
    param_grid ={'n_estimators': [300]}
    # 'n_estimators': list(r,ange(30,60130)), ,"max_depth":list(range(40,61,10)),"max_features":list(range(120,211,30)),min_samples_split':list(range(20,201,20)),  'min_samples_leaf':list(range(10,80,10))
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring= None, iid=False, cv=10)
    model.fit(X_train, y_train)
    # model.grid_scores_, model.best_params_, model.best_score_
    print('Random forecast classifier...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    y_pred = model.predict(Xtest)
    final1 = model.predict(add)
    return y_pred, final1



def model_GBR(Xtrain,Xtest,y_train,add):
    X_train = Xtrain
    rfr = GradientBoostingRegressor(random_state=10,learning_rate= 0.01,loss= "ls")
    param_grid = {'n_estimators': [300]}
    # 'n_estimators': list(range(30,601,30)), ,"max_depth":list(range(40,61,10)),"max_features":list(range(120,211,30)),min_samples_split':list(range(20,201,20)),  'min_samples_leaf':list(range(10,80,10))
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring= None, iid=False, cv=10)
    model.fit(X_train, y_train)
    print('Best Params:')
    print(model.best_params_)
    y_pred = model.predict(Xtest)
    final2 = model.predict(add)
    return y_pred,final2

def model_XG(Xtrain,Xtest,y_train,add):
    clf = XGBRegressor(nthread=4)
    clf.fit(Xtrain,y_train)
    y_pred = clf.predict(Xtest)
    final3 = clf.predict(add)
    return y_pred,final3


if __name__ == '__main__':
    # extract()
    with open("data.pickle", 'rb') as f:
        X_train, Ytrain, X_test, y_ids = pickle.load(f)

    ##每次只选则一个feature,观察单个feature下的模型表现,调试过程中可用
    #     pea = []
    #     mse = []
    # print(X_train[0])
    # for i in range(len(X_train[0])):
    #     Xtrain, Xtest = X_train[:,i].reshape(-1,1), X_test[:,i].reshape(-1,1)
    #     print(i)
    #     x_train,x_cross,y_train,y_cross = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state= 1021)
    #     # y_pred, final1= model_random_forest(x_train,x_cross, y_train,Xtest)
    #     # Y_pred, final2 = model_GBR(x_train,x_cross, y_train,Xtest)
    #     xg_pred,final3 = model_XG(x_train,x_cross, y_train,Xtest)
    #     # svr_rbf = SVR(kernel="rbf",C = 1e3, gamma=0.1)
    #     # final = svr_rbf.fit(np.c_[np.array(y_pred),np.array(Y_pred),np.array(xg_pred)],y_cross)
    #     # score = final.predict(np.c_[np.array(final1),np.array(final2),np.array(final3)])
    #     # print("The MSE of RandomForest:",mean_squared_error(y_pred,y_cross))
    #     # print("The MSE of GradientBoostingRegression:", mean_squared_error(Y_pred, y_cross))
    #     print("The MSE of XGboost", mean_squared_error(xg_pred, y_cross))
    #     # print("The Peasorn of RandomForest:", pearsonr(y_pred, y_cross)[0])
    #     # print("The peasorn of GradientBoostingRegression:", pearsonr(Y_pred, y_cross)[0])
    #     # print("The peasorn of all:", pearsonr((Y_pred+y_pred+xg_pred)/3, y_cross)[0])
    #     print("The Peasorn of Xgboost:", pearsonr(xg_pred, y_cross)[0])
    #     pea.append(pearsonr(xg_pred, y_cross)[0])
    #     mse.append(mean_squared_error(xg_pred, y_cross))
    # print(pea)
    # print(mse)
    # index = [i for i,j in enumerate(pea) if j>=0]
    # # print(X_train[:,0])
    # print(index)
    # Xtrain = X_train[:,index[0]]
    # Xtest = X_test[:,index[0]]
    # for i in index[1:]:
    #     Xtrain = np.c_[Xtrain,X_train[:,i]]
    #     Xtest = np.c_[Xtest, X_test[:,i]]
    #
    x_train, x_cross, y_train, y_cross = train_test_split(X_train, Ytrain, test_size=0.3, random_state=1021)
    y_pred, final1= model_random_forest(x_train,x_cross, y_train,X_test)
    Y_pred, final2 = model_GBR(x_train,x_cross, y_train,X_test)
    xg_pred, final3 = model_XG(x_train, x_cross, y_train, X_test)
    print("The MSE of RandomForest:",mean_squared_error(y_pred,y_cross))
    print("The MSE of GradientBoostingRegression:", mean_squared_error(Y_pred, y_cross))
    print("The MSE of XGboost", mean_squared_error(xg_pred, y_cross))
    print("The Peasorn of RandomForest:", pearsonr(y_pred, y_cross)[0])
    print("The peasorn of GradientBoostingRegression:", pearsonr(Y_pred, y_cross)[0])
    print("The Peasorn of Xgboost:", pearsonr(xg_pred, y_cross)[0])
    print("The mse of all:",mean_squared_error((Y_pred+y_pred+xg_pred)/3, y_cross))
    print("The peasorn of all:", pearsonr((Y_pred+y_pred+xg_pred)/3, y_cross)[0])
    ##尝试直接将unsupervised learning的结果直接加到最后,但是效果不理想
    # with open("Us.txt","r") as f:
    #     contents = f.readlines()
    # us_score = []
    # for line in contents:
    #     us_score.append(float((line.strip('\n').split(','))[1]))
    # us_score = np.array(us_score)
    #
    with open("submission_sample", "w") as f:
        for i in range(len(X_test)):
            f.write(y_ids[i] + "," + str((final1[i]+final2[i]+final3[i])/3) + "\n")
