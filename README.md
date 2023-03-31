# Datavisl
1. 导入数据集
from scipy import io as sio 
test_data = sio.loadmat('/home/oneran/Mycodes/计算机大赛/milkdata/Xtest.mat') 
train_data = sio.loadmat('/home/oneran/Mycodes/计算机大赛/milkdata/Xtrain.mat' 
test_value = sio.loadmat('/home/oneran/Mycodes/计算机大赛/milkdata/Ytest.mat') 
train_value = sio.loadmat('/home/oneran/Mycodes/计算机大赛/milkdata/Ytrain.mat 
X_test = test_data['Xtest'] 
X_train = train_data['Xtrain'] 
Y_test = test_value['Ytest'] 
Y_train = train_value['Ytrain']

print('训练集的形状:', X_train.shape) 
print('验证集的形状:', X_test.shape) 
从Scipy库中导入io模块并将其命名为sio
从指定路径加载.mat格式的测试数据和训练数据，分别存储在test_data和train_data变量中，并将它们的值分别赋给X_test和X_train
打印出训练集和测试集的形状
对数据进行缩放

import numpy as np 
# scale the data 
def scale_data(data): 
temp = data.T.copy() 
for i in range(len(temp)): 
temp[i] = (temp[i] - np.min(temp[i])) / (np.max(temp[i]) - np.min(tem 
return temp.T 
X_test_scaled = scale_data(X_test) 
X_train_scaled = scale_data(X_train)

定义一个名为scale_data的函数，该函数将数据进行缩放，使每列数据的最小值为0，最大值为1
使用定义的scale_data函数对训练数据和测试数据进行缩放，并将它们分别存储在X_train_scaled和X_test_scaled中
 2. 观察样本间数据分布情况

from matplotlib import pyplot as plt 
for i in range(40): 
plt.plot(np.arange(len(X_train[0])), X_train[i]) 

从Matplotlib库中导入pyplot模块并将其命名为plt
使用for循环，逐个绘制训练数据的前40个样本的曲线图，其中np.arange(len(X_train[0]))生成了一个与训练数据列数相同的一维数组，表示数据列的下标。plt.plot函数用于绘制折线图
3. 绘制变量与自变量的散点图

from matplotlib import pyplot as plt 
for i in range(X_train.shape[1]): 
plt.figure(i, figsize=(10, 10)) 
plt.xlim(0, 4) 
plt.ylim(0, 4) 
plt.scatter(X_train[:, i], Y_train.flatten()) 
plt.savefig('/home/oneran/Pictures/feature_'+str(i+1))
这段代码使用多种机器学习技术对给定数据集进行模型训练和评估。具体步骤如下：

导入matplotlib.pyplot库，该库用于绘制数据可视化图形。
使用循环遍历数据集中的每一列，对于每一列，使用plt.figure()函数创建一个名为i的图形，并将其大小设置为(10,10)。
使用plt.xlim()和plt.ylim()函数设置x轴和y轴的范围。
使用plt.scatter()函数绘制散点图，其中x轴为X_train数据集中的当前列，y轴为Y_train数据集，将图形保存到/home/oneran/Pictures/feature_+列数的文件中。
4. 搭建模型 
Lasso Alpha=0.00355 

from sklearn.linear_model import Lasso 
lasso = Lasso(alpha=0.00355, max_iter=1000000, fit_intercept=True, tol=0.0000 
lasso.fit(X_train_scaled[:, np.concatenate([np.arange(400, 600), np.arange(90 
lasso_prediction = lasso.predict(X_test_scaled[:, np.concatenate([np.arange(4 
plt.scatter(np.arange(len(X_test)), Y_test.flatten(), marker='o') 
plt.scatter(np.arange(len(X_test)), lasso_prediction, marker='x') 
#* y_mean 
y_mean = np.mean(Y_test) 
print('y_mean:', y_mean) 
#* y_sst 
y_sst = np.sum(np.square(Y_test - y_mean)) 
print('y_sst:', y_sst) 
#* y_reg 
y_reg = np.sum(np.square(lasso_prediction - y_mean)) 
print('y_reg:', y_reg) 
#* y_res 
y_res = np.sum(np.square(Y_test.flatten() - lasso_prediction.flatten())) 
print('y_res:', y_res) 
#* R^2 
R_square = 1 - y_res / y_sst 
print('The R^2 is on:', R_square)

from sklearn.linear_model import Ridge 
ridge = Ridge(alpha=0.1, max_iter=1000000) 
ridge.fit(X_train_scaled[:, np.concatenate([np.arange(400, 600), np.arange(90 
ridge_prediction = ridge.predict(X_test_scaled[:, np.concatenate([np.arange(4 
plt.scatter(np.arange(len(X_test)), Y_test.flatten(), marker='o') 
plt.scatter(np.arange(len(X_test)), ridge_prediction, marker='x') 
#* y_mean 
y_mean = np.mean(Y_test) 
print('y_mean:', y_mean) 
#* y_sst 
y_sst = np.sum(np.square(Y_test - y_mean)) 
print('y_sst:', y_sst) 
#* y_reg 
y_reg = np.sum(np.square(ridge_prediction - y_mean)) 
print('y_reg:', y_reg) 
#* y_res 
y_res = np.sum(np.square(Y_test - ridge_prediction)) 
print('y_res:', y_res) 
#* R^2 
R_square = 1 - y_res / y_sst 
print('The R^2 is on:', R_square)

导入Lasso回归模型，并使用X_train_scaled数据集训练模型。
使用模型对X_test对训练数据进行拟合，并使用测试数据集来评估模型性能。
使用Ridge回归算法对训练数据进行拟合，并使用测试数据集来评估模型性能。
最后，计算模型的R平方值，以衡量模型的拟合程度。

随机森林 树数量:600, 最大深度:5, 最大特征:400, 单个叶子最小样本数:1, 指 
标:MAE 

from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler 
forest = RandomForestRegressor(n_estimators=600, max_depth=5, max_features=40 0）
forest.fit(X_train, Y_train.flatten()) 
print(forest.score(X_test, Y_test.flatten())) 
pre_test = forest.predict(X_test) 
pre_train = forest.predict(X_train) 
plt.figure(1) 
plt.scatter(np.arange(len(X_train)), Y_train.flatten()) 
plt.scatter(np.arange(len(X_train)), pre_train, marker='o') 
plt.figure(2) 
plt.scatter(np.arange(len(X_test)), Y_test.flatten()) 
plt.scatter(np.arange(len(X_test)), pre_test, marker='o')

首先，导入了导入了RandomForestRegressor和StandardScaler模块。随机森林回归和数据标准化处理的类。接着，使用RandomForestRegressor类来构建随机森林回归模型，该模型采用600个决策树，树的最大深度为5，每个决策树使用40个特征。并使用fit()方法拟合训练数据。使用X_train和Y_train来训练模型，接着，使用score()方法计算了在测试集上的得分，并使用predict()方法预测了测试集和训练集的结果。使用训练好的随机森林回归模型对测试集进行预测并打印出来。
5.特征工程
NMF超参数搜索
from sklearn.decomposition import NMF 
from sklearn.linear_model import Lasso 
n_components = [15, 20, 25, 30, 40] 
alpha = [0.5, 0.25, 0.2, 0.1, 0.07, 0.05] 
i = 0 
plt.figure(figsize=(30, 25)) 
for n in n_components: 
nmf_transformer = NMF(n_components=n, max_iter=1000) 
nmf_transformer.fit(X_train_scaled) 
X_train_nmf = nmf_transformer.transform(X_train_scaled) 
X_test_nmf = nmf_transformer.transform(X_test_scaled) 
for a in alpha: 
model = Lasso(alpha=a) 
model.fit(X_train_nmf, Y_train) 
score = model.score(X_test_nmf, Y_test) 
pre = model.predict(X_test_nmf)
plt.subplot(5, 6, i+1) 
i += 1 
plt.title('Alpha: '+str(a)+', n_components: '+str(n)+',score:{:.2f}'. 
plt.scatter(np.arange(len(X_test_nmf)), Y_test.flatten()) 
plt.scatter(np.arange(len(X_test_nmf)), pre, marker='x') 
plt.savefig('/home/oneran/Pictures/hypertest.png') 
本段代码使用 NMF 对数据进行降维，尝试不同的 n_components 和 alpha 参数组合，使用 Lasso 模型对降维后的数据进行拟合，并计算模型的得分。将不同参数组合下的预测结果和实际结果可视化，用于对比不同参数组合下的模型表现。



绘制新特征原始自变量贡献权重柱形图
from matplotlib import pyplot as plt 
plt.figure(figsize=(20, 40)) 
for i in range(20): 
plt.subplot(20, 1, i+1) 
plt.bar(np.arange(1557), nmf_transformer.components_[i]) 
plt.show()

绘制新特征-因变量散点图
from matplotlib import pyplot as plt 
plt.figure(figsize=(20, 16)) 
for i in range(20): 
plt.subplot(5, 4, i+1) 
plt.scatter(X_train_nmf[:, i], Y_train) 
plt.show() 

NMF分解后的特征配合Lasso模型 Alpha(Lasso)=0.01251
from sklearn.decomposition import NMF 
train_data, test_data = X_train_scaled, X_test_scaled 
nmf_transformer = NMF(n_components=20, max_iter=10000) 
nmf_transformer.fit(train_data) 
X_train_nmf = nmf_transformer.transform(train_data) 
X_test_nmf = nmf_transformer.transform(test_data) 
model = Lasso(alpha=0.01251) 
model.fit(X_train_nmf, Y_train.flatten()) 
score = model.score(X_test_nmf, Y_test.flatten()) 
print('Score:', score) 
pre_1 = model.predict(X_test_nmf) 
plt.scatter(np.arange(len(X_test_nmf)), Y_test.flatten()) 
plt.scatter(np.arange(len(X_test_nmf)), pre_1, marker='x') 
本段代码与上一段类似，但只选择了一个固定的参数组合，使用 Lasso 模型进行拟合，并计算模型的得分。同样，将预测结果和实际结果可视化。



NMF分解后的特征配合Ridge模型 Alpha(Ridge)=1.56
ridge = Ridge(alpha=1.56) 
ridge.fit(X_train_nmf, Y_train.flatten()) 
print(ridge.score(X_test_nmf, Y_test.flatten())) 
pre = ridge.predict(X_test_nmf) 
plt.scatter(np.arange(len(X_test_nmf)), Y_test.flatten()) 
plt.scatter(np.arange(len(X_test_nmf)), pre, marker='x')
本段代码使用 Ridge 模型对降维后的数据进行拟合，并计算模型的得分。将预测结果和实际结果可视化。


集成Lasso和Ridge模型
plt.scatter(np.arange(len(X_test_nmf)), Y_test.flatten()) 
plt.scatter(np.arange(len(X_test_nmf)), 1/2*(pre+pre_1), marker='x') 
#* y_mean 
y_mean = np.mean(Y_test) 
print('y_mean:', y_mean) 
#* y_sst 
y_sst = np.sum(np.square(Y_test - y_mean)) 
print('y_sst:', y_sst) 
#* y_reg 
y_reg = np.sum(np.square(pre.flatten() - y_mean)) 
print('y_reg:', y_reg) 
#* y_res 
pre_combined = 1/2*(pre+pre_1) 
y_res = np.sum(np.square(Y_test.flatten() - pre_combined.flatten())) 
print('y_res:', y_res) 
#* R^2 
R_square = 1 - y_res / y_sst 
print('The R^2 is on:', R_square) 

最后一段代码计算模型的 R2 得分，其中 y_mean 为实际结果的均值，y_sst 为总平方和，y_reg 为回归平方和，y_res 为残差平方和。使用 R2 得分来评估模型的性能，R2 得分越高，模型的性能越好。
这一部分代码涉及到机器学习中的回归问题，使用了 Scikit-learn 中的 Random Forest Regressor、StandardScaler、NMF 和 Lasso 等算法和模型。


多次项特征拓展
from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(degree=2, include_bias=False) 
poly.fit(X_train_nmf) 
X_train_poly = poly.transform(X_train_nmf) 
X_test_poly = poly.transform(X_test_nmf) 
lasso = Lasso(alpha=0.00004) 
lasso.fit(X_train_poly, Y_train.flatten()) 
print(lasso.score(X_test_poly, Y_test.flatten())) 
pre = lasso.predict(X_test_poly) 
plt.scatter(np.arange(len(X_test_poly)), Y_test.flatten()) 
plt.scatter(np.arange(len(X_test_poly)), pre, marker='x')


poly = PolynomialFeatures(degree=2, include_bias=False) 
poly.fit(X_train_nmf) 
X_train_poly = poly.transform(X_train_nmf) 
X_test_poly = poly.transform(X_test_nmf) 
ridge = Ridge(alpha=10) 
ridge.fit(X_train_poly, Y_train.flatten()) 
print(ridge.score(X_test_poly, Y_test.flatten())) 
pre = ridge.predict(X_test_poly) 
plt.scatter(np.arange(len(X_test_poly)), Y_test.flatten()) 
plt.scatter(np.arange(len(X_test_poly)), pre, marker='x')
这段代码是使用多项式特征扩展数据集，然后在Lasso和Ridge回归模型上进行训练和测试，并绘制预测结果与实际结果的散点图。具体来说，这里使用PolynomialFeatures函数将数据集的每个特征扩展为多项式特征，这里扩展的次数是2，即每个特征最多只扩展为二次项。然后，对于Lasso回归模型，设置alpha值为0.00004进行训练，并计算测试集的R平方值，最后绘制预测结果与实际结果的散点图。同样的，对于Ridge回归模型，设置alpha值为10进行训练，并计算测试集的R平方值，最后绘制预测结果与实际结果的散点图。


7. 集成Lasso-Ridge模型和随机森林模型进行交叉验证

from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split 
import random 
scores = [] 
for i in range(100): 
data = np.concatenate([train_data, test_data]) 
target = np.concatenate([Y_train, Y_test]) 
train_data, test_data, train_target, test_target = train_test_split(data, 
nmf_transformer = NMF(n_components=20, max_iter=10000000) 
nmf_transformer.fit(train_data) 
X_train_nmf = nmf_transformer.transform(train_data) 
X_test_nmf = nmf_transformer.transform(test_data) 
Y_train = train_target 
Y_test = test_target 
model = Lasso(alpha=0.01251) 
model.fit(X_train_nmf, Y_train) 
score = model.score(X_test_nmf, Y_test) 
print('Score:', score) 
pre_1 = model.predict(X_test_nmf) 
ridge = Ridge(alpha=1.56) 
ridge.fit(X_train_nmf, Y_train.flatten()) 
print(ridge.score(X_test_nmf, Y_test.flatten())) 
pre = ridge.predict(X_test_nmf) 
scores.append(R_square) 
forest = RandomForestRegressor(n_estimators=600, max_depth=5, max_feature 
forest.fit(train_data, Y_train.flatten()) 
print(forest.score(test_data, Y_test.flatten())) 
pre_test = forest.predict(test_data) 
pre_train = forest.predict(train_data) 
#* y_mean 
y_mean = np.mean(Y_test) 
print('y_mean:', y_mean) 
#* y_sst 
y_sst = np.sum(np.square(Y_test - y_mean)) 
print('y_sst:', y_sst) 
#* y_reg 
y_reg = np.sum(np.square(pre.flatten() - y_mean)) 
print('y_reg:', y_reg) 
#* y_res 
pre_combined = (1/2*(pre+pre_1) + pre_test) / 2 
y_res = np.sum(np.square(Y_test.flatten() - pre_combined.flatten())) 
print('y_res:', y_res) 
#* R^2 
R_square = 1 - y_res / y_sst 
print('The R^2 is on:', R_square)
这段代码的功能是使用Lasso-Ridge模型和随机森林模型进行交叉验证，对数据进行拟合和预测，并计算R^2来评估模型的性能。具体操作包括：

将训练数据和测试数据合并为一个数组，并将目标值合并为一个数组。

使用train_test_split函数将数据集划分为训练集和测试集。

使用NMF进行数据转换。

训练Lasso模型，并使用测试数据对其进行评分和预测。

训练Ridge模型，并使用测试数据对其进行评分和预测。

计算随机森林模型在测试集上的R^2，并对测试集和训练集进行预测。

计算平均目标值、SST、SSE和R^2。其中，平均目标值是目标值的平均值，SST是总平方和，SSE是残差平方和，R^2是模型的拟合优度指标，用于评估模型的性能。



8. 集成模型最终成绩
np.mean(scores)
这行代码的作用是计算集成模型的平均得分。其中，scores是一个列表，包含了多次运行集成模型后得到的R^2得分。np.mean()是numpy库中的函数，用于计算列表中所有元素的平均值








