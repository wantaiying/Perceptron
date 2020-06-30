
'''利用高斯白噪声生成基于某个直线附近的若干个点
y=wx+b
w:weight 权值
b：bias 直线偏执
size： 点的个数
'''
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt#对应数据可视化
 #训练点生成函数
def random_point_nearby_creat(weight,bias,size=10):
   
    x_point=np.linspace(-1,1,size)[:,np.newaxis]

    noise=np.random.normal(0,0.5,x_point.shape)#加入噪声
    y_point=weight*x_point+bias+noise
    input_arr=np.hstack((x_point,y_point))
    return input_arr


#直线真正的参数
real_weight=1
real_bias=3
size=100
#使用函数，生成训练数据
input_point=random_point_nearby_creat(real_weight,real_bias,size)
#给数据打上标签
label=np.sign(input_point[:,1]-(input_point[:,0]*real_weight+real_bias)).reshape((size,1))



#将数据拆分为测试集与训练集
testSize=15
x_train,x_test,y_train,y_test=train_test_split(input_point,label,test_size=testSize)
trainSize=size-testSize


#绘制初始点与直线
fig=plt.figure()#生成一个图片框
ax=fig.add_subplot(1,1,1)#编号
for i in range(y_train.size):
    if y_train[i]==1:
        ax.scatter(x_train[i,0],x_train[i,1],color="r")#输入真实值（点的形式）红色在线上
    else:
        ax.scatter(x_train[i,0],x_train[i,1],color="b")#输入真实值（点的形式）蓝色在线下
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.show()



#模型训练 ，采用随机梯度下降法
#初始化w,b  w为向量
Weight=np.random.rand(2,1)#起始权值随机即可
Bias=0#这个其实也可以随机
def trainByStochasticGradientDescent(input,output,x_test,y_test,test_size,input_num,train_num=10000,learning_rate=1.0):
   
    global Weight,Bias#声明用到的是全局变量
    x=input
    y=output
    for rounds in range(train_num):#设定训练次数
        
        for i in range(input_num):#对训练集中的每个点进行训练
            x1,x2=x[i]
            prediction=np.sign(Weight[0]*x1+Weight[1]*x2+Bias)#生成判断值
            if y[i]*prediction<=0:#判断与真实不符合，需要改变w和b
                Weight[0]=Weight[0]+learning_rate*y[i]*x1
                Weight[1]=Weight[1]+learning_rate*y[i]*x2
                Bias=Bias+learning_rate*y[i]
        if rounds %10 ==0:#每十次训练，用测试数据展示下当前准确率
            learning_rate*=0.9  #不断缩短步长
            accuracy=compute_accuracy(x_test,y_test,test_size,Weight,Bias)
            print("rounds{},accuracy{}".format(rounds,accuracy))

#真正测试
def compute_accuracy(x_test,y_test,test_size,weight,bias):
    x1,x2=np.reshape(x_test[:,0],(test_size,1)),np.reshape(x_test[:,1],(test_size,1))
    prediction=np.sign(y_test*(x1*weight[0]+x2*weight[1]+bias))#真正测试(看训练的效果)
    count=0
    for i in range(prediction.size):
        if prediction[i]>0:
            count=count+1
    return (count+0.0)/test_size



trainByStochasticGradientDescent(x_train,y_train,x_test,y_test,testSize,85,train_num=100,learning_rate=1)



