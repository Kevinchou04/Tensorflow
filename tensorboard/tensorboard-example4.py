#构建神经网络
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#构建训练数据
#x_data是一个范围(-1,1)，以300分之二等分的列向量,noise值属于正态分布，y是x平方
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data) - 0.5 + noise
#神经层函数，输入值，输入形状，输出形状，激励函数
def add_nn_layer(inputs,in_size,out_size,activation_function=None):
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    wx_plus_b=tf.matmul(inputs,weights)+biases
    if activation_function is None:
        outputs=wx_plus_b
    else:
        outputs=activation_function(wx_plus_b)
    return outputs
#占位符，列向量，列数1
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
#输入层1个神经元，隐藏层10个，输出层1个
#调用函数定义隐藏层和输出层，输入size是上一层的神经元个数，输出是size的本层个数
ll=add_nn_layer(xs,1,10,activation_function=tf.nn.relu)
#输出层
prediction=add_nn_layer(ll,10,1,activation_function=None)
#计算误差，对二者的平方求和再取平均作为损失函数
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    prediction_value=sess.run(prediction,feed_dict={xs:x_data})
    if i % 50==0:
        print('loss:',sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
ax.plot(x_data,prediction_value,'g-',lw=6)
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.show()
