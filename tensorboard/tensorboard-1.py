import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#准备训练数据，假设其分布大致符合y=1.2x+0.0
n_train_examples=200
x_train=np.linspace(-5,5,n_train_examples)
#增加随机扰动
y_train=1.2*x_train+np.random.uniform(-1.0,1.0,n_train_examples)

#准备验证数据，用于验证模型好坏
n_test_examples=50
x_test=np.linspace(-5,5,n_test_examples)
y_test=1.2*x_test

#参数学习算法相关变量配置
learning_rate=0.01
batch_size=20
summary_dir='logs'

#使用placeholder将训练/验证数据送入网络进行训练/验证
#Shap=None 表示形状由送入的张量的形状来确定
with tf.name_scope('Input'):
    X=tf.placeholder(dtype=tf.float32,shape=None,name='X')
    Y=tf.placeholder(dtype=tf.float32,shape=None,name='Y')

#决策函数（参数初始化）
with tf.name_scope('Inference'):
    W=tf.Variable(initial_value=tf.truncated_normal(shape=[1]),name='weight')
    b=tf.Variable(initial_value=tf.truncated_normal(shape=[1]),name='bias')
    Y_pred=tf.multiply(W,X)+b

#损失函数
with tf.name_scope('Loss'):
    loss=tf.reduce_mean(tf.square(Y_pred-Y),name='loss')
    tf.summary.scalar('loss',loss)

#参数学习算法
with tf.name_scope('Optimization'):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#初始化所有变量
init=tf.global_variables_initializer()

#汇总记录节点
merge=tf.summary.merge_all()

#开启会话，进行训练
with tf.Session() as sess:
    sess.run(init)
    summary_writer=tf.summary.FileWriter(logdir=summary_dir,graph=sess.graph)

    for i in range(201):
        #200份训练数据分10份
        j=np.random.randint(0,10)
        x_batch=x_train[batch_size*j:batch_size*(j+1)]
        y_batch = y_train[batch_size * j:batch_size * (j + 1)]

        train_optimizer,summary,train_loss,w_pred,b_pred=sess.run([optimizer,merge,loss,W,b],feed_dict={X:x_batch,Y:y_batch})
        test_loss=sess.run(loss,feed_dict={X:x_test,Y:y_test})
        summary_writer.add_summary(summary,global_step=i)
        print('step:{}.losses:{},test_loss:{}.w_pred:{},b_pred:{}'.format(i,train_loss,test_loss,w_pred[0],b_pred[0]))

        if i==200:
            plt.plot(x_train,y_train,'bo',label='Train data')
            plt.plot(x_test,y_test,'gx',label='Test data')
            plt.plot(x_train,x_train*w_pred+b_pred,'r',label='Predicted data')
            plt.legend()
            plt.show()
    summary_writer.close()




