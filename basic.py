#!usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on 2018年1月24日
@author: sui
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告

'''
Tensorflow变量
'''
def test0():
    #创建一个变量
    state = tf.Variable(0,name='counter')
    #创建一个op，使得state逐渐加1
    one = tf.constant(1)
    new_value = tf.add(state,one)
    update = tf.assign(state,new_value)#通过给它分配'值'来更新'ref'
    #启动图进行运算
    initop = tf.global_variables_initializer()
    with tf.Session() as sess:
        #运行initop
        sess.run(initop)
        #打印state的值
        print(sess.run(state))
        # 运行 op, 更新 'state', 并打印 'state'
        for _ in range(3):
            sess.run(update)
            print (sess.run(state))

'''
取回操作
'''
def testFetch():
    input1 = tf.constant(3.0)
    input2 = tf.constant(4.0)
    input3 = tf.constant(5.0)
    a = tf.zeros((2,3), tf.float32)
    b = tf.zeros((2,3), tf.float32)
    c = tf.ones((1,3))
    d = tf.zeros((4,1))
    intermed = tf.add(input2,input3)
    concat = tf.concat([a,b],axis = 0)
    diancheng = c * c
    mul = tf.multiply(input1,intermed)#乘法
    with tf.Session() as sess:
        result = sess.run([diancheng,concat,mul,intermed])
        print (result)
    #结果[27.0,9.0]
    #需要获取的多个 tensor 值，在 op 的一次运行中一起获得（而不是逐个去获取 tensor）。
    
'''
Feed操作
上述示例在计算图中引入了 tensor, 以常量或变量的形式存储. 
TensorFlow 还提供了 feed 机制, 该机制 可以临时替代图中的任意操作中的 
tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.
'''
def testFeed():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)    
    with tf.Session() as sess:
        result = sess.run(output,feed_dict={
                input1:[2.0],input2:[3.0]})
        print(result)
    
'''
tensorflow基础教程
'''
def test1():
    import tensorflow as tf
    import numpy as np
    #使用np生成假的数据
    x_data = np.float32(np.random.rand(2,100))
    y_data = np.dot([0.1,0.2],x_data)+0.3
    
    #构造线性模型
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
    y = tf.matmul(W,x_data)+ b
    
    #最小化方差
    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    
    #初始化变量
    init = tf.initialize_all_variables()
    
    #启动图(graph)
    sess = tf.Session()
    sess.run(init)
    
    #拟合平面
    for step in range(0,201):
        sess.run(train)
        if step % 20 == 0:
            print (step,sess.run(W),sess.run(b))

'''
简单的全连接的mnist实例
'''
def test2():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    '''
    None表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小
    '''
    x = tf.placeholder(tf.float32, [None,784])#占位符
    y_ = tf.placeholder(tf.float32,  [None,10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    #tf.reduce_sum将整个minibatch中的交叉熵相加
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True
    with tf.Session(config=gpuConfig) as sess:
        sess.run(init)
        for step in range(10000):
            batch_xs,batch_ys =mnist.train.next_batch(100)
            '''
            每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，
            并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
            '''

            print(sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys}))
        #返回的是一个bool布尔类型的数组，[True,False,True....]
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #计算准确率  tf.cast类型转换函数
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #将其计算并进行打印
        print( sess.run(accuracy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels}))
        #保存加载模型
        # saver = tf.train.Saver()
        # #保存模型
        # saver.save(sess,"modelsave",global_step=step)
        # #加载模型
        # saver.restore(sess, 'modelsave')
        '''
        数据的测试和评估
        print 'Training Data Eval:'
        do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_sets.train)
        print 'Validation Data Eval:'
        do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_sets.validation)
        print 'Test Data Eval:'
        do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_sets.test)
        '''
        
        
'''
使用卷积的MNIST实例
'''
def test3():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    def weight_variable(shape):
        #从截断的正态分布中输出随机值
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
    '''
    第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape
    第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    
    '''
    def conv2d(x,W):
        #strides步长是1   padding="SAME"表示输入和输出一样的了    valid的话不添加边界
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME",use_cudnn_on_gpu=True)
    '''
    第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
    '''
    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],
                              strides=[1,2,2,1],
                              padding= 'SAME')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    '''
            None表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小
            '''
    x = tf.placeholder(tf.float32, [None, 784])  # 占位符
    y_ = tf.placeholder(tf.float32, [None, 10])
    '''
    卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，
    接着是输入的通道数目，最后是输出的通道数目(灰度图为1,RGB图为3)
    '''
    W_con1 = weight_variable([5, 5, 1, 32])
    b_con1 = bias_variable([32])
    # 将x变成一个4d的向量
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 第一层卷积运算
    h_con1 = tf.nn.relu(conv2d(x_image, W_con1) + b_con1)
    h_pool1 = max_pool_2x2(h_con1)

    # 第二层卷积运算
    W_con2 = weight_variable([5, 5, 32, 64])
    b_con2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_con2)) + b_con2
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(h_fc1)

    # 加入dropout防止过拟合
    '''
    我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
    TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，
    还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。
    '''
    keep_prob = tf.placeholder("float")  # 定义了一个单个的占位符
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 模型的训练与评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # tf.equal得到的是一个布尔类型的数组
    # tf.cast将张量投射到新的类型
    # tf.reduce_mean计算张量维度上元素的平均值。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        # 每运行100词计算一下训练集的准确率
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d,train accuracy %g" % (i, train_accuracy))
        # 训练数据
        sess.run(train_step, feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.5})
    # 计算测试数据集的准确率
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: mnist.test.image, y_: mnist.test.labels, keep_prob: 1.0}))

    sess.close()

    '''
    可以使用
    with tf.Session() as sess:
        ....
    这样就可以不用调用sess.close()来释放资源
    '''

'''
文档中的 Python 示例使用一个会话 Session 来 启动图, 并调用 Session.run() 方法执行操作.
为了便于使用诸如 IPython 之类的 Python 交互环境, 可以使用 InteractiveSession 代替 Session 类, 使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run(). 这样可以避免使用一个变量来持有会话.
# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.sub(x, a)
print sub.eval()
# ==> [-2. -1.]

'''
'''
从头开始构建一个简单的神经网络
'''
def test4():
    '''
    由以下三个步骤构成：
    1.inference():尽可能的构建好图标，并且满足前馈神经网络向前反馈并作出预测的要求
    2.loss():往inference表中添加生成损失（loss）所需要的操作
    3.training():往损失表中添加计算并应用梯度所需的操作
    '''
    '''
    数据连接操作
    '''
    x = tf.Variable([[1,2],[2,3]])
    y = tf.Variable([[2,3],[4,5]])
    z = tf.concat([x,y],1)
    a = tf.Variable([[0,0,0,1,0,0,0],[0,0,0,1,0,0,0]])
    batch = tf.size(a)#包含元素的总数
    '''
    ```python
  # 't' is a tensor of shape [2]
  tf.shape(tf.expand_dims(t, 0))  # [1, 2]
  tf.shape(tf.expand_dims(t, 1))  # [2, 1]
  tf.shape(tf.expand_dims(t, -1))  # [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
  tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
  tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]
  ```
    '''#增加维度，由二维变成三维   设置哪一维变成1
    NUM_CLASSES = 4
    labels = tf.Variable([0,1,0,0])#,[1,0,0,0],[0,0,1,0],[0,0,0,1]
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    sha = tf.shape(indices)
    '''
    concated = tf.concat([indices, labels],1)
    onehot_labels = tf.sparse_to_dense(
            concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    '''
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
#         print sess.run(z)
        print (sess.run(batch_size))
        print (sess.run(indices))
        print (sess.run(sha))
    
'''
tensorflow中常用的函数
'''
def test5():
    '''
    x = tf.random_normal()#正态分布
    x = tf.truncated_normal()#满足正态分布的随机值,若随机值偏离平均值超过2个标准差，则这个数会被重新随机
    x = tf.random_uniform()#平均分布
    x = tf.random_gamma()#Gramma分布
    x = tf.zeros()#产生全0的函数
    x = tf.ones()#产生全1的函数
    x = tf.fill([2,3],9)#给定元素的函数
    x = tf.constant([2,3,4])#产生给定值的常量
    
    y = tf.constant_initializer()#将变量初始化为给定常数
    y = tf.random_normal_initializer()#将变量初始化为满足正态分布的随机值
    
    '''
    '''
    tf.log()#对数操作
    tf.matul()#矩阵相乘
    tf.multiply()#矩阵点乘
    z = tf.transpose()#矩阵转置
    '''
    '''
    常用的矩阵运算函数
    tf.diag(diagonal,name=None)   #生成对角矩阵
    tf.diag_part(input,name=None)  #功能与tf.diag函数相反,返回对角阵的对角元素
    tf.trace(x,name=None)  #求一个2维Tensor足迹，即为对角值diagonal之和
    #适用于很多转置操作  perm是调整后的维度，比如之前，维度是2X3X4的矩阵，按照perm=[0,2,1]操作之后
     变为了2X4X3的矩阵
    tf.transpose(a,perm=None,name='transpose')  #调换tensor的维度顺序，按照列表perm的维度排列调换tensor的顺序
    x.get_shape()#获取大小                                   
    '''
    '''
    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
    tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
    '''
    '''
    x = tf.constant([[[1,2,3,4],[5,6,7,8]],[[1,2,3,4],[5,6,7,8]]])
    y = tf.unstack(x,axis =0)
    with tf.Session() as sess:
        z = sess.run(y)
        print(z[0])
        结果：
        [[1 2 3 4]
         [5 6 7 8]]
        '''
    '''
    tf.slice()切片
    begin 起点的坐标
    size 切片的大小
    begin[i] + size[i] <=input.shape[i]
    
    tf.split()分割
    分割后list中的每个tensor的  维度不降 维度不降
    '''
    input = tf.placeholder('float')
    begin = tf.placeholder('int32')
    size = tf.placeholder('int32')
    out = tf.slice(input, begin, size)
    s = tf.split(input, num_or_size_splits=3, axis=0)
    sess= tf.Session()
    #3*2*3
    input_ = [[[1, 1, 1], [2, 2, 2]],[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]]
    begin_ = [1, 0, 0]
    size_ = [1,1,3]
    o = sess.run(out, feed_dict={input:input_, begin:begin_, size:size_})
    print (o)
    s_ = sess.run(s, feed_dict={input:input_})
    print(s_)
    print(type(s_))
    print(s_[0])
    print(type(s_[0]))
    print(type(s))
    print(type(out))
    sess.close()

def test6():
    '''
    GPU训练测试
    :return:
    '''
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # 新建session with log_device_placement并设置为True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # 运行这个 op.
    sess.run(c)

if __name__ == '__main__':
#     test5()
    test3()
    pass