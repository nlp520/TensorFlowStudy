#coding:utf-8
#Team:BuaaNlp
#Author: Sui Guobin
#Date: 2019/12/5
#Tool: PyCharm
import tensorflow as tf
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
message = tf.constant("hello world!!!")

with tf.compat.v1.Session() as sess:
    print(sess.run(message).decode())


