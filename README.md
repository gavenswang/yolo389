a# 模型架构

## 模型trick

1. 用convolution表达局部的信息，用self-attention表达全局信息.
2. 用翻译的技术来做数据增强。

## 模型的优势

1. 训练的速度提升3˜13倍
2.预测的速度提升4到9倍

## 一般的模型架构

 1. An Embedding Layer
 2. Embedding Encode Layer
 3. context-query attention Layer
 4. Model Encode Layer
 5. Output Layer

## 模型设计图

该模型主要用了self-attention, convolution-net的技术。丢弃了所有的RNN技术。

![QANet](./images/QANet.png)

### 数据预处理输入

模型包括问句的词、字向量输入，以及问句的词、字向量输入。维度的大小参考如下：

1. 上下文的词向量(context_idxs) : 

    ```python
    [batch_size, paragraph_length_limit, word_embedding_size]
    ```

2. 上下文的字向量(context_char_idxs) :

    ```python
    [batch_size, paragraph_length_limit, words_char_limit, char_embedding_size]
    ```

3. 问句的词向量(question_idxs) :

    ```python
        [batch_size, question_length_limit, word_embedding_size]
    ```

4. 问句的字向量(question_char_idxs):

    ```python
        [batch_size, question_length_limit, words_char_limit, char_embedding_size]
    ```

5. 答案y1

    ```python
        [batch_size, pargraph_length_limit]
    ```

6. 答案y2

    ```python
        [batch_size, pargraph_length_limit]
    ```

### Embedding 输入层

输入层的数学公式如下:
    $$input = [x_w, x_c] \in R^{p1 + p2}$$
其中p1是词向量的维度，p2是字向量的维度。
这里的主要的操作是通过卷积将字向量转变为词向量，以question为例，具体的做法如下：

1. reshape

    ```python
        x = [batch_size, question_length_limit, words_char_limit, char_embedding_size]
        x = x.reshape( [batch_size * question_length_limit,
            words_char_limt, char_embedding_size])
    ```
2. 一维卷积
    和二维卷积的区别是filter的维度如下
    $$ filter = [width, in\_channel, out\_channel] $$

    这样最终的结果的维度变为
    $$ new\_x\_shape = [batch\_size * question\_length\_limit, new\_size, hidden\_size] $$

3. 做一个max-pooling获得每一个词字向量

    ```python
        new_x = tf.reduce_max(new_x, axis=1)
        new_x = tf.reshpae( [batch_size, question_length_limit, hidden_size])
    ```

在论文里，对于\<UNK\>的字向量可以训练，而非\<UNK\>不需要训练。具体的方法可以参考
```python

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(cost)
# grad  = grad * self.W_mask 可以完成定制化的梯度改进，将某些梯度设置为0即可。
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)
```

### Embedding Encode 层
<strong>前提的知识1</strong>
Deepwise separable convolution 可以用来压缩卷积的参数的个数。具体来说，一般的卷积的参数个数是 
$$ kernel\_height * kernel\_width * input\_channel * output\_channel $$
Deepwise separable convolution可以将参数减小到:
$$ kernel\_height * kernel\_width * input\_channel + input\_channel * ouput\_channel
$$
举例来说，假设

 ```python
        height = 3
        width = 3
        input_channel = 16
        output_channel = 32
        orginal_convolution_para_size = 3 * 3 * 16 * 32 = 4608
        deepwise_para_size = 3 * 3 * 16 + 16 * 32 = 656
 ```

 <strong>前提知识2</strong>

 self attention, attention的步骤解释如下：

   1. <strong>Q</strong>: 原本的数据 [query_length, query_embedding_size]
   2. <strong>M</strong>: Memory [memeory_length, query_embedding_size]
   3. tf.matmul(Q,M, transpose_b=True) 得到 query_length, memory_length
   4. <strong>Attention</strong>: 通过softmax 得到 [query_length, memory_length], memory_length里的值是一个权重值
   5. <strong>C</strong>: 将memory的信息加总到query中，即 tf.matmul(Attention, M)

 在这里attention可以理解为如何将之前的信息(Memmory)根据当前的情况(Query)组织成一个当前的环境。

 在以上的两个前提知识下，模型基本的步骤如下：

 1. 多次卷积
 2. 做一次self-attention
 3. 再次卷积输出
 
 此时的输出的维度
 1. query [batch_size, question_length_limit, hidden_size]
 2. context [batch_size, paragraph_length_limit, hidden_size]
 在论文中hidden_size = 128

### Contex-Query Attention Layer

<strong>前提知识1</strong>
trilinear function, 简单的讲，一个三元函数$y = f(x,y,z)$, 给定8个点的坐标，如何估算这8个点的坐标包含的点的坐标。具体的做法是多次叠加双线性插值。参考下图
![估算C点的值](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Enclosing_points2.svg/440px-Enclosing_points2.svg.png)
![估算C点的值](https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/3D_interpolation2.svg/440px-3D_interpolation2.svg.png)

另外可以参考[线性插值](https://en.wikipedia.org/wiki/Linear_interpolation) 和[双线性插值](https://en.wikipedia.org/wiki/Bilinear_interpolation)


<strong>前提知识点2</strong>

attention中首先是要计算similarity，如下理解这个similarity：

1. attention 解决的是如何将之前的向量加总成为当前预测的一个环境。如果之前的向量的个数是query_len, 当前预测的长度为context_len, 那么权重的向量一定是(context_len,query_len) 我们计算这个矩阵的过程称做similarity computation. 为此attention的做法一定有多种，只要你能得到维度是(context_len, query_len)这样的矩阵。在论文[Bi-Diretional Attention Flow For Machine Comprehension](https://arxiv.org/pdf/1611.01603.pdf)中有比较详细的关于similarity计算的讨论。

2. 论文中similarity的计算采用如下的公式
   $$ f(q,c) = W_0[q,c,q \cdot c] $$

在这一步中，模型需要输出

1. 原本的contex信息， 记为c
2. query给予context的环境信息，也就是atteintion， 这里我们有两个attention，一个是context2query， 记为a, 另外一个是query2context，记为b。

3. 总的输出整理为

$$ [ c,a, c \cdot a, c \cdot b] $$

## Model Encoder

和Embedding Encode 类似， 以context-query的输出作为输入，采用卷积，self-attention的方式输出。不同点是参数不一样。

注意这里完成三次enode。分别记录为$M_0, M_1, M_2$。

## Output Layer

这里值需要理解好下面两个公式即可

预测概率公式：
$$
start\_position\_prob = softmax(W_1 [ M_0,M_1])
$$
$$
start\_position\_prob = softmax(W_1 [ M_0,M_2])
$$

损失函数公式：

$$
L(\theta) = - \frac{1}{N} \sum_{i}^{N} [log(p_{y_i^1}^{1}) + log(p_{y_i^2}^{2})]
$$


# 算法分析

## 实验参数

Name | value | explaination
-|:-:|:-:
max context length | 400 | 段落最大长度是400个字（单词）； 超过的丢弃
max answer length | 30 |
word_vector_size|300|Glove pretrain.没有在Glove的词用unknown表示，并且可以被训练
char_vector_size|200|随机生成，在模型训练的过程中得到重新的训练
语料1 | 107.7k-train, 87.5k-validation|
语料2|140k-train, 87.5k-validation|
语料3|240-train，87.5k-validation|
$\lambda$ of L2|$3 * 10^{-7}$ |每一个参数都参与到L2正则化中
dropout of word_embedding| 0.1|
dropout of char_embedding| 0.05|
dropout between layers | 0.1|
layer dropout | $1-\frac{l}{L}(1-P_L)$ $P_L=0.9$| $l$表示层级，L表示最后一层， 
convolution filter size | 128
batch size | 32
\# convolution layer for embedding | 4
\# convolution layer for model encoder | 2
kernel size for convolution layer of embedding | 7
kernel size for model encoder | 5
block number for convolution layer of embedding | 1
block number for convolution layer of model encoder | 7
其他trainning的参数，参考[论文](https://arxiv.org/pdf/1804.09541.pdf)。

## 模型数据分析

在验证集合上达到当前最好的水平。EM=76.2%, F1=84.6%

* 替换自身的网络层比较性能

```
* 换一层rnn, 速度提升2.9倍
* 两层，6倍
* 3层, 8倍
````

* 和BiDAF 比较

```
5倍
```

* 模型部件的分析

```
convolution in encoders  提升 2个百分点
self-attention 提升1.3个百分点
sep-convolution 提升0.7个百分点
语料增强3:1:1达到最佳，3代表原始的语料，第一个1表示通过法语得到的，第二个1表示通过德语得到的
```

* 模型的鲁棒性分析

```
测试集合的准备
* 加一些和句子相关的问句到context中，和原问句不矛盾
* 加一些被人工确认的和context不相关的句子
实验证明，模型和[MneMonic]模型相当，比其他的好很多
```


# github 代码

## 参考代码
[QANet implementation](https://github.com/NLPLearn/QANet)

