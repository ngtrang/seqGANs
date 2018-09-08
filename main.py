import data
from gen_model import Generator
from dis_model import Discriminator
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch import tensor
import numpy as np
import torch.nn.functional as F
import os

rnn_type='LSTM'

#字典个数待定
ntoken=0

#词嵌入维度
ninp=64

#隐藏层维度
nhid=128

#循环神经忘了层数
nlayers=2

batch_size=8

pre_gen_epochs=3

#是否双向
bidirect=False

#判别器滤波器的形状
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19]

#判别器滤波器个数
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

adversarial_epochs=5


#把batch data 分为inp 和　target
def split_batch(batch_data,eval=False):
    inp=batch_data[:,:-1].type(torch.LongTensor).cuda()
    target=batch_data[:,1:].type(torch.LongTensor).cuda()
    return inp,target


#从语料库中采样一组batch数据
def sample_one_batch(data,batch_size):
    shuffle_idx=np.random.permutation(data.size(0)-batch_size)
    start_idx=shuffle_idx[0]
    end_idx=start_idx+batch_size
    inp=data[start_idx:end_idx,:-1].type(torch.LongTensor).cuda()
    target=data[start_idx:end_idx,1:].type(torch.LongTensor).cuda()
    return inp,target


#预训练生成器
def pre_train_gen(gen,corpus,batch_size,epochs,ntoken,optimizer):
    train_data=corpus.train
    test_data=corpus.test
    criterion = nn.CrossEntropyLoss()
    train_batch_num=len(train_data)//batch_size
    test_batch_num=len(test_data)//batch_size
    print("start pretrain gen ")
    for epoch in range(epochs):
        train_total_loss=0
        for idx in tqdm(range(0,len(train_data),batch_size)):
            batch_data=train_data[idx:idx+batch_size]
            if batch_data.size(0)!=batch_size:
                continue
            inp,target=split_batch(batch_data)
            inp=Variable(inp)
            target=Variable(target)
            h=gen.init_hidden(batch_size)
            output,_=gen(inp,h)
            loss = criterion(output.view(-1, ntoken), target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss+=loss.item()

        test_total_loss=0
        for idx in tqdm(range(0,len(test_data),batch_size)):
            batch_data=test_data[idx:idx+batch_size]
            if batch_data.size(0)!=batch_size:
                continue
            inp,target=split_batch(batch_data,eval=True)
            inp=Variable(inp,volatile=True)
            target=Variable(target,volatile=True)
            h=gen.init_hidden(batch_size)
            output,_=gen(inp,h)
            loss = criterion(output.view(-1, ntoken), target.view(-1))
            test_total_loss+=loss.data[0]

        print("epoch-{},Train mean_loss-{}".format(epoch,train_total_loss/train_batch_num))
        print("epoch-{},Test mean_loss-{}".format(epoch,test_total_loss/test_batch_num))


#模型梯度开关
def model_grad(model,requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad

#打乱判别器输入数据
def shuffle(inp,target):
    #打乱数据
    shuffle_tensor=torch.zeros(inp.size()).type(torch.LongTensor).cuda()
    shuffle_target=torch.zeros(target.size()).type(torch.LongTensor).cuda()
    assert shuffle_tensor.size(0)==shuffle_target.size(0)

    shuffle_idx=np.random.permutation(inp.size(0))
    for i,idx in enumerate(shuffle_idx):
        shuffle_tensor[i]=inp[idx]
        shuffle_target[i]=target[idx]
    return shuffle_tensor,shuffle_target


def train_dis(dis,gen,corpus,batch_size,epochs,optimizer):
    model_grad(dis,True)#avoid compute grad
    model_grad(gen,False)#avoid compute grad
    train_data=corpus.train
    #采样出的一个batch data

    criterion = nn.CrossEntropyLoss()
    batch_num=len(train_data)//batch_size
    for epoch in range(epochs):
        total_loss=0.
        total_acc=0.
        for idx in tqdm(range(0,len(train_data),batch_size)):

            batch_data=train_data[idx:idx+batch_size]
            sample_data=gen.sample(batch_size,19)
            if batch_data.size(0)!=batch_size:
                continue
            inp,target=split_batch(batch_data)
            dis_inp=torch.cat((sample_data,target))
            dis_target=torch.cat((torch.zeros(batch_size),torch.ones(batch_size)),0).type(torch.LongTensor).cuda()

            dis_inp,dis_target=shuffle(dis_inp,dis_target)

            dis_inp=Variable(dis_inp)
            dis_target=Variable(dis_target)

            #判别器预测结果
            pred_logist=dis(dis_inp)
            loss=criterion(pred_logist,dis_target)
            total_loss+=loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,pred_idx=pred_logist.max(1)
            acc=int(torch.sum(pred_idx==dis_target))/(batch_size*2)
            total_acc+=acc

        print("epoch-{},dis_mean_loss is : {}".format(epoch,total_loss/batch_num))
        print("epoch-{},dis_mean_acc is : {}".format(epoch,total_acc/(batch_num)))


#对抗训练函数
def train_gen_advs(gen,dis,corpus,batch_size,optimizer):
    #设定是否训练
    model_grad(gen,True)
    model_grad(dis,False)

    #采集的生成样本
    samples_from_gen= gen.sample(batch_size, 19)
    inp=torch.cat((torch.zeros(batch_size,1).type(torch.LongTensor).cuda(),samples_from_gen[:,:-1]),1)
    target=samples_from_gen

    rewards = dis(target)[:,1]
    optimizer.zero_grad()
    pg_loss = gen.batchPGLoss(inp, target, rewards)
    pg_loss.backward()
    optimizer.step()

    #采样从语料库中生成样本，加快算法收敛速度
    inp,target=sample_one_batch(corpus.train,batch_size)
    rewards = dis(target)[:,1]
    optimizer.zero_grad()
    pg_loss = gen.batchPGLoss(inp, target, rewards)
    pg_loss.backward()
    optimizer.step()

#对抗损失计算函数
class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        prob=prob.view(-1,prob.size(2)).contiguous()
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()

        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


#主函数入口
def main():

    path='my_data4'

    #加载数据，默认是data1
    corpus = data.Corpus(path=os.path.join("data", path))
    #更新词库个数
    ntoken=len(corpus.dictionary)

    print("Building corpus done---")
    print("Ntoken:{}".format(ntoken))
    print("Train dataset batch nums is :{}".format(len(corpus.train)//batch_size))
    print("Test dataset batch nums is :{}".format(len(corpus.test)//batch_size))


    #定义生成器训练生成器
    gen=Generator(rnn_type, ntoken, ninp, nhid, nlayers).cuda()

    gen_optimizer = optim.Adam(gen.parameters(),lr=0.0005)
    pre_train_gen(gen,corpus,batch_size,pre_gen_epochs,ntoken,gen_optimizer)

    #定义训练预训练判别器
    dis=Discriminator(2, ntoken, ninp, d_filter_sizes,d_num_filters).cuda()
    dis_optimizer = optim.Adam(dis.parameters(),lr=0.0005)
    train_dis(dis,gen,corpus,batch_size,epochs=4,optimizer=dis_optimizer)

    #开始对抗训练
    for total_batch in range(adversarial_epochs):

        print("adversarial-training step----{}".format(total_batch))

        for it in range(200):
            train_gen_advs(gen,dis,corpus,batch_size,gen_optimizer)

        for _ in range(2):
             train_dis(dis,gen,corpus,batch_size,epochs=2,optimizer=dis_optimizer)

        torch.save(gen,"models/gen"+path+".pt")
        torch.save(dis,"models/dis"+path+".pt")


if __name__ == '__main__':
    main()
