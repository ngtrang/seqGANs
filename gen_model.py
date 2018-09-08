import torch
import torch
import torch.nn as nn
from torch.autograd import Variable


#生成器网络
class Generator(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        super(Generator, self).__init__()

        #词嵌入层
        self.encoder = nn.Embedding(ntoken, ninp)

        #判断循环神经网络类型
        if rnn_type == 'LSTM':
            self.rnn =nn.LSTM(ninp, nhid, nlayers,batch_first=True)
        else:
            self.rnn =nn.GRU(ninp, nhid, nlayers,batch_first=True)

        self.decoder = nn.Linear(nhid, ntoken)

        self.softmax=nn.Softmax()

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.softmax = nn.LogSoftmax()

    def init_weights(self):
        #初始化范围
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded =self.softmax( self.decoder(output.contiguous().view(-1, self.nhid)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    #初始化参数
    def init_hidden(self, bsz):
        if self.rnn_type == 'LSTM':
            return (Variable(torch.rand(self.nlayers, bsz, self.nhid)).cuda(),
                    Variable(torch.rand(self.nlayers, bsz, self.nhid)).cuda())
        else:
            return Variable(torch.rand(self.nlayers, bsz, self.nhid)).cuda()

    def sample(self,batch_size,seq_len,data=None):
        #用来采样出一个batch的结果，
        """
        data 是已有序列
        """
        #如果没有data就从０开始
        if data is None:
            sample_batch=(torch.zeros(batch_size,seq_len).type(torch.LongTensor)).cuda()
            inp=Variable(torch.zeros(batch_size,1).type(torch.LongTensor)).cuda()
            h=self.init_hidden(batch_size)
            for i in range(seq_len):
                output,h=self.forward(inp,h)
                output=torch.multinomial(output.exp().squeeze(),1)
                sample_batch[:,i]=output.data
                inp=output
            return sample＿batch
        #否则就从部分开始
        else:
            sample_batch=(torch.zeros(batch_size,seq_len).type(torch.LongTensor)).cuda()
            inp=Variable(torch.zeros(batch_size,1).type(torch.LongTensor)).cuda()
            h=self.init_hidden(batch_size)
            for i in range(seq_len):
                if i<data.size(1):
                    inp=data[:,i].unsqueeze(1)
                else:
                    inp=sample_batch[:,i-1].unsqueeze(1)
                output,h=self.forward(inp,h)
                output=torch.multinomial(output.exp().squeeze(),1)
                sample_batch[:,i]=output.data
            return sample＿batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)
            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()

        # inp = inp.permute(1, 0)          # seq_len x batch_size
        # target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)
        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[:,i].unsqueeze(1), h)
            for j in range(batch_size):
                out=out.squeeze()
                loss += -out[j][target[j][i]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size
