import torch
import sys
import data
from torch.autograd import Variable
import nltk
import torch.nn as nn
from tqdm import tqdm
import math
from numpy import random
import numpy as np
import os
from collections import Counter



#把batch data 分为inp 和　target
def split_batch(batch_data,eval=False):
    inp=batch_data[:,:-1].type(torch.LongTensor).cuda()
    target=batch_data[:,1:].type(torch.LongTensor).cuda()
    return inp,target


#计算softmax的向量值
def log_softmax(unnormalized_probs, bin_,bins,bins_num):

    # col softmax
    denom = torch.sum(unnormalized_probs.exp(), 1) # denom is a 200 * 1 tensor
    denom = (denom.expand(unnormalized_probs.size(1),denom.size(0))).permute(1,0).contiguous()
    probs = torch.div(unnormalized_probs.exp(), denom)

    if bins_num >= 2:
        probs=mask_probabilities(probs, bin_,bins,bins_num)
    log_probs = torch.log(probs)
    return log_probs # output is a n * vocab tensor

#用平均的值代替所不在bin的单词的概率
def mask_probabilities(probs, bin_,bins,bins_num):
    mask_words = bins[bin_]
    mask_words = list(set(mask_words))

    divided_probs = torch.div(probs, bins_num)
    numpy_divided_probs = divided_probs.cpu().data.numpy()
    numpy_probs = probs.cpu().data.numpy()
    numpy_probs[:,mask_words] = numpy_divided_probs[:,mask_words]
    probs.data = torch.FloatTensor(numpy_probs).cuda()
    return probs


#计算blue值
def blue_eval(output,corpus):
    #采样以后为２０＊１９
    sent_idx=torch.multinomial(output.exp().cpu(), 1).view(-1,19)
    sent_idx=sent_idx.cpu().data.numpy()
    sent_str=[]

    #对生产的一个batch量数据进行处理
    for i in range(sent_idx.shape[0]):
        str_=[str(int(x)) for x in sent_idx[i,:-1]]
        sent_str.append(str_)

    eval_data=[]
    for sent in corpus.valid.numpy():
        eval_data.append([str(int(x)) for x in sent[1:-1]])

    weight = tuple((1. / 4 for _ in range(4)))
    BLEUscores=[]

    for gen_sent in sent_str:
        ref_sent_info=[]
        for ref_sent in eval_data:
            #找到与这个最相似的句子
            common_tokens = Counter(gen_sent) & Counter(ref_sent)
            correct_preds = sum(common_tokens.values())
            recall_wrt = float(correct_preds) / len(gen_sent)
            ref_sent_info.append((ref_sent,recall_wrt))

        ref_sent_info.sort(key=lambda x: -x[1])
        top_refs=[x[0] for x in ref_sent_info[:50]]

        BLEUscore = nltk.translate.bleu_score.sentence_bleu(top_refs, gen_sent, weight)
        BLEUscores.append(BLEUscore)

    score=(np.mean(BLEUscores))
    return score


#实验评价函数,参数，语料库对象和生成器模型
def evaluate(corpus,gen,batch_size,bins_num):

    data_source=corpus.valid
    ntokens = len(corpus.dictionary)

    if bins_num >= 2:
        tokens = list(range(ntokens)) # * args.replication_factor
        random.shuffle(tokens)

        words_in_bin = int(len(tokens) / bins_num)

        bins = [tokens[i:i + words_in_bin] for i in range(0, len(tokens), words_in_bin)] # words to keep in each bin...
        bins = [list(set(tokens) - set(bin_)) for bin_ in bins] #对bin字典去反的结果

    #固定住生成器，不计算梯度
    for p in gen.parameters():
        p.requires_grad = False

    #data_source 是验证数据
    total_loss = 0
    blue_sorces=[]
    criterion = nn.NLLLoss()
    for i in tqdm(range(0, data_source.size(0), batch_size)):
        inp, targets = split_batch(data_source[i:i+batch_size])
        if inp.size(0)!=batch_size:
            continue
        inp=Variable(inp).cuda()
        targets=Variable(targets).view(-1).cuda()

        h = gen.init_hidden(batch_size)
        output, hidden = gen(inp, h)
        output_flat = output.view(-1, ntokens)

        #随机生产一个块
        if bins_num>=2:
            bin_ = random.choice(range(bins_num))
            output_flat = log_softmax(output_flat, bin_,bins,bins_num)

        blue_sorce=blue_eval(output_flat,corpus)
        blue_sorces.append(blue_sorce)

        total_loss += len(inp) * criterion(output_flat, targets).data
        #再次包装隐藏变量
    return total_loss[0] / len(data_source),np.mean(blue_sorces)


def main():

    ##载数据，默认是data1
    file_path='my_data1'
    corpus = data.Corpus(path=os.path.join("data",file_path))
    #更新词库个数
    ntoken=len(corpus.dictionary)
    gen=torch.load("models/gen_"+file_path+".pt").cuda()
    print(gen)
    batch_size=64

    for num in [1,2,4,8]:
        bins_num=num
        print("bins_num:",num)
        valid_loss,blue=evaluate(corpus,gen,batch_size,bins_num)
        print('| End of training | valid loss {:5.2f} | valid ppl {:8.2f}  blue loss:{:8.5f}'.format(
        valid_loss, math.exp(valid_loss),blue))
        print("---"*85)







if __name__ == '__main__':
    main()
