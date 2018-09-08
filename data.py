import os
import torch
import codecs
import sys

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = {}

    def add_word(self, word):
        if word not in self.word_count:
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1

    def cut_min_word(self,min_cut=5):
        #搜集要删除的词
        del_words=[w for w,c in self.word_count.items() if c<=min_cut and w not in ['<start>','<pad>','<eos>']]
        #删除查找到的词
        for w in del_words:
            del self.word_count[w]
        #更新wor和vocab
        words=self.word_count.keys()
        self.word2idx=dict(zip(words,range(len(words))))
        self.idx2word=dict(zip(range(len(words)),words))

    def __len__(self): # function
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path='data/my_data1',sentence_num=1500000,max_seq_len=20):
        #初始化语料库累的饿一些变量
        self.dictionary = Dictionary()
        #先加入词典的特殊字符
        self.dictionary.add_word('<start>')
        self.dictionary.add_word('<pad>')
        self.dictionary.add_word('<eos>')

        self.train_sents = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid_sents = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test_sents = self.tokenize(os.path.join(path, 'test.txt'))
        self.dictionary.cut_min_word()

        self.train=self.sents_to_tensor(self.train_sents,max_seq_len)
        self.valid=self.sents_to_tensor(self.valid_sents,max_seq_len)
        self.test =self.sents_to_tensor(self.test_sents,max_seq_len)

    def sents_to_tensor(self,sents,max_seq_len):
        ids=torch.zeros(len(sents),max_seq_len)
        #做一下填充和拉长
        for idx,line in enumerate(sents):
            line_paded=['<start>']+line+['<pad>']*(20-len(line)-2)+['<eos>']
            for i,word in enumerate(line_paded):
                ids[idx,i] = self.dictionary.word2idx.get(word,self.dictionary.word2idx['<pad>'])
        return ids

    def tokenize(self, path):
        #将句子令牌化
        """Tokenizes a text file."""
        print(path)
        assert os.path.exists(path)
        #生产字典,先加入特殊字符
        sentences=[]
        with codecs.open(path,'r',encoding='utf8',errors='ignore') as f:
            for line in f:
                words = line.split()[:18]
                #太长太短都不要,太长了切断，太短了补全
                if len(words)>15 and len(words)<=18:
                    sentences.append(words)
                    for word in words:
                        self.dictionary.add_word(word)
        return sentences
