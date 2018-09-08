import torch
import data
import os
import math
import numpy as np
from torch.autograd import Variable

#将二进制字符串转换为二进制码
def string2bins(sec_text, n_bins):
    bit_string = ''.join(bin(ord(letter))[2:].zfill(8) for letter in sec_text)
    n_bits = int(math.log(n_bins, 2))
    return [bit_string[i:i+n_bits] for i in range(0, len(bit_string), n_bits)]








def main():
    path='my_data1'
    sec_text="I wanna go out tonight"
    n_bins=4

    #加载数据，默认是data1
    corpus = data.Corpus(path=os.path.join("data", path))
    gen=torch.load("models/gen_"+path+".pt").cuda()
    print(gen)

    bin_stream=string2bins(sec_text,n_bins)

    ntokens = len(corpus.dictionary)
    tokens = list(range(ntokens)) # * args.replication_factor
    np.random.shuffle(tokens)
    words_in_bin = int(len(tokens) /n_bins)
    bins = [tokens[i:i + words_in_bin] for i in range(0, len(tokens), words_in_bin)]
    zero = [list(set(tokens) - set(bin_)) for bin_ in bins]



    #循环生成每一个词

    for _ in range(10):

        input = Variable(torch.Tensor([corpus.dictionary.word2idx['<start>']]), volatile=True).view(-1,1).type(torch.LongTensor).cuda()
        h=gen.init_hidden(1)
        gen_words=[]
        for i in range(len(bin_stream[:16])):
            output,h=gen(input,h)

            zero_index = zero[int(bin_stream[i],2)]
            zero_index = torch.LongTensor(zero_index).cuda()

            output = output.squeeze().data.div(0.8).exp()
            output.index_fill_(0, zero_index, 0)

            word_idx = torch.multinomial(output, 1)[0]
            gen_words.append(word_idx)
            input.data.fill_(word_idx)

        print(len(gen_words))
        str_=" ".join([corpus.dictionary.idx2word[x] for x in gen_words])
        print(str_)






if __name__ == '__main__':
    main()
