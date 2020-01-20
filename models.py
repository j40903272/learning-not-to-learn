import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MalConv(nn.Module):
    def __init__(self,vocab_size, input_length=2000000,window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(vocab_size, 8, padding_idx=0)
        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.pooling = nn.MaxPool1d(int(input_length/window_size))
        self.fc_1 = nn.Linear(128,128)
        self.fc_2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        

    def forward(self,x):
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x,-1,-2)
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)
        x = x.view(-1,128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        #x = self.sigmoid(x)
        return x


class convnet(nn.Module):
    def __init__(self,num_classes=10):
        super(convnet,self).__init__()
        self.bn0     = nn.BatchNorm2d(3)
        self.conv1   = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2   = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1)
        self.conv4   = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x) # 14x14

        x = self.conv2(x)
        x = self.relu(x) #14x14
        feat_out = x  
        x = self.conv3(x)
        x = self.relu(x) # 7x7
        x = self.conv4(x)
        x = self.relu(x) # 7x7

        feat_low = x
        feat_low = self.avgpool(feat_low)
        feat_low = feat_low.view(feat_low.size(0),-1)
        y_low = self.fc(feat_low)

        return feat_out, y_low


class Predictor(nn.Module):
    def __init__(self, input_ch=32, num_classes=8):
        super(Predictor, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.pred_bn1   = nn.BatchNorm2d(input_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, num_classes, kernel_size=3,
                                    stride=1, padding=1)
        self.softmax    = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        px = self.softmax(x)

        return x,px
    
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, \
                char_vocab_size, word_seq_len, char_seq_len, emb_size, \
                l2_reg_lambda=0, kernel_sizes=[3,4,5,6], dropout=0.2):
        super(TextCNN, self).__init__()
        

        # parameters
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.emb_size = emb_size
        self.filter_size = 256
        self.num_filters_total = self.filter_size * len(kernel_sizes) 


        # embedding layers
        self.char_emb = nn.Embedding(char_ngram_vocab_size, emb_size)
        self.word_emb = nn.Embedding(word_ngram_vocab_size, emb_size)
        self.charseq_emb = nn.Embedding(char_vocab_size, emb_size)
        self.drop = nn.Dropout(dropout)

        # conv2d layers
        self.conv_dict = {}
        for size in kernel_sizes:
            self.conv_dict[size] = nn.Conv2d(in_channels=1, 
                                             out_channels = self.filter_size, 
                                             kernel_size = (size, emb_size), 
                                             stride = 1)
        self.convnet = nn.Sequential(*[self.conv_dict[i] for i in self.conv_dict])
        ### char
        self.char_conv_dict = {}
        for size in kernel_sizes:
            self.char_conv_dict[size] = nn.Conv2d(in_channels=1, 
                                             out_channels = self.filter_size, 
                                             kernel_size = (size, emb_size), 
                                             stride = 1)
        self.char_convnet = nn.Sequential(*[self.char_conv_dict[i] for i in self.char_conv_dict])
            
        # maxpool2d layers
        self.maxpool_dict = {}
        for size in kernel_sizes:
            self.maxpool_dict[size] = nn.MaxPool2d(kernel_size=(self.word_seq_len - size + 1, 1),
                                                   stride = 1)
        self.maxpool = nn.Sequential(*[self.maxpool_dict[i] for i in self.maxpool_dict])
        
        # fc layers
        self.fc_char = nn.Linear(self.num_filters_total, 512)
        self.fc_word = nn.Linear(self.num_filters_total, 512)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)


    def forward(self, inp_word, inp_char, inp_pad, inp_charseq):
        # embedding lookup
        char_emb = self.char_emb(inp_char)
        word_emb = self.word_emb(inp_word)
        charseq_emb = self.charseq_emb(inp_charseq)
        # mask padding
        char_emb = torch.mul(char_emb, inp_pad)
        sum_ngram_char = torch.sum(char_emb, dim=2)
        # combine word and char-level-word
        sum_ngram = torch.add(sum_ngram_char, word_emb)
        # expand
        sum_ngram_expand = sum_ngram.unsqueeze(1)
        char_emb_expand = charseq_emb.unsqueeze(1)

        
        ### char ###
        output = []
        for e, filter_size in enumerate(self.char_conv_dict):
            conv = self.char_conv_dict[filter_size]
            maxpool = self.maxpool_dict[filter_size]
            x = F.relu(conv(char_emb_expand))
            x = maxpool(x)
            output.append(x)

        output = torch.cat(output, dim=1)
        output = output.view(-1, self.num_filters_total)
        output = self.drop(output)
        char_output = output


        ### word ###
        output = []
        for e, filter_size in enumerate(self.conv_dict):
            conv = self.conv_dict[filter_size]
            maxpool = self.maxpool_dict[filter_size]
            x = F.relu(conv(sum_ngram_expand))
            x = maxpool(x)
            output.append(x)

        output = torch.cat(output, dim=1)
        output = output.view(-1, self.num_filters_total)
        output = self.drop(output)
        word_output = output
        

        ### concat ###
        char_output = self.fc_char(char_output)
        word_output = self.fc_word(word_output)
        output = self.drop(torch.cat((char_output, word_output), dim=1))
        output = self.drop(F.relu(self.fc1(output)))
        output = self.drop(F.relu(self.fc2(output)))
        output = self.drop(F.relu(self.fc3(output)))
        logits = F.relu(self.fc4(output))

        return F.log_softmax(logits, dim=1)

    
class WordCNN(nn.Module):
    def __init__(self, word_ngram_vocab_size, word_seq_len, 
                 emb_size, l2_reg_lambda=0, dropout=0.2,
                 kernel_sizes=[3,4,5,6], filter_size=256):
        
        """
        input (batch, 1, seq_len, emb_siqe) (64, 1, 200, 32)
        conv 
            kernel (ksize, emb_dim) (3, 32)
            output (batch, filter_size, h_out, w_out) (64, 256, 198, 1)
        maxpool
            kernel (seq_len-ksize, 1) (198, 1)
            output (batch, filter_size, h_out, w_out) (64, 256, 1, 1)
        """
        super(WordCNN, self).__init__()
        
        # parameters
        self.word_seq_len = word_seq_len
        self.emb_size = emb_size
        self.filter_size = 256
        self.num_filters_total = self.filter_size * len(kernel_sizes) 


        # embedding layers
        self.word_emb = nn.Embedding(word_ngram_vocab_size, emb_size)
        self.drop = nn.Dropout(dropout)

        # conv2d layers
        self.conv_dict = {}
        for size in kernel_sizes:
            self.conv_dict[size] = nn.Conv2d(in_channels=1, 
                                             out_channels = self.filter_size, 
                                             kernel_size = (size, emb_size), 
                                             stride = 1)
        self.convnet = nn.Sequential(*[self.conv_dict[i] for i in self.conv_dict])
            
        # maxpool2d layers
        self.maxpool_dict = {}
        for size in kernel_sizes:
            self.maxpool_dict[size] = nn.MaxPool2d(kernel_size=(self.word_seq_len - size + 1, 1),
                                                   stride = 1)
        self.maxpool = nn.Sequential(*[self.maxpool_dict[i] for i in self.maxpool_dict])
        
        # fc layers
        self.fc1 = nn.Linear(self.num_filters_total, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
    
    def forward(self, inp_word):
        # embedding lookup
        word_emb = self.word_emb(inp_word)
        # expand
        word_emb_expand = word_emb.unsqueeze(1)

        ### word ###
        output = []
        for e, filter_size in enumerate(self.conv_dict):
            conv = self.conv_dict[filter_size]
            maxpool = self.maxpool_dict[filter_size]
            #conv = self.convnet[e]
            #maxpool = self.maxpool[e]
            x = F.relu(conv(word_emb_expand))
            x = maxpool(x)
            output.append(x)

        output = torch.cat(output, dim=1)
        output = output.view(-1, self.num_filters_total)
        output = self.drop(output)
        
        output = self.drop(F.relu(self.fc1(output)))
        output = self.drop(F.relu(self.fc2(output)))
        output = self.drop(F.relu(self.fc3(output)))
        logits = self.fc4(output)

        return logits #F.log_softmax(logits, dim=1)
    
    

    
 


    
    
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)
    

class BiasPredictor(nn.Module):
    def __init__(self, emb, emb_size, hidden_size=32, num_classes=2):
        super(BiasPredictor, self).__init__()
        self.emb = emb
        self.fc1 = nn.Linear(emb_size, num_classes)
        #self.fc2 = nn.Linear(hidden_size, )
        self.dropout_p = 0.2

    def forward(self, x):
        x = self.emb(x)
        x = grad_reverse(x)
        x = x.flatten(0, 1)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, self.dropout_p)
        logits = self.fc1(x)
        return logits