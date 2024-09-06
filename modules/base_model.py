import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import math
import utils.config as config
from modules.fc import FCNet
from modules.classifier import SimpleClassifier
from modules.attention import Attention, NewAttention
from modules.language_model import WordEmbedding, QuestionEmbedding
from collections import Counter
import numpy as np
from torch.autograd import Variable

import torch.nn.init as init

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
"""
def get_pred(dataloader, indexs, preds, name):
    label2ans = dataloader.dataset.label2ans
    entries = dataloader.dataset.entries
    preds=torch.softmax(preds, dim=1)
    for i in range(len(indexs)):
        index = indexs[i]
        entry = entries[index]
        question = entry['question']
        image_id = entry['image_id']

        if  "color" not in question or "banana" not in question:
            continue
        
        print(name,":")
        
        sets = 'train2014' #or 'val2014'
        image_address = "/home/sdc1/gjl/dataset/COCO_images/%s/COCO_%s_%s.jpg"%(sets, sets, str(image_id).zfill(12))
        if not os.path.exists(image_address):
            sets = 'val2014'
            image_address = "/home/sdc1/gjl/dataset/COCO_images/%s/COCO_%s_%s.jpg"%(sets, sets, str(image_id).zfill(12))
        print(image_address)
        print(question)
        pred = preds[i]
        _, labels = torch.topk(pred, k=3, dim=-1)

        print([pred[label].item() for label in labels])
        print([label2ans[label] for label in labels])
        print('\n')
"""
class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, fusion, num_hid, num_class):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.weight = SimpleClassifier(num_hid, num_hid * 2, num_class, 0.5)
        #self.qweight = SimpleClassifier(num_hid, num_hid * 2, 65, 0.5)
        # self.weight = nn.Parameter(torch.FloatTensor(num_class, num_hid))
        # nn.init.xavier_normal_(self.weight)

    def forward(self, v, q):
        """
        Forward=
        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)

        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr #Final multimodal features, denoted as x in the main paper. This is the UpDn model.

        #This is the bias injecting component, as shown in subsection 3.4 of the main paper
        clf_logits = self.weight(joint_repr)
        #q_logits = self.qweight(joint_repr)
        
        #return att, joint_repr, clf_logits
        return joint_repr, clf_logits

    
class VQBD(nn.Module):
    def __init__(self, num_hid, dataset):
        super(VQBD, self).__init__()
       
        #self.classifier = weight_norm(nn.Linear(dataset.num_ans_candidates* 2, dataset.num_ans_candidates), dim=None)
        if config.use_QBM:
            self.QBM = Question_Bias_Model(num_hid, dataset)
        if config.use_VBM:
            self.VBM = Vision_Bias_Model(num_hid, dataset)
      
    def forward(self, v, q, gen=False):
        if config.use_QBM and config.use_VBM is False:
            pred_QBM = self.QBM(v, q, gen)
            return pred_QBM

        elif config.use_VBM and config.use_QBM is False :
            pred_VBM= self.VBM(v, q, gen)
            return pred_VBM
        
        elif config.use_QBM and config.use_VBM:
            pred_QBM = self.QBM(v, q, gen)
            pred_VBM= self.VBM(v, q, gen)
            
            
            #pred_bias = self.classifier(torch.cat((pred_QBM, pred_VBM), 1))
           
            pred_bias = pred_QBM + pred_VBM
            return  pred_bias


class TargetModel(nn.Module):
    def __init__(self, num_hid, dataset):
        super(TargetModel, self).__init__()
        w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
        q_net = FCNet([num_hid, num_hid])
        v_net = FCNet([dataset.v_dim, num_hid])
        fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    
        self.basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                        fusion, num_hid, dataset.num_ans_candidates)
        #self.margin_model = MarginProduct(num_hid, dataset.num_ans_candidates)
        self.weight = nn.Parameter(torch.FloatTensor(dataset.num_ans_candidates, num_hid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, v, q, use_margin=True):
        joint_repr, clf_logits = self.basemodel(v, q)

        if use_margin:
            cos_logits = F.linear(F.normalize(joint_repr), F.normalize(self.weight))

            target_pred = F.softmax(F.normalize(clf_logits.detach()) / config.temp, 1)
            cos_pred = F.softmax(F.normalize(cos_logits.detach()), 1)
           
            #pred = config.alpha * target_pred + (1-config.alpha) * cos_pred
            pred = target_pred + cos_pred
            return joint_repr, clf_logits, cos_logits, pred 
        else:
            pred = clf_logits
            return joint_repr, clf_logits, None, pred 


class Question_Bias_Model(nn.Module):
    def __init__(self, num_hid, dataset):
        super(Question_Bias_Model, self).__init__()
        self.num_hid = num_hid
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.v_att = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([dataset.v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers
        self.generate = nn.Sequential(
            *block(num_hid//8, num_hid//4),
            *block(num_hid//4, num_hid//2),
            *block(num_hid//2, num_hid),
            nn.Linear(num_hid, num_hid*2),
            nn.ReLU(inplace=False)
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, v, q, gen=True):
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)

        b, c, f = v.shape   # batchsize,36,2048

        # generate from noise
        if gen==True:
            v_z = Variable(torch.cuda.FloatTensor(np.random.normal(0,1, (b,c, 128))))
            v = self.generate(v_z.view(-1, 128))
            v = v.view(b,c,f)

        att = self.v_att(v, q_emb)

        att = nn.functional.softmax(att, 1)
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr

        clf_logits = self.classifier(joint_repr)
        
        
        return clf_logits 

class Vision_Bias_Model(nn.Module):
    def __init__(self, num_hid, dataset):
        super(Vision_Bias_Model, self).__init__()
        self.num_hid = num_hid
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.v_att = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([dataset.v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers
        self.generate = nn.Sequential(
            *block(num_hid//16, num_hid//8),
            *block(num_hid//8, num_hid//4),
            *block(num_hid//4, num_hid//2),
            nn.Linear(num_hid//2, num_hid),
            nn.ReLU(inplace=False)
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, v, q, gen=True):
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)

        b, c = q_emb.shape  # batchsize,1024

        # generate q_emb from noise
        if gen==True:
            q_z = Variable(torch.cuda.FloatTensor(np.random.normal(0,1, (b, 64))))
            q_emb = self.generate(q_z.view(-1, 64))
            

        att = self.v_att(v, q_emb)

        att = nn.functional.softmax(att, 1)
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr

        clf_logits = self.classifier(joint_repr)
        
        #return logits
        return clf_logits 


class Discriminator(nn.Module):
    def __init__(self, num_hid, dataset):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dataset.num_ans_candidates, 1024),
            nn.ReLU(True),
            nn.Linear(num_hid, num_hid//2),
            nn.ReLU(True),
            nn.Linear(num_hid//2, num_hid//4),
            nn.ReLU(True),
            nn.Linear(num_hid//4, 1),
            nn.Sigmoid(),
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


        
# def l2_norm(input, dim=-1):
#     norm = torch.norm(input, dim=dim, keepdim=True)
#     output = torch.div(input, norm)
#     return output


# def build_baseline(dataset, num_hid):
#     w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
#     q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
#     v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
#     q_net = FCNet([num_hid, num_hid])
#     v_net = FCNet([dataset.v_dim, num_hid])
#     fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
#     return BaseModel(w_emb, q_emb, v_att, q_net, v_net,
#                      fusion, num_hid, dataset.num_ans_candidates)


# def build_baseline_newatt(dataset, num_hid):
#     w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
#     q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
#     v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
#     q_net = FCNet([num_hid, num_hid])
#     v_net = FCNet([dataset.v_dim, num_hid])
#     fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
#     basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net,
#                      fusion, num_hid, dataset.num_ans_candidates)
#     margin_model = MarginProduct(num_hid, dataset.num_ans_candidates)
#     #target_model = TargetModel(w_emb, q_emb, v_att, q_net, v_net, fusion, num_hid, dataset.num_ans_candidates)
#     return basemodel, margin_model
