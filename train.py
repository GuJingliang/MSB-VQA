import math
import os
import pickle
import json
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.config as config
from torch.nn import functional as F
from torch.nn import CosineSimilarity
from torch import cosine_similarity
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import time
from tsnecuda import TSNE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

writer_tsne = SummaryWriter('runs/tsne')
Tensor = torch.cuda.FloatTensor

def calc_genb_loss(logits, bias, labels):
    gen_grad = torch.clamp(2 * labels * torch.sigmoid(-2 * labels * bias.detach()), 0, 1)
    loss = F.binary_cross_entropy_with_logits(logits, gen_grad)
    loss *= labels.size(1)
    return loss

def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if isinstance(qtype, tuple):
      i = 0
      dic = {}
      for item in qtype:
          if item not in dic:
              dic[item] = i
              i = i + 1
      tau = 1.0
      qtype = torch.tensor([dic[item] for item in qtype]).cuda()
    feats_filt = F.normalize(feats, dim=1)
    targets_r = qtype.reshape(-1, 1)
    targets_c = qtype.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim/negative_sum)*mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)
    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss

def MarginLoss(clf_logits, cos_logits, m, epoch, label, f1):
    temp = config.temp
    std = 0.1
    easy_margin = False
    s = config.scale

    step = config.alpha_step
    alpha =  (epoch // step) * 0.1
        
    learned_mg = torch.where(m > 1e-12, clf_logits.double(), -1000.0).float()   # clf_logits

    margin = F.softmax(learned_mg / temp, dim=1)   # -m_diff
    
    m = torch.normal(mean=m, std=std)

    if config.diff_margins:
        # m:-m_freq, margin:-m_diff
        m[label != 0] = (1 - alpha) * m[label != 0] + alpha * margin[label != 0]            
    m = 1 - m   # m_comb
    #Compute the AdaArc angular margins and the corresponding logits
    cos_m = torch.cos(m)
    sin_m = torch.sin(m)
    th = torch.cos(math.pi - m)
    mm = torch.sin(math.pi - m) * m
    # --------------------------- cos(theta) & phi(theta) ---------------------------

    # cosine = input
    sine = torch.sqrt((1.0 - torch.pow(cos_logits, 2)).clamp(0, 1))
    phi = cos_logits * cos_m - sine * sin_m # cos(θi + mkfreq[i])
    if easy_margin:
        phi = torch.where(cos_logits > 0, phi, cos_logits)
    else:
        phi = torch.where(cos_logits > th, phi, cos_logits - mm)

    output = s * phi   # s*cos(θi + m_comb[i])

    
    nll = F.log_softmax(output, dim=-1)
    loss = -nll * label
    loss = loss * f1

    return loss.sum(dim=-1).mean()

def compute_score_with_logits(logits, labels):
    t = torch.softmax(logits, dim=1)
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """ Save as a format accepted by the evaluation server. """
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            'question_id': q.item(),
            'answer': a,
        }
        results.append(entry)
    return results

def contrastive_loss(fea, pos_fea, neg_fea, tao=1):

    fea = F.normalize(fea, dim=1)
    pos_fea = F.normalize(pos_fea.detach(), dim=1)
    neg_fea = F.normalize(neg_fea.detach(), dim=1)

    pos_sim = cosine_similarity(fea, pos_fea, dim=-1)
    neg_sim = cosine_similarity(fea, neg_fea, dim=-1)

    logits = torch.exp(pos_sim / tao) / (torch.exp(pos_sim / tao) + torch.exp(neg_sim / tao))
    loss = (-1.0 * torch.log(logits))

    return loss.mean()

def visualize(dataloader, indexs, atts, answers, preds):
    label2ans = dataloader.dataset.label2ans
    entries = dataloader.dataset.entries
    for i in indexs:
        entry = entries[i]

        _, spatials = dataloader.dataset.load_image(entry['image'])
        question = entry['question']
        image_id = entry['image_id']
        att = atts[i]
        
        att_topk_values, att_topk_indices = torch.topk(att.view(-1), k=3)
        
        ans = answers[i]
        pred = preds[i]
        gt_ans = label2ans[torch.argmax(ans)]
        pred_ans = label2ans[torch.argmax(pred)]
        print('question:', question)
        print('gt:', gt_ans)
        print('pred:', pred_ans)
        sets = 'train2014' #or 'val2014'
        image_address = "/home/sdc1/gjl/dataset/COCO_images/%s/COCO_%s_%s.jpg"%(sets, sets, str(image_id).zfill(12))
        if not os.path.exists(image_address):
            sets = 'val2014'
            image_address = "/home/sdc1/gjl/dataset/COCO_images/%s/COCO_%s_%s.jpg"%(sets, sets, str(image_id).zfill(12))
        print(image_address)
        print('\n')
        image = mpimg.imread(image_address)
        #most_att_spatial = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), axis=1)参数为比例而不是具体值
        image_height, image_width = image.shape[0], image.shape[1]
       
        fig, ax = plt.subplots()

        ax.imshow(image)
        color = ['r', 'g', 'b']
        for j in range(len(att_topk_indices)):
            spatial = spatials[att_topk_indices[j]]
            x, y, width, height = spatial[0], spatial[1], spatial[4], spatial[5]
            x, y, width, height = x * image_width, y * image_height, width * image_width, height * image_height
            rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor=color[j], facecolor='none')
            ax.add_patch(rect)
            
            
            att_value = att_topk_values[j].item()
            ax.text(x, y, f'{att_value:.2f}', color=color[j], fontsize=12, weight='bold')
            
        
        plt.axis('off')
        plt.show()

def var_name(var,all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

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


def train(model, vqbd_model, discriminator, optim, optim_VQBD, optim_D, train_loader, loss_fn, tracker, logger, epoch, args):
    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))
    #alpha_trk = tracker.track('alpha', tracker.MovingMeanMonitor(momentum=0.99))
    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()  
    starttime = time.time()
    for v, q, a, mg, bias, qids, f1, qtype, answer_type, indexs in loader:        
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        mg = mg.cuda()
        bias = bias.cuda()
        f1 = f1.cuda()
        valid = torch.full([v.size(0), 1], 1.0).cuda().requires_grad_(False)
        fake = torch.full([v.size(0), 1], 0.0).cuda().requires_grad_(False)
        loss = 0

        # get T model output
        optim.zero_grad()
        joint_repr, clf_logits, cos_logits, pred = model(v, q, config.use_margin)       
        
     
        
        if config.use_margin:
            loss += MarginLoss(clf_logits, cos_logits, mg, epoch, a, f1)
            
 
            
        if config.use_QBM or config.use_VBM:
            # train G model
            vqbd_model.train(True)
            pred_g = vqbd_model(v, q, gen=True)

            gt_loss = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
            gt_loss *= a.size(1)

            vae_preds = discriminator(pred_g)
            main_preds = discriminator(clf_logits)

            g_distill = kld(pred_g, clf_logits.detach())
            dsc_loss = bce(vae_preds, valid) + bce(main_preds, valid)
            
            g_loss = gt_loss + dsc_loss + g_distill*5

            optim_VQBD.zero_grad()
            g_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vqbd_model.parameters(), 0.25)
            optim_VQBD.step()
            
            # done training VQBD
            
            # train the discriminator
            vae_preds = discriminator(pred_g.detach())
            main_preds = discriminator(clf_logits)        
            dsc_loss = bce(vae_preds, fake) + bce(main_preds, valid)
            optim_D.zero_grad()
            dsc_loss.backward(retain_graph=True)
            optim_D.step()
            # done training the discriminator

            # use VQBD to train the robust model
            vqbd_model.train(False)
            
            pred_g = vqbd_model(v, q, gen=False)
            #get_pred(train_loader, indexs, pred_g, "pred_g")

            grad_loss = calc_genb_loss(clf_logits, pred_g, a)
            loss += grad_loss
            
      
        if config.use_ce:
            ce_loss = - F.log_softmax(clf_logits, dim=-1) * a
            ce_loss = ce_loss.sum(dim=-1).mean() 
            loss += ce_loss

        
        
        if config.use_supcon:
            gt = torch.argmax(a, 1)
            loss += compute_supcon_loss(joint_repr, gt)


        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()

        # done training target model

        
        
        
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        #alpha_trk.append(model.alpha.item())
        loader.set_postfix(loss=fmt(loss_trk.mean.value), acc=fmt(acc_trk.mean.value))
    
    endtime = time.time()
    logger.write('Epoch %d, time: %.2f' % (epoch, endtime - starttime))
    logger.write('\ttrain:')
    logger.write('\t\tscore: %.2f, loss: %.2f' % (acc_trk.mean.value * 100, loss_trk.mean.value))


#Evaluation code
def evaluate(model, dataloader, epoch=0, write=True, logger=None):
    score_all = 0
    results = []  # saving for evaluation
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 

    ####################
    '''
    embs = None
    labels = None
    answer_types = ()
    q_types = ()
    '''
    ####################
    #count = torch.zeros(3129)
    for v, q, a, mg, _, qids, _, qtype, answer_type, indexs in tqdm(dataloader, ncols=0, leave=True):
        v = v.cuda()
        q = q.cuda()
        mg = mg.cuda()
        a = a.cuda()

       
        joint_repr, clf_logits, cos_logits, pred = model(v, q, config.use_margin)    
        
        
        #get_pred(dataloader, indexs, clf_logits, "clf_logits")
        #get_pred(dataloader, indexs, cos_logits, "cos_logits")
        #get_pred(dataloader, indexs, pred, "pred")
       
     
        batch_score = compute_score_with_logits(pred, a.cuda())
        #################################
        '''
        answer_types += answer_type
        q_types += qtype
        
        if embs is None:
            embs = joint_repr.detach().cpu()
            labels = a.detach().cpu()
        else:
            embs = torch.cat((embs, joint_repr.detach().cpu()), 0)
            labels = torch.cat((labels, a.detach().cpu()), 0)
        
        '''
        #################################
        score_all += batch_score.sum()

        qids = qids.detach().cpu().int().numpy()
        
        for j in range(len(answer_type)):
         

            typ = answer_type[j]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

        if write:
            results = saved_for_eval(dataloader, results, qids, pred)
    
        # visualize the attention
        # visualize(dataloader, indexs, att, a, pred)  
         
    ############################
    '''       
    with open('q_types.pkl', 'wb') as f:
            pickle.dump(q_types, f)
    with open('answer_types.pkl', 'wb') as f:
            pickle.dump(answer_types, f)
    torch.save(embs,"embs_ce.pth") 
    #torch.save(labels,"labels.pth")
    ''' 
    ############################
    
    score_all = score_all / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_number /= total_number
    score_other /= total_other

    if write:
        # Used to visualize question type accuracy
        print("saving prediction results to disk...")
        result_file = 'vqa_{}_{}_{}_results.json'.format(
            config.task, config.test_split, config.version)
        result_file = os.path.join('data/Results/', result_file)
        with open(result_file, 'w') as fd:
            json.dump(results, fd)
    

    print('eval score: %.2f, yn score: %.2f num score: %.2f other score: %.2f' % (100 * score_all, 100 * score_yesno, 100 * score_number, 100 * score_other))
    if logger is not None:
        logger.write('\tevel:')
        logger.write('\t\tall score: %.2f, yn score: %.2f num score: %.2f other score: %.2f' % (100 * score_all, 100 * score_yesno, 100 * score_number, 100 * score_other))
    return score_all


