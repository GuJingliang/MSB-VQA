import os
import sys
import json
import argparse
from pprint import pprint
import click
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils.utils as utils
import utils.config as config
from train import train, evaluate
import modules.base_model as base_model
from modules.base_model import Discriminator, VQBD, TargetModel
from utils.dataset import Dictionary, VQAFeatureDataset
from utils.losses import Plain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of running epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate for adamax')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--num-hid', type=int, default=1024,
                        help='number of dimension in last layer')
    parser.add_argument('--model', type=str, default='baseline_newatt',
                        help='model structure')
    parser.add_argument('--name', type=str, default='exp0.pth',
                        help='saved model name')
    parser.add_argument('--name-new', type=str, default=None,
                        help='combine with fine-tune')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--fine-tune', action='store_true', default=False,
                        help='fine tuning with our loss')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether resume from checkpoint')
    parser.add_argument('--not-save', action='store_true',default=False,
                        help='do not overwrite the old model')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='test one time')
    parser.add_argument('--eval-only', action='store_true',default=False,
                        help='evaluate on the val set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument('--output', type=str, default='test',)

    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()



    args.output=os.path.join('logs',args.output)
    if args.eval_only is False and args.resume is False:
        if not os.path.isdir(args.output):
            utils.create_dir(args.output)
        else:
            if click.confirm('Exp directory already exists in {}. Erase?'
                                    .format(args.output, default=False)):
                os.system('rm -r ' + args.output)
                utils.create_dir(args.output)

            else:
                if args.load_checkpoint_path is None:
                    os._exit(1)
    print(args)
    print_keys = ['cp_data', 'version', 'train_set', 'loss_type', 
                  'entropy', 'scale', 'alpha_step', 'temp', 
                  'diff_margins', 'use_ce', 'use_supcon', 'use_QBM', 'use_VBM', 'use_margin']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    if 'log' not in args.name:
        args.name = os.path.join(args.output, args.name)
    if args.test_only or args.fine_tune or args.eval_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError("Resuming requires folder name!")
    if args.resume:
        logs = torch.load(args.name)
        print("loading logs from {}".format(args.name))

    # ------------------------DATASET CREATION--------------------
    dictionary = Dictionary.load_from_file(config.dict_path)
    if args.test_only:
        eval_dset = VQAFeatureDataset('test', dictionary)
    else:
        train_dset = VQAFeatureDataset('train', dictionary)
        eval_dset = VQAFeatureDataset('val', dictionary)
    if config.train_set == 'train+val' and not args.test_only:
        train_dset = train_dset + eval_dset
        eval_dset = VQAFeatureDataset('test', dictionary)
    if args.eval_only:
        eval_dset = VQAFeatureDataset('val', dictionary)


    tb_count = 0
    writer = SummaryWriter() # for visualization

    if not config.train_set == 'train+val' and 'LM' in args.loss_fn:
        utils.append_bias(train_dset, eval_dset, len(eval_dset.label2ans))

    # ------------------------MODEL CREATION------------------------
    

    target_model = TargetModel(num_hid=args.num_hid, dataset=eval_dset).cuda()
    target_model.basemodel.w_emb.init_embedding(config.glove_embed_path)

    if config.use_QBM or config.use_VBM:
        vqbd_model = VQBD(num_hid=1024, dataset=train_dset).cuda()
        if config.use_QBM:
            vqbd_model.QBM.w_emb.init_embedding(config.glove_embed_path)
        if config.use_VBM:
            vqbd_model.VBM.w_emb.init_embedding(config.glove_embed_path)
   
        discriminator = Discriminator(num_hid=1024, dataset=train_dset).cuda()
    else:
        vqbd_model = None
        discriminator = None    

    
    # model = nn.DataParallel(model).cuda()
    optim = torch.optim.Adamax(target_model.parameters(), lr=args.lr)
    if config.use_QBM or config.use_VBM:
        optim_VQBD= torch.optim.Adamax(vqbd_model.parameters(), lr=args.lr)
        optim_D = torch.optim.Adamax(discriminator.parameters(), lr=args.lr)
    else:
        optim_VQBD = None
        optim_D = None

    if args.loss_fn == 'Plain':
        loss_fn = Plain()
    else:
        raise RuntimeError('not implement for {}'.format(args.loss_fn))

    # ------------------------STATE CREATION------------------------
    eval_score, best_val_score, start_epoch, best_epoch = 0.0, 0.0, 0, 0
    tracker = utils.Tracker()
    

    if args.resume:
        target_model.load_state_dict(logs['target_model_state'])
        #m_model.load_state_dict(logs['m_model_state'])
        
        optim.load_state_dict(logs['optim_state'])

        if config.use_QBM or config.use_VBM:
            vqbd_model.load_state_dict(logs['vqbd_model_state'])
            discriminator.load_state_dict(logs['discriminator_model_state'])
            optim_VQBD.load_state_dict(logs['optim_VQBD_state'])
            optim_D.load_state_dict(logs['optim_D_state'])

        if 'loss_state' in logs:
            loss_fn.load_state_dict(logs['loss_state'])
            start_epoch = logs['epoch']
            best_epoch = logs['epoch']
            best_val_score = logs['best_val_score']
        if args.fine_tune:
            print('best accuracy is {:.2f} in baseline'.format(100 * best_val_score))
            args.epochs = start_epoch + 10 # 10 more epochs
            for params in optim.param_groups:
                params['lr'] = config.ft_lr

            # if you want save your model with a new name
            if args.name_new:
                if 'log' not in args.name_new:
                    args.name = 'logs/' + args.name_new
                else:
                    args.name = args.name_new
    
    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=8)
    
    if args.test_only or args.eval_only:
        target_model.eval()
        #m_model.eval()
        evaluate(target_model, eval_loader)
    
    if args.test_only or args.eval_only:
        pass
    else:
        train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=8)
        logger = utils.Logger(os.path.join(args.output, 'log.txt'))
        for epoch in range(start_epoch, args.epochs):
            print("training epoch {:03d}".format(epoch))

            train(target_model, vqbd_model, discriminator, optim, optim_VQBD, optim_D, train_loader, loss_fn, tracker, logger, epoch, parse_args)

            if not (config.train_set == 'train+val' and epoch in range(args.epochs - 3)):
                # save for the last three epochs
                write = True if config.train_set == 'train+val' else False
                print("validating after epoch {:03d}".format(epoch))
                target_model.train(False)
                #m_model.train(False)
                eval_score = evaluate(target_model, eval_loader, epoch, write, logger)
                target_model.train(True)
                #m_model.train(True)
                print("eval score: {:.2f} ".format(100 * eval_score))

            if eval_score > best_val_score:
                best_val_score = eval_score
                best_epoch = epoch
            
                results = {
                    'epoch': epoch + 1,
                    'best_val_score': best_val_score,
                    'target_model_state': target_model.state_dict(),
                    #'m_model_state': m_model.state_dict(),
                    'optim_state': optim.state_dict(),
                    'loss_state': loss_fn.state_dict(),
                        
                }
                if config.use_QBM or config.use_VBM:
                    results['vqbd_model_state'] = vqbd_model.state_dict()
                    results['discriminator_model_state'] = discriminator.state_dict()
                    results['optim_VQBD_state'] = optim_VQBD.state_dict()
                    results['optim_D_state'] = optim_D.state_dict()
                    
                if not args.not_save:
                    torch.save(results, args.name)
                    print("save pth")
            print("\n")
        print("best accuracy {:.2f} on epoch {:03d}".format(100 * best_val_score, best_epoch))
        
        logger.write("\nbest accuracy {:.2f} on epoch {:03d}".format(100 * best_val_score, best_epoch))


