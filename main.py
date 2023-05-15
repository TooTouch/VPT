import numpy as np
import pandas as pd
import os
import random
import wandb

import torch
import argparse
import timm
import logging
import yaml

from stats import dataset_stats
from train import fit
from timm import create_model
from datasets import create_dataloader
from log import setup_default_logging
from models import VPT

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):
    # make save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['dataname'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(cfg['SEED'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    if cfg['MODEL']['prompt_type']:
        model = VPT(
            modelname      = cfg['MODEL']['modelname'],
            num_classes    = cfg['DATASET']['num_classes'],
            pretrained     = True,
            prompt_tokens  = cfg['MODEL']['prompt_tokens'],
            prompt_dropout = cfg['MODEL']['prompt_dropout'],
            prompt_type    = cfg['MODEL']['prompt_type']
        )
    else:
        model = create_model(
            model_name      = cfg['MODEL']['modelname'],
            num_classes    = cfg['DATASET']['num_classes'],
            pretrained     = True,
        )
    model.to(device)
    _logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    # load dataset
    trainset, testset = __import__('datasets').__dict__[f"load_{cfg['DATASET']['dataname'].lower()}"](
        datadir            = cfg['DATASET']['datadir'], 
        img_size           = cfg['DATASET']['img_size'],
        mean               = cfg['DATASET']['mean'], 
        std                = cfg['DATASET']['std']
    )
    
    # sampling 1k 
    sample_df = pd.read_csv(f"{cfg['DATASET']['dataname']}_1k_sample.csv")
    trainset.data = trainset.data[sample_df.sample_index]
    trainset.targets = np.array(trainset.targets)[sample_df.sample_index]
    
    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['OPTIMIZER']['opt_name']](model.parameters(), **cfg['OPTIMIZER']['params'])

    # scheduler
    if cfg['TRAINING']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAINING']['epochs'])
    else:
        scheduler = None

    if cfg['TRAINING']['use_wandb']:
        # initialize wandb
        wandb.init(name=cfg['EXP_NAME'], project='Visual Prompt Tuning', config=cfg)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = cfg['TRAINING']['epochs'], 
        savedir      = savedir,
        log_interval = cfg['TRAINING']['log_interval'],
        device       = device,
        use_wandb    = cfg['TRAINING']['use_wandb'])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visual Prompt Tuning')
    parser.add_argument('--default_setting', type=str, default=None, help='exp config file')    
    parser.add_argument('--modelname', type=str, help='model name')
    parser.add_argument('--dataname', type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100','SVHN','Tiny_ImageNet_200'], help='data name')
    parser.add_argument('--img_resize', type=int, default=None, help='Image Resize')
    parser.add_argument('--prompt_type', type=str, choices=['shallow','deep'], help='prompt type')
    parser.add_argument('--prompt_tokens', type=int, default=5, help='number of prompt tokens')
    parser.add_argument('--prompt_dropout', type=float, default=0.0, help='prompt dropout rate')
    parser.add_argument('--no_wandb', action='store_false', help='no use wandb')

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.default_setting,'r'), Loader=yaml.FullLoader)
    
    d_stats = dataset_stats[args.dataname.lower()]
    
    cfg['MODEL'] = {}
    cfg['MODEL']['modelname'] = args.modelname
    cfg['MODEL']['prompt_type'] = args.prompt_type
    cfg['MODEL']['prompt_tokens'] = args.prompt_tokens
    cfg['MODEL']['prompt_dropout'] = args.prompt_dropout
    cfg['DATASET']['num_classes'] = d_stats['num_classes']
    cfg['DATASET']['dataname'] = args.dataname
    cfg['DATASET']['img_size'] = args.img_resize if args.img_resize else d_stats['img_size']
    cfg['DATASET']['mean'] = d_stats['mean']
    cfg['DATASET']['std'] = d_stats['std']
    cfg['TRAINING']['use_wandb'] = args.no_wandb
        
    cfg['EXP_NAME'] = f"{args.modelname}-{args.prompt_type}-n_prompts{args.prompt_tokens}" if args.prompt_type else args.modelname

    run(cfg)