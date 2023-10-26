import time
import argparse
import os, sys
import yaml
import shutil
import numpy as np
import torch
from torch import inverse, nn
import torch.nn.functional as F

from loader import get_dataloader
from models import get_model
from optimizers import get_optimizer, get_scheduler
from metrics import averageMeter, accuracy, percls_accuracy
from losses import get_loss
from utils import get_logger, cvt2normal_state, loop_iterable, calc_coeff
from UDA_trainer import eval

torch.autograd.set_detect_anomaly(True)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    data_loader_tgt = get_dataloader(cfg['data']['target'], ["val"], cfg['testing']['batch_size'])

    n_classes = cfg["data"]["target"]["n_class"]
    write_file = cfg['testing']['write_file']

    # setup model (feature extractor + classifier + discriminator)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg['model']['feature_extractor'], verbose=False).cuda()
    model_cls = get_model(cfg['model']['classifier'], verbose=False).cuda()

    if cfg['testing']['resume'].get('model', None):
        resume = cfg['testing']['resume']
        resume_model = resume['model']
        if os.path.isfile(resume_model):
            logger.info('Loading model from checkpoint {}'.format(resume_model))

            checkpoint = torch.load(resume_model)
            try:
                model_fe.load_state_dict((checkpoint['model_fe_state']))
                model_cls.load_state_dict((checkpoint['model_cls_state']))
            except:
                model_fe.load_state_dict(cvt2normal_state(checkpoint['model_fe_state']))
                model_cls.load_state_dict(cvt2normal_state(checkpoint['model_cls_state']))
                
            logger.info('Loading feature extractor and classifier')
            logger.info("Loaded checkpoint '{}'".format(resume_model))
        else:
            logger.info("No checkpoint found at '{}'".format(resume_model))
            # sys.exit(0)

    if n_gpu>1:
        model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))
        model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))

    eval(data_loader_tgt['val'], model_fe, model_cls, n_classes, write_file, cfg, logger)
    


if __name__ == '__main__':
    global cfg, args, logger

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default='configs/default.yml',
        help='Configuration file to use',
    )
    parser.add_argument("--target" , help="Target file path")
    parser.add_argument("--data_root", type=str, help="Data root")
    parser.add_argument("--norm", type=int, default=0, help="Normalize features [0/1]")
    parser.add_argument("--num_class", type=int, help="Number of classes")
    parser.add_argument("--saved_model", help="Resume training from checkpoint")
    parser.add_argument("--write_file", help="write classwise accuracy to a file.")
    parser.add_argument("--backbone", help="backbone network", choices=["resnet50", "vits16", "vitb16","vitl16"], default="resnet50")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    ## overwrite config parameters
    n_class = args.num_class
    cfg["model"]["classifier"]["n_class"] = n_class
    if args.norm:
        cfg["model"]["classifier"]["norm"] = args.norm
    cfg["data"]["target"]["n_class"] = n_class

    cfg["data"]["target"]["data_root"] = args.data_root
    cfg["data"]["target"]["val"] = args.target

    ## for domain net, seperate test set
    cfg["testing"]["resume"]["model"] = args.saved_model
    cfg['testing']['write_file'] = args.write_file

    cfg["model"]["feature_extractor"]["arch"] = args.backbone

    if args.backbone == "resnet50":
        cfg["model"]["classifier"]["feat_size"] = [2048,256]
    elif args.backbone == "vits16":
        cfg["model"]["classifier"]["feat_size"] = [384,256]
    elif args.backbone.startswith(("vitb16", "vitl16")):
        cfg["model"]["classifier"]["feat_size"] = [768,256]

    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], cfg['exp'])

    print('RUNDIR: {}'.format(logdir))

    logger = get_logger("./test/" , True)
    logger.info('Start logging')

    logger.info(args)

    main()


