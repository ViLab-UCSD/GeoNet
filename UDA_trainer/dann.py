import torch
from utils import calc_coeff

def train_dann(batch_iterator, model_fe, model_cls, model_d, opt, it, criterion_cls, criterion_d, 
                    cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()
    model_d.train()
    opt.zero_grad()

    # get data
    (_, img_src, lbl_src), (_, img_tgt) = next(batch_iterator)
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()

    output_src, imfeat_src = model_cls(model_fe(img_src), feat=True)
    output_tgt, imfeat_tgt = model_cls(model_fe(img_tgt), feat=True)
    feature = torch.cat((imfeat_src, imfeat_tgt), dim=0)
    output = torch.cat((output_src, output_tgt), dim=0)

    ad_out = model_d(feature, calc_coeff(it+1))

    closs = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    # compute loss
    daloss = criterion_d(ad_out=ad_out)
    loss = closs + daloss

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'DALoss {daloss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=closs.item(), daloss=daloss.item()
            )

        logger.info(print_str)