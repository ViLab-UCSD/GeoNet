import torch
from utils import calc_coeff

def train_mcc(batch_iterator, model_fe, model_cls, model_d, opt, it, criterion_cls, criterion_d, criterion_cov,
                    cfg, logger, writer):
    """ Refer: https://github.com/thuml/Versatile-Domain-Adaptation/blob/master/pytorch/train_image_office.py
    """

    # setting training mode
    model_fe.train()
    model_cls.train()
    model_d.train()
    opt.zero_grad()

    # get data
    (img_src, lbl_src), (img_tgt, _) = next(batch_iterator)
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()

    # forward
    # bs_size = img_src.size(0)
    # all_images = torch.cat([img_src, img_tgt], dim=0)
    # output, feature = model_cls(model_fe(all_images), feat=True)
    # output_src, output_tgt = output.split(bs_size)

    output_src, imfeat_src = model_cls(model_fe(img_src), feat=True)
    output_tgt, imfeat_tgt = model_cls(model_fe(img_tgt), feat=True)
    feature = torch.cat((imfeat_src, imfeat_tgt), dim=0)
    output = torch.cat((output_src, output_tgt), dim=0)

    softmax_output = torch.softmax(output, dim=1)

    ## classification loss
    closs = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    ## adversarial loss: CDAN
    op_out = torch.bmm(softmax_output.detach().unsqueeze(2), feature.unsqueeze(1))
    ad_out = model_d(op_out.view(-1, softmax_output.size(1) * feature.size(1)), calc_coeff(it+1)).squeeze()
    daloss = criterion_d(ad_out=ad_out)

    ## covariance loss
    cov_loss = criterion_cov(output_tgt)

    loss = closs + daloss + cov_loss

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'DALoss {daloss:.4f}\t' \
            'COVLoss {covloss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=closs.item(), daloss=daloss.item(), covloss=cov_loss.item()
            )

        logger.info(print_str)

    # writer.add_scalar('train/lr', curr_lr, it + 1)
    # writer.add_scalar('train/c_loss', closs.item(), it + 1)
    # writer.add_scalar('train/da_loss', daloss.item(), it + 1)
    # writer.add_scalar('train/cov_loss', cov_loss.item(), it + 1)