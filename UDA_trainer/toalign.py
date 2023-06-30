import torch
from utils import calc_coeff

def train_toalign(batch_iterator, model_fe, model_cls, model_d, opt, it, criterion_cls, criterion_d, criterion_hda,
                    cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()
    model_d.train()
    opt.zero_grad()

    # get data
    (img_src, lbl_src), (img_tgt, _) = next(batch_iterator)
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()

    # --------- classification --------------
    outputs_all_src = model_cls(model_fe(img_src))  # [f, y, z]
    assert len(outputs_all_src) == 3, \
        f'Expected return with size 3, but got {len(outputs_all_src)}'
    closs = torch.mean(criterion_cls(outputs_all_src[1], lbl_src).squeeze()) ## classification loss
    focals_src = outputs_all_src[-1]

    # --------- alignment --------------
    outputs_all_src = model_cls(model_fe(img_src), toalign=True, labels=lbl_src)  # [f_p, y_p, z_p]
    outputs_all_tar = model_cls(model_fe(img_tgt))  # [f, y, z]
    assert len(outputs_all_src) == 3 and len(outputs_all_tar) == 3, \
        f'Expected return with size 3, but got {len(outputs_all_src)}'
    focals_tar = outputs_all_tar[-1]

    logits_all = torch.cat((outputs_all_src[1], outputs_all_tar[1]), dim=0)
    if focals_src is not None:
        focals_all = torch.cat((focals_src, focals_tar), dim=0)
    else:
        focals_all = None

    ## adversarial loss
    softmax_output = torch.softmax(logits_all, dim=1)
    ad_out = model_d(softmax_output, calc_coeff(it+1)).squeeze()
    daloss = criterion_d(ad_out=ad_out, softmax_output=softmax_output)

    # hda
    if focals_all is not None:
        hdaloss = criterion_hda(focals_all)
    else:
        hdaloss = (logits_all*0).mean()

    loss = closs + daloss + cfg["training"]["losses"]["loss_toalign"]["coeff"] * hdaloss

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'DALoss {daloss:.4f}\t'\
            'HDALoss {hdaloss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=closs.item(), daloss=daloss.item(), hdaloss=hdaloss.item()
            )

        logger.info(print_str)

    # writer.add_scalar('train/lr', curr_lr, it + 1)
    # writer.add_scalar('train/c_loss', closs.item(), it + 1)
    # writer.add_scalar('train/da_loss', daloss.item(), it + 1)
    # writer.add_scalar('train/hda_loss', hdaloss.item(), it + 1)