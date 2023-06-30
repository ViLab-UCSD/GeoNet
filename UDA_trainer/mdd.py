import torch

def train_mdd(batch_iterator, model_fe, model_cls, opt, it, criterion_cls, criterion_mdd, 
                    cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()
    opt.zero_grad()

    # get data
    (img_src, lbl_src), (img_tgt, _) = next(batch_iterator)
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()
        

    # forward
    bs_size = img_src.size(0)
    all_images = torch.cat([img_src, img_tgt], dim=0)
    output, output_adv = model_cls(model_fe(all_images))
    output_src, output_tgt = output.split(bs_size)
    output_src_adv, output_tgt_adv = output_adv.split(bs_size)

    # output_src, output_src_adv = model_cls(model_fe(img_src))
    # output_tgt, output_tgt_adv = model_cls(model_fe(img_tgt))

    closs = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    # compute loss
    mdd_loss = criterion_mdd(output_src, output_tgt, output_src_adv, output_tgt_adv)

    # total loss
    loss = closs + mdd_loss

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'MDD Loss {daloss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=closs.item(), daloss=mdd_loss.item()
            )

        logger.info(print_str)

    # writer.add_scalar('train/lr', curr_lr, it + 1)
    # writer.add_scalar('train/c_loss', closs.item(), it + 1)
    # writer.add_scalar('train/mdd_loss', mdd_loss.item(), it + 1)