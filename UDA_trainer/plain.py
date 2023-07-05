import torch

def train_plain(batch_iterator, model_fe, model_cls, opt, it, criterion_cls,
            cfg, logger, writer):

    # setting training mode
    opt.zero_grad()

    # get data
    (_, img_src, lbl_src), (_, img_tgt,_) = next(batch_iterator)
    img_src, lbl_src = img_src.cuda(), lbl_src.cuda()

    # forward

    output_src = model_cls(model_fe(img_src), feat=False)
    loss = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=loss.item()
            )

        logger.info(print_str)
