import torch

def train_plain(batch_iterator, model_fe, model_cls, opt, it, criterion_cls,
            cfg, logger, writer):

    # print("{}/{}".format(it+1,cfg['training']['iteration']), end="\r")

    # setting training mode
    opt.zero_grad()

    # get data
    (img_src, lbl_src), (img_tgt, _) = next(batch_iterator)
    img_src, lbl_src = img_src.cuda(), lbl_src.cuda()
    # img_tgt = img_tgt.cuda()

    # forward

    if cfg["training"]["freeze_encoder"]: ## Linear Probing
        with torch.no_grad():
            output_fe = model_fe(img_src).detach()
        output_src = model_cls(output_fe, feat=False)
    else:
        output_src = model_cls(model_fe(img_src), feat=False)
    # output_tgt = model_cls(model_fe(img_tgt), feat=False)

    # output = model_cls(model_fe(torch.cat([img_src, img_tgt])), feat=False)
    # output_src = output[:len(img_src)]

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

    # writer.add_scalar('train/lr', curr_lr, it + 1)
    # writer.add_scalar('train/c_loss', loss.item(), it + 1)
