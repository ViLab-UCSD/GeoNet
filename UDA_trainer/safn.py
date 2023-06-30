import torch
import torch.nn.functional as F

def train_safn(batch_iterator, model_fe, model_cls, opt, it, criterion_cls, criterion_entropy, criterion_safn,
            cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()
    opt.zero_grad()

    # get data
    (img_src, lbl_src), (img_tgt, _) = next(batch_iterator)
    img_src, lbl_src, img_tgt = img_src.cuda(), lbl_src.cuda(), img_tgt.cuda()

    # forward
    # bs_size = img_src.size(0)
    # all_images = torch.cat([img_src, img_tgt], dim=0)
    # output, feature = model_cls(model_fe(all_images), feat=True)
    # output_src, output_tgt = output.split(bs_size)
    # imfeat_src, imfeat_tgt = feature.split(bs_size)

    output_src, imfeat_src = model_cls(model_fe(img_src), feat=True)
    output_tgt, imfeat_tgt = model_cls(model_fe(img_tgt), feat=True)

    ## classification loss
    c_loss = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    ## entropy loss
    ent_loss = criterion_entropy(F.softmax(output_tgt, dim=1))

    ## AFN Loss
    s_afn_loss = criterion_safn(imfeat_src)
    t_afn_loss = criterion_safn(imfeat_tgt)

    ## total loss
    loss = c_loss + s_afn_loss + t_afn_loss + ent_loss

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'S_AFN {s_afn:.4f}\t' \
            'T_AFN {t_afn:.4f}\t' \
            'ENT {entropy:.4f}\t'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=loss.item(),
                s_afn=s_afn_loss.item(),
                t_afn = t_afn_loss.item(),
                entropy=ent_loss.item()
            )

        logger.info(print_str)

    # writer.add_scalar('train/lr', curr_lr, it + 1)
    # writer.add_scalar('train/c_loss', c_loss.item(), it + 1)
    # writer.add_scalar('train/s_afn_loss', s_afn_loss.item(), it + 1)
    # writer.add_scalar('train/t_afn_loss', t_afn_loss.item(), it + 1)
    # writer.add_scalar('train/ent_loss', ent_loss.item(), it + 1)