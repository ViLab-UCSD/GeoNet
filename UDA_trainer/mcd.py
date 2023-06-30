import torch
import torch.nn.functional as F

def train_mcd(batch_iterator, model_fe, model_cls, opt, it, criterion_cls, 
                    cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()

    model_fe.zero_grad()
    model_cls.zero_grad()

    # get data
    (img_src, lbl_src), (img_tgt, _) = next(batch_iterator)
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()
        

    # forward
    bs_size = img_src.size(0)
    all_images = torch.cat([img_src, img_tgt], dim=0)

    ## Step A: train all networks to minimize loss on source
    output_1, output_2 = model_cls(model_fe(all_images))
    output_1_src, output_1_tgt = output_1.split(bs_size)
    output_2_src, output_2_tgt = output_2.split(bs_size)
    closs_1 = torch.mean(criterion_cls(output_1_src, lbl_src).squeeze())
    closs_2 = torch.mean(criterion_cls(output_2_src, lbl_src).squeeze())

    output_1_tgt = F.softmax(output_1_tgt, dim=1)
    ent_1_loss = - torch.mean(torch.log(torch.mean(output_1_tgt,0)+1e-6))

    output_2_tgt = F.softmax(output_2_tgt, dim=1)
    ent_2_loss = - torch.mean(torch.log(torch.mean(output_2_tgt,0)+1e-6))

    stage_1_loss = closs_1 + closs_2 + 0.01*(ent_1_loss + ent_2_loss)
    stage_1_loss.backward()
    opt.step()

    ## Step B: train classifier to maximize discrepancy
    model_fe.zero_grad()
    model_cls.zero_grad()

    ## switch off gradients in generator.
    for param in model_fe.parameters():
        param.requires_grad = False
    
    with torch.no_grad():
        features = model_fe(all_images)
    
    output_1, output_2 = model_cls(features)
    output_1_src, output_1_tgt = output_1.split(bs_size)
    output_2_src, output_2_tgt = output_2.split(bs_size)
    closs_1 = torch.mean(criterion_cls(output_1_src, lbl_src).squeeze())
    closs_2 = torch.mean(criterion_cls(output_2_src, lbl_src).squeeze())

    output_1_tgt = F.softmax(output_1_tgt, dim=1)
    ent_1_loss = - torch.mean(torch.log(torch.mean(output_1_tgt,0)+1e-6))

    output_2_tgt = F.softmax(output_2_tgt, dim=1)
    ent_2_loss = - torch.mean(torch.log(torch.mean(output_2_tgt,0)+1e-6))
    
    loss_dis = torch.mean(torch.abs(output_1_tgt-output_2_tgt))

    stage_2_loss = closs_1 + closs_2 + 0.01*(ent_1_loss + ent_2_loss) - loss_dis
    stage_2_loss.backward()
    opt.step()

    ## Step C: train generator to minimize discrepancy
    
    ## switch on gradients in generator
    for param in model_fe.parameters():
        param.requires_grad = True

    ## switch off gradients in classifier
    for param in model_cls.parameters():
        param.requires_grad = False

    for _ in range(cfg["training"]["num_d_iter"]):
        ## check if model_cls has all grad zeros
        model_fe.zero_grad()
        model_cls.zero_grad()

        output_1, output_2 = model_cls(model_fe(all_images))

        output_1_src, output_1_tgt = output_1.split(bs_size)
        output_2_src, output_2_tgt = output_2.split(bs_size)

        output_1_tgt = F.softmax(output_1_tgt, dim=1)
        output_2_tgt = F.softmax(output_2_tgt, dim=1)

        loss_dis = torch.mean(torch.abs(output_1_tgt-output_2_tgt))

        loss_dis.backward()
        opt.step()

    ## switch on gradients in classifier
    for param in model_cls.parameters():
        param.requires_grad = True

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'Dis. Loss {d_loss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=(closs_1 + closs_2).item(), d_loss=loss_dis.item()
            )

        logger.info(print_str)

    # writer.add_scalar('train/lr', curr_lr, it + 1)
    # writer.add_scalar('train/c_loss', (closs_1 + closs_2).item(), it + 1)
    # writer.add_scalar('train/dis_loss', loss_dis.item(), it + 1)
    # writer.add_scalar('train/ent_loss', (ent_1_loss + ent_2_loss).item(), it+1)