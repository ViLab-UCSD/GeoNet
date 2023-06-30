import torch
from metrics import averageMeter, accuracy, percls_accuracy

def eval(data_loader, model_fe, model_cls, n_classes, write_file=None, cfg=None, logger=None):

    # setup average meters
    losses = averageMeter()
    top1 = averageMeter()
    top5 = averageMeter()

    # setting eval mode
    model_fe.eval()
    model_cls.eval()

    all_preds = []
    all_labels = []
    cls_num_list_tgt = []
    for (step, value) in enumerate(data_loader):

        print("{}/{}".format(step+1, len(data_loader)), end="\r")

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model_cls(model_fe(image), feat=False)

        # measure accuracy
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1, image.size(0))
        top5.update(prec5, image.size(0))

        # per class accuracy metrics
        all_preds.extend(output.argmax(1).cpu().numpy().tolist())
        all_labels.extend(target.cpu().numpy().tolist())

    classwise_accuracy = percls_accuracy(all_preds, all_labels, num_class=n_classes)

    if write_file:
        with open(write_file, "w") as fh:
            for ca in classwise_accuracy.tolist():
                write_str = "{}\n".format(ca)
                fh.write(write_str)

    if logger:
        logger.info('[Val] Avg. Acc Top 1: {top1.avg:.3f}\tCls. Avg. Acc. Top 1 {cls_avg:.3f}'.format(top1=top1, cls_avg=classwise_accuracy.mean()))
        logger.info('[Val] Avg. Acc Top 5: {top5.avg:.3f}'.format(top5=top5))

    # setting training mode
    model_fe.train()
    model_cls.train()

    return classwise_accuracy.mean().item()
