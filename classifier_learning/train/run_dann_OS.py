import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from train.util import save_models, get_model
from train.test import eval_step


def get_dataloader_with_sampling(config, logger, ld, ud):
    if config.clustering_method == "simCLR" and config.parent != 'Digit':
        logger.info('use weighted sampler')
        edls = np.concatenate([ld.edls, ud.edls])
        domain_count = list(Counter(edls).values())  # 各推定ドメインの出現個数を取得
        weight = np.sum(domain_count) / np.array(domain_count)  # ****** 分子と分母逆じゃない？ ******
        weight_src = torch.DoubleTensor(weight[ld.edls])
        weight_tgt = torch.DoubleTensor(weight[ud.edls])
        sampler_src = torch.utils.data.WeightedRandomSampler(weight_src, len(weight_src))
        sampler_tgt = torch.utils.data.WeightedRandomSampler(weight_tgt, len(weight_tgt))
        src_train_dataloader = torch.utils.data.DataLoader(ld, config.batch_size, sampler=sampler_src, num_workers=2)
        tgt_train_dataloader = torch.utils.data.DataLoader(ud, config.batch_size, sampler=sampler_tgt, num_workers=2)
    else:
        src_train_dataloader = torch.utils.data.DataLoader(ld, config.batch_size, shuffle=True, num_workers=2)
        tgt_train_dataloader = torch.utils.data.DataLoader(ud, config.batch_size, shuffle=True, num_workers=2)
    
    return src_train_dataloader, tgt_train_dataloader


def _get_data_from_minibatch(config, sdata, tdata):
    src_ims, src_labels, src_edls, _, _ = sdata
    tgt_ims, _, tgt_edls, tgt_hist, tgt_idx = tdata
    
    size = min((src_ims.shape[0], tgt_ims.shape[0]))
    src_ims, src_labels, src_edls = src_ims[0:size, :, :, :], src_labels[0:size], src_edls[0:size]
    tgt_ims, tgt_edls, tgt_hist, tgt_idx = tgt_ims[0:size, :, :, :], tgt_edls[0:size], tgt_hist[0:size], tgt_idx[0:size]
    src_labels, tgt_hist = src_labels.long(), tgt_hist.long()
    src_edls, tgt_edls = src_edls.long(), tgt_edls.long()
    src_ims, src_labels, src_edls, tgt_ims, tgt_edls, tgt_hist = [r.to(config.device) for r in [src_ims, src_labels, src_edls, tgt_ims, tgt_edls, tgt_hist]]

    return src_ims, src_labels, src_edls, tgt_ims, tgt_edls, tgt_hist, tgt_idx




def dann_OS_step(
    config, logger, feature_extractor, class_classifier, domain_classifier, source_dataloader, target_dataloader,
    class_criterion, domain_criterion, optimizer, scheduler, epoch, ood_class=-1
):
    """
        source_dataloader: img, label, edl, hist(pseudo_label), index
        target_dataloader: img, label, edl, hist(pseudo_label), index
    """
    start_steps = epoch * len(source_dataloader)
    total_steps = config.epochs * len(source_dataloader)

    if epoch == config.change_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.next_lr

    if epoch == config.change_epoch or epoch >= config.change_epoch2:
        target_dataloader.dataset.update_pseudo()

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
        feature_extractor.train()
        class_classifier.train()
        domain_classifier.train()

        # prepare the data
        src_ims, src_labels, src_edls,  tgt_ims, tgt_edls, tgt_hist, tgt_idx = _get_data_from_minibatch(config, sdata, tdata)
        
        # setup hyperparameters
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-config.gamma * p)) - 1

        # setup optimizer
        optimizer.zero_grad()

        if epoch < config.change_epoch:
            ### compute the output of source domain and target domain
            src_feature = feature_extractor(src_ims)
            tgt_feature = feature_extractor(tgt_ims)
            ### compute the class loss of src_feature
            class_preds = class_classifier(src_feature)
            class_loss = class_criterion(class_preds, src_labels)
            ### compute the domain loss of src_feature and target_feature
            src_preds = domain_classifier(src_feature, constant)
            tgt_preds = domain_classifier(tgt_feature, constant)
            src_loss = domain_criterion(src_preds, src_edls)
            tgt_loss = domain_criterion(tgt_preds, tgt_edls)
            domain_loss = src_loss + tgt_loss

            loss = class_loss + config.theta * domain_loss

        else:
            cat_input = torch.cat([src_ims, tgt_ims])
            cat_label = torch.cat([src_labels, tgt_hist])
            domain_label = torch.cat([src_edls, tgt_edls])

            feat = feature_extractor(cat_input)
            class_preds = class_classifier(feat)
            class_loss = class_criterion(class_preds, cat_label)

            domain_preds = domain_classifier(feat, constant)
            domain_loss = domain_criterion(domain_preds, domain_label)

            class_preds_soft = F.softmax(class_preds, dim=1)  # エラー回避目的で引数dimを指定. 差分確認したから多分あっていると思う.
            avg_probs = torch.mean(class_preds_soft, dim=0)

            p = torch.Tensor(config.prior).to(config.device)
            prior_loss = -torch.sum(torch.log(avg_probs) * p) * config.prior_weight

            loss = class_loss + config.theta * domain_loss + prior_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            feature_extractor.eval()
            class_classifier.eval()
            eval_feat = feature_extractor(tgt_ims)
            eval_output = class_classifier(eval_feat)

            if epoch < config.change_epoch:
                soft2 = F.softmax(eval_output[:, :-1], dim=1)  # エラー回避目的で引数dimを指定. 差分確認したから多分あっていると思う.
                ent2 = - torch.sum(soft2 * torch.log(soft2 + 1e-6), dim=1)
                
                pred = eval_output.max(1)[1]
                ent_sort, _ = torch.sort(ent2, descending=True)
                threshold = ent_sort[int(len(ent_sort) * config.sigma)]
                new_label = torch.where(ent2 > threshold, torch.ones(pred.size()).long().to(config.device) * ood_class, pred)
            else:
                pred = eval_output.max(1)[1]
                new_label = pred

            target_dataloader.dataset.update_labels(tgt_idx.numpy(), new_label.cpu().numpy())

    logger.info(f"\t [Loss]: {loss.item():.4f} \t [Class Loss]: {class_loss.item():.4f} \t [Domain Loss]: {domain_loss.item():.4f}")


def run_dann_OS(config, logger, writer, ld, ud, td_list):
    ### 何でランダムサンプリングしてんの？する意味なくない？
    # src_train_dataloader, tgt_train_dataloader = sample_dataloader(config, ld, ud)
    ### とりあえずランダムサンプリングは一切しない方法でやってみる.
    src_train_dataloader = torch.utils.data.DataLoader(ld, config.batch_size, shuffle=True, num_workers=2)
    tgt_train_dataloader = torch.utils.data.DataLoader(ud, config.batch_size, shuffle=True, num_workers=2)

    # get models
    feature_extractor, class_classifier, domain_classifier = get_model(config, logger, use_weights=True)
    # set criterions
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    # set optimizer
    if config.optim == "Adam":
        optimizer = optim.Adam([
            {'params': feature_extractor.parameters(), 'lr': config.lr},
            {'params': class_classifier.parameters(), 'lr': config.lr * config.lr_weight},
            {'params': domain_classifier.parameters(), 'lr': config.lr * config.lr_weight}
        ], lr=config.lr)
    elif config.optim == "momentum":
        optimizer = optim.SGD([
            {'params': feature_extractor.parameters(), 'lr': config.lr},
            {'params': class_classifier.parameters(), 'lr': config.lr * config.lr_weight},
            {'params': domain_classifier.parameters(), 'lr': config.lr * config.lr_weight}
        ], lr=config.lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise NotImplementedError

    # set scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(len(src_train_dataloader), len(tgt_train_dataloader)),
        eta_min=0,
        last_epoch=-1
    )

    config.best = 0
    for epoch in range(config.epochs):
        logger.info(f'Epoch: {epoch+1}/{config.epochs}')
        dann_OS_step(
            config, logger, feature_extractor, class_classifier, domain_classifier, src_train_dataloader, tgt_train_dataloader,
            class_criterion, domain_criterion, optimizer, scheduler, epoch, ood_class=config.num_class-1
        )
        total_acc, accuracy_list, mtx = eval_step(
            config, logger, writer, feature_extractor, class_classifier, td_list, epoch
        )

        save_models(config, [feature_extractor, class_classifier, domain_classifier], epoch, total_acc, accuracy_list, mtx)
