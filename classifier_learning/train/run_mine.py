# https://peluigi.hatenablog.com/entry/2018/08/27/152045
# https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
# https://github.com/burklight/MINE-PyTorch/blob/master/src/mine.py
# https://github.com/MasanoriYamada/Mine_pytorch/blob/master/mine.ipynb

from collections import Counter
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from train.util import save_models, get_model
from train.test import eval_step


# def get_dataloader_with_sampling(config, logger, ld, ud):
#     if config.clustering_method == "simCLR" and config.parent != 'Digit':
#         logger.info('use weighted sampler')
#         edls = np.concatenate([ld.edls, ud.edls])
#         domain_count = list(Counter(edls).values())  # 各推定ドメインの出現個数を取得
#         weight = np.sum(domain_count) / np.array(domain_count)  # ****** 分子と分母逆じゃない？ ******
#         weight_src = torch.DoubleTensor(weight[ld.edls])
#         weight_tgt = torch.DoubleTensor(weight[ud.edls])
#         sampler_src = torch.utils.data.WeightedRandomSampler(weight_src, len(weight_src))
#         sampler_tgt = torch.utils.data.WeightedRandomSampler(weight_tgt, len(weight_tgt))
#         src_train_dataloader = torch.utils.data.DataLoader(ld, config.batch_size, sampler=sampler_src, num_workers=2)
#         tgt_train_dataloader = torch.utils.data.DataLoader(ud, config.batch_size, sampler=sampler_tgt, num_workers=2)
#     else:
#         src_train_dataloader = torch.utils.data.DataLoader(ld, config.batch_size, shuffle=True, num_workers=2)
#         tgt_train_dataloader = torch.utils.data.DataLoader(ud, config.batch_size, shuffle=True, num_workers=2)
    
#     return src_train_dataloader, tgt_train_dataloader


def _get_data_from_minibatch(config, sdata, tdata):
    """ returnの数は他のtraining_modeと異なる """
    src_ims, src_labels, src_ed_feats, _, _ = sdata
    tgt_ims, _, tgt_ed_feats, tgt_hist, tgt_idx = tdata
    
    size = min((src_ims.shape[0], tgt_ims.shape[0]))
    src_ims, tgt_ims = [r[0:size, :, :, :] for r in [src_ims, tgt_ims]]
    src_labels, src_ed_feats, tgt_ed_feats, tgt_hist, tgt_idx = [r[0:size] for r in [src_labels, src_ed_feats, tgt_ed_feats, tgt_hist, tgt_idx]]
    
    src_labels, tgt_hist = src_labels.long(), tgt_hist.long()
    src_ed_feats, tgt_ed_feats = src_ed_feats.float(), tgt_ed_feats.float()

    src_ims, src_labels, src_ed_feats, tgt_ims, tgt_ed_feats, tgt_hist  = [r.to(config.device) for r in [src_ims, src_labels, src_ed_feats, tgt_ims, tgt_ed_feats, tgt_hist]]

    return src_ims, src_labels, src_ed_feats, tgt_ims, tgt_ed_feats, tgt_hist, tgt_idx



def get_mine_loss(config, logger, extracted_feature, ed_feats, mine_net):
    """
        jointはGfより得た特徴とドメイン特徴を対応させる.
        marginalはドメイン特徴の順番をランダムにシャッフルして関係ないドメイン特徴と対応させる

        サンプルの取り方:  類似度が小さい持ったデータからサンプリングする事にする.
        each_not_edl_indices: ドメインラベルが0,1,2の3種類だったら, each_not_edl_indicesの長さは3. それぞれそのドメインに該当しない要素のインデックスのリストを格納. 
        例)  batch_size:8,  バッチのドメインラベル: [0 1 0 1 1 2 0 2]    =>    each_not_edl_indices: [[1, 3, 4, 5, 7], [0, 2, 5, 6, 7], [0, 1, 2, 3, 4, 6]]
    """
    ##############################################################################################################################
    # marginal samplesの取得
    if config.cuda_dir == 0 or config.cuda_dir == 1:
        # 類似度が小さいサンプルを取得
        cosine_similarity = nn.CosineSimilarity(dim=-1)
        similarity_matrix = cosine_similarity(extracted_feature.unsqueeze(1), ed_feats.unsqueeze(0))  # (縦, 横)で類似度行列作成
        marginal_index = similarity_matrix.min(dim=1).indices  # extracted_featureの各行の最小値のインデックスを取得->類似度が最小のed_feat
    if config.cuda_dir == 2 or config.cuda_dir == 3:
        # 完全ランダム
        marginal_index = torch.randint(0, len(ed_feats), size=(len(ed_feats),))
    ##############################################################################################################################
        
    ed_feats = F.normalize(ed_feats, dim=1)    ######(正規化しても最小類似度は変わらない)#####################
    marginal_ed_feats = ed_feats[marginal_index]

    Tj = mine_net(extracted_feature, ed_feats)
    Tj = F.normalize(Tj, dim=1)  ####### これ付けてた方がTotal_Accuracy良くなるっぽい? ####################
    Tj = torch.mean(Tj)
    
    Tm = mine_net(extracted_feature, marginal_ed_feats)
    Tm = F.normalize(Tm, dim=1)  ####### これ付けてた方がTotal_Accuracy良くなるっぽい? ####################
    expTm = torch.mean(torch.exp(Tm))
    
    config.ma_expTm = ((1-config.ma_rate) * config.ma_expTm + config.ma_rate * expTm.detach().item())  # Moving Average expTm

    ### Mutual Information
    # mutual_information = (Tj - torch.log(expTm))
    # loss = -1.0 * (Tj - 1 / (config.ma_expTm.mean()).detach() * expTm )
    mutual_information = (Tj - torch.log(expTm) * expTm.detach() / config.ma_expTm)
    loss = -1.0 * mutual_information

    config.mutual_information = mutual_information  # 処理に全く関係ない．要らない．最大化できているかの確認のみに使う．
    return loss



def mine_step(
    config, logger, feature_extractor, class_classifier, mine_net, source_dataloader, target_dataloader,
    class_criterion, optimizer, scheduler, epoch, ood_class=-1
):
    """
        source_dataloader: img, label, ed_feat, hist(pseudo_label), index
        target_dataloader: img, label, ed_feat, hist(pseudo_label), index
    """
    
    if epoch == config.change_epoch or epoch >= config.change_epoch2:
        target_dataloader.dataset.update_pseudo()

    ### batch loop
    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
        feature_extractor.train()
        class_classifier.train()
        mine_net.train()
        
        # prepare the data
        src_ims, src_labels, src_ed_feats, tgt_ims, tgt_ed_feats, tgt_hist, tgt_idx = _get_data_from_minibatch(config, sdata, tdata)


        # ドメインラベルが全て同じ値だったら, marginalサンプルが取れないので, 逆伝播はしない.
        optimizer.zero_grad()
        
        if epoch < config.change_epoch:
            ### compute the output of source domain and target domain
            src_feature = feature_extractor(src_ims)
            tgt_feature = feature_extractor(tgt_ims)
            ### compute the class loss of src_feature
            class_preds = class_classifier(src_feature)
            class_loss = class_criterion(class_preds, src_labels)
            ### compute the mine loss of src_feature and target_feature
            src_mine_loss = get_mine_loss(config, logger, src_feature, src_ed_feats, mine_net)
            tgt_mine_loss = get_mine_loss(config, logger, tgt_feature, tgt_ed_feats, mine_net)
            mine_loss = src_mine_loss + tgt_mine_loss

            mine_loss = mine_loss * config.mine_weight
            loss = class_loss + mine_loss

        else:
            cat_input = torch.cat([src_ims, tgt_ims])
            cat_label = torch.cat([src_labels, tgt_hist])
            cat_ed_feats = torch.cat([src_ed_feats, tgt_ed_feats])

            feature = feature_extractor(cat_input)
            class_preds = class_classifier(feature)
            class_loss = class_criterion(class_preds, cat_label)
            mine_loss = get_mine_loss(config, logger, feature, cat_ed_feats, mine_net)
            
            class_preds_soft = F.softmax(class_preds, dim=1)  # エラー回避目的で引数dimを指定. 差分確認したから多分あっていると思う.
            avg_probs = torch.mean(class_preds_soft, dim=0)

            p = torch.Tensor(config.prior).to(config.device)
            prior_loss = -torch.sum(torch.log(avg_probs) * p) * config.prior_weight

            mine_loss = mine_loss * config.mine_weight
            loss = class_loss + mine_loss + prior_loss

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

    logger.info(f"\t [Loss]: {loss.item():.4f} \t [Class_Loss]: {class_loss.item():.4f} \t [MINE_Loss]: {mine_loss.item():.4f}")
    logger.info(f"\t [Mutual_Information]: {config.mutual_information}")
    logger.info(f"\t [mine_weight]: {config.mine_weight:.4f},\t [ma_expTm]: {config.ma_expTm:.4f},\t [ma_rate]: {config.ma_rate}")


def run_mine(config, logger, writer, ld, ud, td_list):
    # ランダムサンプリングあり
    # src_train_dataloader, tgt_train_dataloader = sample_dataloader(config, ld, ud)
    # ランダムサンプリングなし
    src_train_dataloader = torch.utils.data.DataLoader(ld, config.batch_size, shuffle=True, num_workers=2)
    tgt_train_dataloader = torch.utils.data.DataLoader(ud, config.batch_size, shuffle=True, num_workers=2)

    # get models
    feature_extractor, class_classifier, mine_net = get_model(config, logger, use_weights=True)
    # set criterions
    class_criterion = nn.CrossEntropyLoss()
    # domain_criterion = nn.CrossEntropyLoss()
    # set optimizer
    if config.optim == "Adam":
        optimizer = optim.Adam([
            {'params': feature_extractor.parameters(), 'lr': config.lr},
            {'params': class_classifier.parameters(), 'lr': config.lr * config.lr_weight},
            {'params': mine_net.parameters(), 'lr': config.lr}
        ], lr=config.lr)
    elif config.optim == "momentum":
        optimizer = optim.SGD([
            {'params': feature_extractor.parameters(), 'lr': config.lr},
            {'params': class_classifier.parameters(), 'lr': config.lr * config.lr_weight},
            {'params': mine_net.parameters(), 'lr': config.lr}
        ], lr=config.lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise NotImplementedError

    # set scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(len(src_train_dataloader), len(tgt_train_dataloader)),
        eta_min=0,  # default value
        last_epoch=-1  # default value
    )

    config.best = 0
    # if config.cuda_dir == 0:
    #     config.ma_rate = 0.5
    #     config.mine_weight = 0.1
    # if config.cuda_dir == 1:
    #     config.ma_rate = 0.5
    #     config.mine_weight = 0.1
    # if config.cuda_dir == 2:
    #     config.ma_rate = 0.5
    #     config.mine_weight = 0.1
    # if config.cuda_dir == 3:
    #     config.ma_rate = 0.5
    #     config.mine_weight = 0.1

    # if config.cuda_dir == 0:
    #     config.ma_rate = 0.5
    #     config.mine_weight = 0.1
    # if config.cuda_dir == 1:
    #     config.ma_rate = 0.1
    #     config.mine_weight = 0.1
    # if config.cuda_dir == 2:
    #     config.ma_rate = 0.1
    #     config.mine_weight = 0.5
    # if config.cuda_dir == 3:
    #     config.ma_rate = 0.5
    #     config.mine_weight = 0.5

    config.ma_rate = 0.5
    config.mine_weight = 1
    config.ma_expTm = 1  # moving_average_expTm
    for epoch in range(config.epochs):
        logger.info(f'Epoch: {epoch+1}/{config.epochs}')

        mine_step(
            config, logger, feature_extractor, class_classifier, mine_net, src_train_dataloader, tgt_train_dataloader,
            class_criterion, optimizer, scheduler, epoch, ood_class=config.num_class-1
        )
        total_acc, accuracy_list, mtx = eval_step(
            config, logger, writer, feature_extractor, class_classifier, td_list, epoch
        )

        save_models(config, [feature_extractor, class_classifier, mine_net], epoch, total_acc, accuracy_list, mtx)
