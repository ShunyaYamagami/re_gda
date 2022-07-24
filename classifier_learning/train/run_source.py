import torch
import torch.nn as nn
import torch.optim as optim

from train.util import save_models, get_model
from train.test import eval_step
    

def source_step(config, feature_extractor, class_classifier, source_dataloader, class_criterion, optimizer):
    """ 
    args:
        source_dataloader: img, label, edl, hist(pseudo_label), index
    """
    # setup models
    feature_extractor.train()
    class_classifier.train()

    for source_data in source_dataloader:
        # prepare the data
        im, true_clabel, _, _, _ = source_data
        im, true_clabel = im.to(config.device), true_clabel.cuda(config.device)

        optimizer.zero_grad()

        # compute the output of source domain and target domain
        src_feature = feature_extractor(im)

        # compute the class loss of src_feature
        class_preds = class_classifier(src_feature)
        class_loss = class_criterion(class_preds, true_clabel)
        class_loss.backward()
        optimizer.step()

    # print loss
    prompts = f'\t Class Loss: {class_loss.item():.6f}'
    config.logger.info(prompts)



def run_source(config, ld, td_list):
    src_train_dataloader = torch.utils.data.DataLoader(ld, config.batch_size, shuffle=True, num_workers=2)
    class_criterion = nn.CrossEntropyLoss()
    feature_extractor, class_classifier, _, = get_model(config, use_weights=True)

    if config.optim == "Adam":
        optimizer = optim.Adam([
            {'params': feature_extractor.parameters(), 'lr': config.lr},
            {'params': class_classifier.parameters(), 'lr': config.lr}
        ], lr=config.lr)
    elif config.optim == "momentum":
        optimizer = optim.SGD([
            {'params': feature_extractor.parameters(), 'lr': config.lr},
            {'params': class_classifier.parameters(), 'lr': config.lr}
        ], lr=config.lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise NotImplementedError

    config.best = 0.0
    for epoch in range(config.epochs):
        config.logger.info(f'Epoch: {epoch+1}/{config.epochs}')
        
        source_step(config, feature_extractor, class_classifier, src_train_dataloader, class_criterion, optimizer)
        total_acc, accuracy_list, mtx = eval_step(config, feature_extractor, class_classifier, td_list, epoch)

        save_models(config, [feature_extractor, class_classifier], epoch, total_acc, accuracy_list, mtx)
