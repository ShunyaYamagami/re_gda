"""
Test the model with target domain
"""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score


def test_step(config, feature_extractor, class_classifier, target_dataloader):
    """
    args:
        target_dataloader: img, label, edl, hist(pseudo_label), index
    """
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()

    preds = []
    true_labels = []
    outputs = []

    for tdata in target_dataloader:
        im, true_label, _, _, _ = tdata
        im, true_label = im.to(config.device), true_label.to(config.device)

        feat = feature_extractor(im)
        output2 = class_classifier(feat)
        pred = output2.data.max(1, keepdim=True)[1].squeeze(1)

        outputs.append(output2.cpu().numpy())
        preds.append(pred.cpu().numpy())
        true_labels.append(true_label.cpu().numpy())

    outputs = np.concatenate(outputs)
    preds = np.concatenate(preds)
    true_labels = np.concatenate(true_labels)

    return outputs, true_labels, preds




def eval_step(config, feature_extractor, class_classifier, td_list, epoch):
    """
    args:
        td_list: [['amazon', <dataset.LabeledDataset>], ['dslr', <dataset.LabeledDataset>]]
    """
    labels_list = []
    preds_list = []
    accuracy_list = []

    for domain, td in td_list:
        if len(td) == 0:
            continue
        tgt_test_dataloader = torch.utils.data.DataLoader(td, batch_size=56, shuffle=False, num_workers=1)
        with torch.no_grad():
            outputs, labels, preds = test_step(config, feature_extractor, class_classifier, tgt_test_dataloader)

        accuracy = accuracy_score(labels, preds)
        
        labels_list.append(labels)
        preds_list.append(preds)
        accuracy_list.append((domain, accuracy))

        correct_num = np.sum(np.equal(preds, labels))  # 正解データ数
        config.logger.info(f'\t Target_Accuracy [{domain}]: {correct_num}/{len(labels)} ({100 * accuracy:.2f}%)')
        config.writer.add_scalar(f'logs/{domain}/Target_Accuracy', accuracy, epoch)

    labels_list = np.concatenate(labels_list)
    preds_list = np.concatenate(preds_list)


    total_acc = accuracy_score(labels_list, preds_list)
    config.logger.info(f"\t Total_Accuracy: {total_acc:.4f}")
    config.writer.add_scalar('logs/Total_Accuracy', total_acc, epoch)

    mtx = confusion_matrix(labels_list, preds_list, labels=np.arange(config.num_class))

    return total_acc, accuracy_list, mtx