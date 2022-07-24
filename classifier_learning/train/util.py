import numpy as np
import os
import torch
from scipy.special import softmax
from models import models, resnet, alexnet


def entropy(output):
    soft = softmax(output, axis=1)
    entropy = - np.sum(soft * np.log(soft + 1e-6), axis=1)
    return entropy


def write_log_file(config, epoch, total_acc, accuracy_list, mtx):
    with open(os.path.join(config.log_dir, "best.txt"), "w") as f:
        f.write(f"Epoch {epoch}\n")
        for domain, acc in accuracy_list:
            f.write(f"{domain}: {100 * acc}\n")
        f.write(f"Total_Accuracy: {total_acc}\n")
        # f.write("confusion_matrix\n" + str(mtx))


def get_model(config, use_weights=True):
    parent, model, num_class, num_domains = config.parent, config.model, config.num_class, config.num_domains
    
    if parent == 'Digit':
        feature_extractor = models.SVHN_Extractor()
        class_classifier = models.SVHN_Class_classifier(num_classes=num_class)
        domain_classifier = models.SVHN_Domain_classifier(num_domains=num_domains)
    else:
        if model == 'resnet':
            feature_extractor, class_classifier, domain_classifier = resnet.get_models(num_class, num_domains, use_weights)
        elif model == 'alexnet':
            feature_extractor, class_classifier, domain_classifier = alexnet.get_models(num_class, num_domains, use_weights)
        else:
            raise ValueError('args.model should be resnet or alexnet')

    feature_extractor, class_classifier, domain_classifier = feature_extractor.to(config.device), class_classifier.to(config.device), domain_classifier.to(config.device) 
    return feature_extractor, class_classifier, domain_classifier
    
    
def save_models(config, save_nets: list, epoch, total_acc, accuracy_list, mtx):
    """ 最新のモデルを保存する. また, best accuracyだったら保存する """
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    latest_filenames = ["feature_extractor_latest", "class_classifier_latest", "domain_classifier_latest"]
    best_filenames = ["feature_extractor_best", "class_classifier_best", "domain_classifier_best"]
    
    for i, save_net in enumerate(save_nets):
        torch.save(save_net.state_dict(), os.path.join(config.checkpoint_dir, f"{latest_filenames[i]}.tar"))

        if total_acc > config.best:
            config.best = total_acc
            torch.save(save_net.state_dict(), os.path.join(config.checkpoint_dir, f"{best_filenames[i]}.tar"))
            write_log_file(config, epoch, total_acc, accuracy_list, mtx)