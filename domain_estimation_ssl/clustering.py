import os
import numpy as np

import torch
from models.baseline_encoder import Encoder
from models.alexnet_simclr import AlexSimCLR
from models.resnet_simclr import ResNetSimCLR

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import normalized_mutual_info_score as NMI
import matplotlib.pyplot as plt

from dataset import get_datasets


def plot_pdf(feats_dim_reduced, dim_red_method, log_dir, c):
    plt.clf()
    plt.figure(figsize=(3,3))
    plt.scatter(feats_dim_reduced[:, 0], feats_dim_reduced[:, 1], s=1, c=c)
    plt.axis("off")
    plt.savefig(os.path.join(log_dir, f'{dim_red_method}_plot.pdf'), box_inches="tight")


def clustering_exec(feats, dim_red_method, clust_method, dataset, log_dir, nmi_output=False):
    if dim_red_method == 'tsne':
        dim_reduce = TSNE(n_components=2, perplexity=30, verbose=1, n_jobs=3)
    elif dim_red_method == 'pca':
        dim_reduce = PCA()
    feats_dim_reduced = dim_reduce.fit_transform(feats)
    ### domain_labels  c=dataset.domain_labelsにより，推定ドメインラベルにより色分けした特徴量の散布図をplot
    # plot_pdf(feats_dim_reduced, dim_red_method, log_dir, c=dataset.domain_labels, outname=f'{dim_red_method}_plot.pdf')
    # # class labels  推定クラスラベルにより色分けした特徴量の散布図をplot
    # plot_pdf(feats_dim_reduced, dim_red_method, log_dir, c=dataset.labels, outname=f'{dim_red_method}_labels_plot.pdf')
    ### 推定domain_labelsをk-meansでクラスタリングし，csvにexport
    if clust_method == 'kmeans':
        domain_clustering = KMeans(n_clusters=len(np.unique(dataset.domain_labels)))
        class_clustering = KMeans(n_clusters=len(np.unique(dataset.labels)))
    elif clust_method == 'gmm':
        domain_clustering = GMM(n_components=len(np.unique(dataset.domain_labels)))
        class_clustering = GMM(n_components=len(np.unique(dataset.labels)))
    domain_cluster = np.array(domain_clustering.fit_predict(feats_dim_reduced))
    class_cluster = np.array(class_clustering.fit_predict(feats_dim_reduced))
    
    np.savetxt(os.path.join(log_dir, f'real_domain.csv'), dataset.domain_labels, delimiter=",")
    np.savetxt(os.path.join(log_dir, f'real_class.csv'), dataset.labels, delimiter=",")
    np.savetxt(os.path.join(log_dir, f'cluster_{dim_red_method}_{clust_method}.csv'), domain_cluster, delimiter=",")
    np.savetxt(os.path.join(log_dir, f"cluster_{dim_red_method}_{clust_method}_class.csv"), class_cluster, delimiter=",")
    # plot_pdf(feats_dim_reduced, dim_red_method, log_dir, c=domain_cluster, outname=f'{dim_red_method}_cluster.pdf')
    # plot_pdf(feats_dim_reduced, dim_red_method, log_dir, c=class_cluster, outname=f'{dim_red_method}_cluster_class.pdf')

    if nmi_output:
        nmi = NMI(dataset.domain_labels, domain_cluster)
        nmi_class = NMI(dataset.labels, class_cluster)
        with open(os.path.join(log_dir, "nmi.txt"), 'w') as f:
            f.write(f'nmi:{nmi}\n')
        with open(os.path.join(log_dir, "nmi_class.txt"), "w") as f:
            f.write(f'nmi_class:{nmi_class}\n')
        print(f'nmi:{nmi}')
        print(f'nmi class:{nmi_class}')

    dataset.edls = domain_cluster
    return dataset


def run_clustering(config):
    if config.dataset.parent == 'Digit':
        model = Encoder()
    elif config.dataset.parent == 'Office31':
        if config.model.base_model == 'alexnet':
            model = AlexSimCLR(config.model.out_dim)
        elif config.model.base_model == 'encoder':
            model = Encoder(3, config.model.out_dim)
        else:
            model = ResNetSimCLR(config.model.base_model, config.model.out_dim)

    if isinstance(config.dataset.grid, list):
        config.dataset.grid = min(config.dataset.grid)

    dataset = get_datasets(config, 'eval')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=56, shuffle=False)

    model.eval()
    model_path = os.path.join(config.log_dir, "checkpoints", "model.pth")
    print(f"  -----  Load Model from {config.log_dir} for clustering  -----")
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    feats = []
    with torch.no_grad():
        for im, _ in dataloader:
            im = im.cuda()
            feat, _ = model(im)
            feats.append(feat.cpu().numpy())
            # feats.append(feat[3].cpu().numpy())  # each layer用
    feats = np.concatenate(feats)

    ### TSNE clustering
    # dataset, domain_cluster = clustering_exec(feats, 'tsne', 'kmeans', dataset, config.log_dir)
    ### PCA clustering
    edls_dataset = clustering_exec(feats, 'pca', 'gmm', dataset, config.log_dir, nmi_output=True)

    # return feats, edls_dataset

