import os
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from dataset import get_datasets
from util import get_models_func


def plot_pdfs(config, feats_dim_reduced, dim_red_method, dataset, domain_cluster, class_cluster):
    """ 推定ドメインラベル/クラスラベル により色分けした特徴量の散布図をplot """
    title_data = {
        f'{dim_red_method}_plot.pdf': dataset.true_domain_labels,
        f'{dim_red_method}_labels_plot.pdf': dataset.labels,
        f'{dim_red_method}_cluster.pdf': domain_cluster,
        f'{dim_red_method}_cluster_class.pdf': class_cluster,
    }

    for title, c, in title_data.items():
        plt.clf()
        plt.figure(figsize=(3,3))
        plt.scatter(feats_dim_reduced[:, 0], feats_dim_reduced[:, 1], s=1, c=c)
        plt.axis("off")
        plt.savefig(os.path.join(config.log_dir, title), box_inches="tight")

def save_csvs(config, dataset, feats, domain_cluster, class_cluster):
    title_data = {
        'real_domain.csv': dataset.true_domain_labels,
        'real_class.csv': dataset.labels,
        f'{config.target_dsets_name}_ed_feats.csv': feats,
        f'{config.target_dsets_name}_edls.csv': domain_cluster,
        f"{config.target_dsets_name}_class.csv": class_cluster,
    }
    for title, data, in title_data.items():
        np.savetxt(os.path.join(config.log_dir, title), data, delimiter=",")

def save_nmis(config, logger):
    title_data = f"nmi: {config.nmi}  \nnmi_class: {config.nmi_class}  \ndomain_accuracy: {config.domain_accuracy}  \n"
    with open(os.path.join(config.log_dir, f"result.txt"), 'w') as f:
        f.write(title_data)
    logger.info(title_data)



def clustering_exec(config, logger, feats, dim_red_method, clust_method, dataset):
    if dim_red_method == 'tsne':
        dim_reduce = TSNE(n_components=2, perplexity=30, verbose=1, n_jobs=3)
    elif dim_red_method == 'pca':
        dim_reduce = PCA()
    feats_dim_reduced = dim_reduce.fit_transform(feats)

    if clust_method == 'kmeans':
        domain_clustering = KMeans(n_clusters=config.dataset.num_domains)
        class_clustering = KMeans(n_clusters=config.dataset.num_classes)
    elif clust_method == 'gmm':
        domain_clustering = GMM(n_components=config.dataset.num_domains)
        class_clustering = GMM(n_components=config.dataset.num_classes)
    domain_cluster = np.array(domain_clustering.fit_predict(feats_dim_reduced))
    class_cluster = np.array(class_clustering.fit_predict(feats_dim_reduced))

    # domain_clusterが0,1逆になってもNMIは変わらない.
    nmi = NMI(dataset.true_domain_labels, domain_cluster)
    nmi_class = NMI(dataset.labels, class_cluster)
    if len(config.target_dsets_name.split('_')) == 2:
        domain_accuracy = np.max([accuracy_score(dataset.true_domain_labels, domain_cluster), 1 - accuracy_score(dataset.true_domain_labels, domain_cluster)])
    else:
        domain_accuracy = -1
    config.nmi, config.nmi_class, config.domain_accuracy = nmi, nmi_class, domain_accuracy

    save_nmis(config, logger)
    save_csvs(config, dataset, feats, domain_cluster, class_cluster)
    # plot_pdfs(config, feats_dim_reduced, dim_red_method, dataset, domain_cluster, class_cluster)

    dataset.edls = domain_cluster
    return dataset



def run_clustering(config, logger):
    if isinstance(config.dataset.grid, list):
        config.dataset.grid = min(config.dataset.grid)

    dataset = get_datasets(config, logger, 'eval')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=56, shuffle=False)
    model = get_models_func(config)

    model.eval()
    log_model_path = os.path.join(config.checkpoints_dir, 'model.pth')
    logger.info(f"  -----  Load Model from {log_model_path} for clustering  -----")
    model.load_state_dict(torch.load(log_model_path))

    feats = []
    sigmoid_pseudo_list = []
    with torch.no_grad():
        for im, _, _, _ in dataloader:
            im = im.cuda()
            if config.model.ssl == "simclr":
                feat, _ = model(im)
            elif config.model.ssl == "simsiam":
                feat, _, _ = model(im)
            elif config.model.ssl == "random_pseudo":
                feat, sigmoid_pseudo = model(im)
                sigmoid_pseudo_list.append(sigmoid_pseudo.cpu().numpy())
            feats.append(feat.cpu().numpy())
        feats = np.concatenate(feats)
        
        if config.model.ssl == "random_pseudo":
            sigmoid_pseudo_list = np.concatenate(sigmoid_pseudo_list)
            np.savetxt(os.path.join(config.log_dir, "sigmoid_pseudo.csv"), sigmoid_pseudo_list, delimiter=",")

    ### TSNE clustering
    # dataset, domain_cluster = clustering_exec(feats, 'tsne', 'kmeans', dataset)
    ### PCA clustering
    edls_dataset = clustering_exec(config, logger, feats, 'pca', 'gmm', dataset)

    return feats, edls_dataset

