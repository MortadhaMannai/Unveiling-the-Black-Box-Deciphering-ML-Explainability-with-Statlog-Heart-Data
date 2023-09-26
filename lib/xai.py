import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

from .dataset import load_split
from .notebook import FlowLayout
from .definitions import RANDOM_SEED


def plot_pdp_feature(estimator, df_data, feature, title, figsize=None,
                     ice=False):
    from pdpbox import pdp

    pdp_feature = pdp.pdp_isolate(model=estimator,
                                  dataset=df_data,
                                  model_features=df_data.columns[:-1],
                                  feature=feature)
        
    if ice:
        plot = pdp.pdp_plot(pdp_feature, feature,
                            figsize=figsize,
                            cluster=False,
                            n_cluster_centers=None,
                            plot_lines=True,
                            plot_pts_dist=True,
                            plot_params={
                                'title': title,
                                'subtitle': feature,
                                'title_fontsize': 12,
                                'subtitle_fontsize': 10,
                            })
    else:
        plot = pdp.pdp_plot(pdp_feature, feature, cluster=False,
                            figsize=figsize,
                            n_cluster_centers=None,
                            plot_lines=False,
                            plot_pts_dist=False,
                            plot_params={
                                'title': title,
                                'subtitle': feature,
                                'title_fontsize': 12,
                                'subtitle_fontsize': 10,
                            })

    return plot


def plot_pdp_feature_per_split(configs, feature_name,
                               figsize=(4, 6), ice=False):
    plots = []

    for split_index in range(3):
        (train_values, train_labels,
         val_values, val_labels) = load_split(split_index)
        train_full = pd.concat([train_values, train_labels], axis=1)

        estimator = configs[split_index]['estimator']

        plot = plot_pdp_feature(estimator, train_full, feature_name,
                                f'Split {split_index + 1}',
                                figsize=figsize, ice=ice)
        
        plots.append(plot)

    return plots


def plot_pdp_features_per_split(configs, feature_names, figsize=(4, 6),
                                ice=False):
    fl = FlowLayout()

    for c in feature_names:
        plots = plot_pdp_feature_per_split(configs, c, figsize=figsize, ice=ice)

        for (fig, axes) in plots:
            fl.add_figure(fig)
            plt.close()

    fl.PassHtmlToCell()


def plot_pdp_feature_paired(estimators, titles, feature_name,
                            features, labels,
                            figsize=(6, 6), ice=False):
    plots = []

    for i, estimator in enumerate(estimators):
        data_full = pd.concat([features, labels], axis=1)

        plot = plot_pdp_feature(estimator, data_full, feature_name,
                                titles[i],
                                figsize=figsize, ice=ice)

        plots.append(plot)

    return plots


def plot_pdp_features_paired(estimators, titles, feature_names,
                             features, labels,
                             figsize=(6, 6), ice=False):
    fl = FlowLayout()

    for c in feature_names:
        plots = plot_pdp_feature_paired(estimators, titles, c,
                                        features, labels,
                                        figsize=figsize, ice=ice)

        for (fig, axes) in plots:
            fl.add_figure(fig)
            plt.close()

    fl.PassHtmlToCell()


def plot_ale_feature_paired(estimators, titles, feature_name,
                            features, figsize=(6, 6)):
    from alepython import ale

    plots = []

    for i, estimator in enumerate(estimators):
        plot = ale.ale_plot(estimator, features, feature_name,
                            figsize=figsize, title=titles[i])

        plots.append(plot)

    return plots


def plot_ale_features_paired(estimators, titles, feature_names,
                             features, figsize=(6, 6)):
    fl = FlowLayout()

    for c in feature_names:
        plots = plot_ale_feature_paired(estimators, titles, c, features,
                                        figsize=figsize)

        for plot in plots:
            fl.add_figure(plot)
            plt.close()

    fl.PassHtmlToCell()


def plot_ale_feature_per_split(configs, feature_name, figsize=(4, 6)):
    from alepython import ale

    plots = []

    for split_index in range(3):
        (train_values, train_labels, val_values, val_labels) = load_split(split_index)

        estimator = configs[split_index]['estimator']

        plot = ale.ale_plot(estimator, train_values, feature_name,
                            figsize=figsize, title=f'Split {split_index + 1}')

        plots.append(plot)

    return plots


def plot_ale_features_per_split(configs, feature_names, figsize=(4, 6)):
    fl = FlowLayout()

    for c in feature_names:
        plots = plot_ale_feature_per_split(configs, c, figsize=figsize)

        for plot in plots:
            fl.add_figure(plot)
            plt.close()

    fl.PassHtmlToCell()


def train_surrogate(estimator, features):
    from sklearn.linear_model import LogisticRegression

    predictions = estimator.predict(features)

    surrogate_estimator = LogisticRegression(random_state=RANDOM_SEED)
    surrogate_estimator.fit(features, predictions)
    return surrogate_estimator


def estimator_stats(estimator, features, gt_labels):
    predictions = estimator.predict(features)
    score = accuracy_score(gt_labels, predictions)
    cm = confusion_matrix(gt_labels, predictions)

    return score, cm, predictions

