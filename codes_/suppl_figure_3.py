#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 08:45:22 2024

@author: spiros
"""

import os
import pickle
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from plotting_functions import fix_names, keep_models
from plotting_functions import my_style, calculate_best_model

# Set the seaborn style and color palette
sns.set_style("white")
plt.rcParams.update(my_style())

datatype = 'fmnist'
dirname_figs = '../FinalFigs_manuscript'
if not os.path.exists(f"{dirname_figs}"):
    os.mkdir(f"{dirname_figs}")

seq = False
seq_tag = "_sequential" if seq else ""
num_layers = 1
data_dir = "../DATA/"
dirname = f"{data_dir}/results_{datatype}_{num_layers}_layer{seq_tag}/"

# Create the figure
fig = plt.figure(
    num=3,
    figsize=(8.27*0.98, 11.69*0.25),
    layout='constrained'
    )

# Create layout
mosaic = [
    ["A", "B"],
]

axd = fig.subplot_mosaic(
    mosaic,
)

for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom')

# Load data (deserialize)
fname_store =  pathlib.Path(f"{dirname}/output_all_dropout_final")
with open(f'{fname_store}.pkl', 'rb') as file:
    results = pickle.load(file)
df_all = fix_names(results['training'])
df_test = fix_names(results['testing'])

# Keep models to plot, i.e., vanilla ANNs.
palette = [
    '#8de5a1',
    '#626262', '#464646', '#383838', '#8d8d8d'
]


model_to_keep = [
    'dANN-R',
    'vANN-0.2',
    'vANN-0.5',
    'vANN-0.8',
    'vANN',
]
df_all_ = keep_models(df_all, model_to_keep)
df_test_ = keep_models(df_test, model_to_keep)
df_all_['train_acc'] *= 100
df_all_['val_acc'] *= 100
df_test_['test_acc'] *= 100

# calculate the stats of
eval_metrics = calculate_best_model(df_test_[df_test_['model'] == 'vANN'])

# Panel A
panel = "A"
sns.lineplot(
    data=df_test_,
    x="trainable_params",
    y="test_loss",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    ax=axd[panel],
    palette=palette,
)
axd[panel].axhline(
    np.min(eval_metrics, axis=0)[0],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].axvline(
    eval_metrics[np.argmin(eval_metrics[:, 0])][-1],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].set_ylabel("test loss")
axd[panel].set_xlabel("trainable params")
axd[panel].set_xscale("log")

# Panel B
panel = "B"
sns.lineplot(
    data=df_test_,
    x="trainable_params",
    y="test_acc",
    hue="model",
    style="model",
    markers=True,
    dashes=False,
    legend=False,
    ax=axd[panel],
    palette=palette,
)
axd[panel].axhline(
    np.max(eval_metrics, axis=0)[1],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].axvline(
    eval_metrics[np.argmax(eval_metrics[:, 1])][-1],
    linestyle="--",
    c="k",
    linewidth=1.5,
)
axd[panel].set_ylabel("test accuracy (%)")
axd[panel].set_xlabel("trainable params")
axd[panel].set_xscale("log")

# fig final format and save
figname = f"{dirname_figs}/supplementray_figure_3"
fig.savefig(
    pathlib.Path(f"{figname}.pdf"),
    bbox_inches='tight',
    dpi=600
)
fig.savefig(
    pathlib.Path(f"{figname}.svg"),
    bbox_inches='tight',
    dpi=600
)
fig.savefig(
    pathlib.Path(f"{figname}.png"),
    bbox_inches='tight',
    dpi=600
)
fig.show()