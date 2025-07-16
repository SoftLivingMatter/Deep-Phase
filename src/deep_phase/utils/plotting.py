import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from deep_phase.utils import data_operations
from deep_phase.models import modified_resnet


def confusion_matrix(data, show_acc=False, save_name=None, plot_order=None, log_norm=False):
    if plot_order is None:
        true_categories = sorted(data["category"].unique())
        called_class = sorted(data["called_class"].unique())
    else:
        true_categories = plot_order
        called_class = plot_order
    heat_data = np.zeros((len(true_categories), len(called_class)), dtype=int)

    for i, true in enumerate(true_categories):
        for j, called in enumerate(called_class):
            heat_data[i, j] = (
                (data["category"] == true) & (data["called_class"] == called)
            ).sum()

            from matplotlib.colors import LogNorm
    if log_norm:
        ax = sns.heatmap(
            heat_data+1,
            annot=False,
            fmt=".0f",
            yticklabels=true_categories,
            xticklabels=called_class,
            norm=LogNorm(),
        )
    else:
        ax = sns.heatmap(
            heat_data,
            annot=True,
            fmt=".0f",
            yticklabels=true_categories,
            xticklabels=called_class,
        )

    if show_acc:
        acc = heat_data.trace() / heat_data.sum()
        ax.set_title(f"Accuracy: {acc:.1%}")
    if save_name:
        plt.savefig(save_name)

    return ax


def called_fraction_plot(data, order=None, rotation=30, save_name=None):
    percents = {}
    for name, dat in data.groupby("category"):
        percents[name] = dat.called_class.value_counts() / dat.called_class.count()
    percents = pd.DataFrame(percents).T.fillna(0)
    if order is not None:
        percents = percents.reindex(order)

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(percents))
    labels = percents.index
    for name in percents.columns:
        ax.bar(labels, percents[name], width=0.8, label=name, bottom=bottom)
        bottom += percents[name]
    ax.legend()
    plt.xticks(rotation=rotation, ha="right")

    if save_name:
        plt.savefig(save_name)

    return ax


def plate_heat_map(
    data,
    well_regex=r"_Well([A-H]\d{1,2})",
    save_name=None,
):
    well = data["FileName_Input"].str.extract(well_regex, expand=True).iloc[:, 0]
    if well.isna().any():
        raise ValueError(
            "Unable to parse well from file "
            + data.loc[well.isna(), "FileName_Input"].iloc[0]
        )
    # standardize well format with leading 0
    # the \g<1> is a capturing group, \10 looks for group 10
    well.replace("^(.)(.)$", r"\g<1>0\2", regex=True, inplace=True)

    dat = pd.DataFrame().assign(
        well=well, counts=data["Count_EnlargedObjects"].astype(int)
    )

    count_per_well = dat.groupby("well")["counts"].sum().reset_index()
    count_per_well[["Row", "Col"]] = count_per_well["well"].str.extract(r"(.)(..)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        count_per_well.pivot(index="Row", columns="Col", values="counts"),
        square=True,
        annot=True,
        ax=ax,
        fmt=".0f",
    )
    if save_name:
        plt.savefig(save_name)


def show_dataset(dataset, max_cols=5, shuffle=True, save_name=None, title=None, fmt='{}'):
    rows = len(np.unique(dataset.sub_label))
    cols = max_cols
    label_df = pd.DataFrame(dataset.sub_label)
    if shuffle:
        label_df = label_df.sample(frac=1)
    label_df = label_df.groupby(0).head(n=cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols + 2, 2 * rows + 1), squeeze=False)

    for plt_row, (lbl, df) in enumerate(label_df.groupby(0)):
        for plt_i, data_i in enumerate(df[0].index):
            axes[plt_row, plt_i].axis("off")
            img, label, _ = dataset[data_i]
            img = img.numpy()
            img = (img - img.min(axis=(1, 2), keepdims=True)) / (
                img.max(axis=(1, 2), keepdims=True)
                - img.min(axis=(1, 2), keepdims=True)
            )
            axes[plt_row, plt_i].imshow(np.moveaxis(img, 0, -1), cmap="gray")
            if title:
                axes[plt_row, plt_i].set_title(fmt.format(dataset.df.loc[data_i][title]))
        axes[plt_row, 0].annotate(
            dataset.get_cat_name(lbl),
            (-0.15, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
        )

    if save_name:
        plt.savefig(save_name)

    return fig, axes


def show_training(log_file, save_name=None):
    data = pd.read_csv(log_file, comment="#")

    long = pd.melt(
        data,
        id_vars="epoch",
        value_vars=["train_loss", "test_loss"],
        value_name="loss",
        var_name="phase",
    )
    long["phase"] = long["phase"].str.removesuffix("_loss")

    data = pd.melt(
        data,
        id_vars="epoch",
        value_vars=["train_acc", "test_acc"],
        value_name="accuracy",
        var_name="phase",
    )
    data["phase"] = data["phase"].str.removesuffix("_acc")
    data = data.merge(long, on=["epoch", "phase"])

    min_epoch = data.iloc[data.loc[data["phase"] == "test", "loss"].argmin()]["epoch"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.lineplot(x="epoch", y="accuracy", data=data, ax=ax[0], hue="phase")
    ax[0].set_title("Accuracy")
    ax[0].axvline(min_epoch, linestyle="--")
    sns.lineplot(x="epoch", y="loss", data=data, ax=ax[1], hue="phase")
    ax[1].set_title("Loss")
    ax[1].axvline(min_epoch, linestyle="--")

    if save_name:
        plt.savefig(save_name)

    return fig, ax


def generate_activation_space(log_file, x_limit, y_limit, samples=100, network_file=None):
    log = data_operations.parse_log(log_file)
    network = modified_resnet.build_network(
        resnet=log['resnet'],
        out_classes=len(log['training_classes']),
    )
    if network_file is None:
        network_file = log['network_name']
    modified_resnet.load_network(network, network_file)

    grid = np.meshgrid(np.linspace(*x_limit, num=samples), np.linspace(*y_limit, num=samples))
    # generate activation space locations
    acts = torch.from_numpy(np.array((grid[0].flatten(), grid[1].flatten()))).to(torch.float)

    # get the output of final layer
    classes = network.fc[1].to(torch.float)(acts.T)
    classes = torch.nn.Softmax(dim=1)(classes).detach().numpy()

    max_class = classes.argmax(axis=1)  # class with highest probability
    # set no calls to -1
    max_class[classes.max(axis=1) < 0.8] = -1

    # return an image like object, number corresponds to class
    return max_class.reshape(*grid[0].shape), grid
