from pathlib import Path
import pprint
import warnings
import yaml
from collections import Counter
import itertools

warnings.filterwarnings(
    "ignore",
    message="`pydantic.error_wrappers:ValidationError` has been moved to `pydantic:ValidationError`.",
)

import torch
import numpy as np
import pandas as pd
import click

from deep_phase.utils import data_operations as data_ops
from deep_phase.models import modified_resnet as resnet
from deep_phase.train import train_network
from deep_phase.datasets.bioio_image_dataset import CellImageDataset


DEFAULT_ARGUMENTS = dict(
    crop_size=192,
    epochs=50,
    batch_size=128,
    rotation=5,
    noise=0.1,
    augmentation="randaug",
    resnet="resnet34",
    network_type="flat",
    test_train_split=[0.8, 0.1, 0.1],
    latent=2,
    freeze=False,
    limit=None,
    training_classes=['class1', 'class2'],
    mapper="plate",
    output_dir=".",
    channels=4,
    rgb_map="2,3,4:1[25]",
    cell_positions="EnlargedObjects.csv",
    cell_position_format="cellprofiler",  # or processed
    network_name="network.pth",
    starting_network=None,
)

def path_representer(dumper, obj):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(obj))

class MyDumper(yaml.dumper.SafeDumper):
    pass

yaml.add_representer(
    type(Path()),
    path_representer,
    MyDumper,
)

def print_defaults(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(yaml.dump(DEFAULT_ARGUMENTS, Dumper=MyDumper))
    ctx.exit()

@click.command()
@click.option("--config",
              default=None,
              type=click.Path(exists=True, file_okay=True),
              help="Config yaml file.")
@click.option("--base-name", required=True)
@click.option("--overwrite", is_flag=True)
@click.option("--train/--no-train", default=True)
@click.option("--eval/--no-eval", default=True)
@click.option("--print-defaults", is_flag=True, callback=print_defaults, expose_value=False, is_eager=True)
@click.argument("inputs", type=click.Path(exists=True, file_okay=False), nargs=-1)
def main(**cli_args):
    config_args = validate_args(cli_args)

    with open(config_args['config_name'], 'w') as config:
        # add in inputs
        config_args['inputs'] = list(cli_args['inputs'])
        yaml.dump(config_args, config, Dumper=MyDumper, default_flow_style=False)

    rgb_map = data_ops.get_rgb_map(config_args["rgb_map"], config_args["channels"])
    categories = config_args["training_classes"]
    device = get_device()

    dataset_args = (
        rgb_map,
        config_args["crop_size"],
        categories,
        config_args["rotation"],
        config_args["noise"],
        config_args["augmentation"],
        device,
    )

    dataset = []

    mapper = data_ops.build_file_mapper()
    csv_reader = data_ops.read_processed_csv
    for directory in cli_args["inputs"]:
        base_dir = Path(directory)
        if config_args["cell_position_format"] == "cellprofiler":
            if config_args["mapper"] == "plate":
                mapper = data_ops.build_well_mapper(base_dir / "plate_layout.csv")
            elif config_args["mapper"] == "plate-loose":
                mapper = data_ops.build_well_mapper(base_dir / "plate_layout.csv", loose=True)
            csv_reader = lambda file, crop: data_ops.read_cellprofiler_csv(file, crop, mapper)
        dataset.append(
            csv_reader(
                base_dir / config_args["cell_positions"],
                config_args["crop_size"],
            )
        )

    if not dataset:
        raise ValueError(f'No data found in inputs: {config_args["inputs"]}')

    dataset = pd.concat(dataset, ignore_index=True)

    # log args
    with open(config_args["log_name"], "w") as log:
        log.write(f"\n# {len(dataset)} cells total\n")
        net_args = dict(
            device=device,
            resnet=config_args["resnet"],
            freeze=config_args["freeze"],
            latent=config_args["latent"],
            starting_network=config_args["starting_network"],
        )

        if config_args['network_type'] == 'flat':
            network = resnet.build_network(out_classes=len(categories), **net_args)
        else:
            network = resnet.build_multiclass_network(categories, **net_args)

        sub_classes = categories
        if isinstance(sub_classes, dict):
            sub_classes = list(itertools.chain(*categories.values()))

        if cli_args["train"]:
            training_kwargs = dict(
                path=config_args['network_name'],
                epochs=config_args["epochs"],
                batch_size=config_args["batch_size"],
            )
            test_train_split = np.array(config_args['test_train_split'])
            test_train_split /= test_train_split.sum()  # normalize
            train(
                network,
                dataset,
                sub_classes,
                config_args["limit"],
                log,
                test_train_split,
                dataset_args,
                training_kwargs,
            )

        log.flush()

        if cli_args["eval"]:
            resnet.load_network(network, config_args['network_name'], device)
            evaluated_dataset = evaluate(
                network,
                dataset,
                sub_classes,
                dataset_args,
                config_args["batch_size"],
            )
            evaluated_dataset.to_csv(config_args['eval_name'], index=None)

        log.flush()
    return


def validate_args(cli_args):
    config_args = DEFAULT_ARGUMENTS
    if cli_args['config'] is not None:
        config_args.update(yaml.safe_load(open(cli_args['config'], 'r')))

    valid_resnets = ["resnet18", "resnet34", "resnet50"]
    if config_args['resnet'] not in valid_resnets:
        raise ValueError(
            f'Invalid resnet {config_args["resnet"]}, '
            f'expected one of {valid_resnets}')

    valid_networks = ["flat", "hierarchical"]
    if config_args['network_type'] not in valid_networks:
        raise ValueError(
            f'Invalid network type {config_args["network_type"]}, '
            f'expected one of {valid_networks}')

    valid_augmentations = ["randaug", "simple"]
    if config_args['augmentation'] not in valid_augmentations:
        raise ValueError(
            f'Invalid augmentation type {config_args["augmentation"]}, '
            f'expected one of {valid_augmentations}')

    if config_args['network_type'] == 'flat':
        if not isinstance(config_args['training_classes'], list):
            raise ValueError(
                f'Invalid training class type for {config_args["network_type"]} network, '
                'expected list of classes')
        # check for repeated classes
        if len(config_args['training_classes']) != len(set(config_args['training_classes'])):
            counts = Counter(config_args['training_classes'])
            duplicates = [i for i, count in counts.items() if count > 1]
            raise ValueError(
                'Training classes must be unique, '
                f'found {duplicates} more than once.')


    if config_args['network_type'] == 'hierarchical':
        if not isinstance(config_args['training_classes'], dict):
            raise ValueError(
                f'Invalid training class type for {config_args["network_type"]} network, '
                'expected dict of super and sub classes')
        # check for repeated sub classes
        sub_classes = list(itertools.chain(*config_args['training_classes'].values()))
        if len(sub_classes) != len(set(sub_classes)):
            counts = Counter(sub_classes)
            duplicates = [i for i, count in counts.items() if count > 1]
            raise ValueError(
                'Training classes must be unique, '
                f'found {duplicates} more than once.')


    output_base = Path(config_args["output_dir"])
    output_base.mkdir(parents=True, exist_ok=True)

    config_args["network_name"] = output_base / config_args["network_name"]

    warned = False
    if (config_args["network_name"].exists()
            and cli_args["train"] and not cli_args["overwrite"]):
        click.echo(f"WARNING: trying to overwrite {config_args['network_name']}")
        warned = True

    config_args["eval_name"] = output_base / (cli_args['base_name'] + "_eval.csv")
    if (config_args["eval_name"].exists()
            and cli_args["eval"] and not cli_args["overwrite"]):
        click.echo(f"WARNING: trying to overwrite {config_args['eval_name']}")
        warned = True

    config_args["log_name"] = output_base / (cli_args['base_name'] + "_log.csv")
    if config_args["log_name"].exists() and not cli_args["overwrite"]:
        click.echo(f"WARNING: trying to overwrite {config_args['log_name']}")
        warned = True

    config_args["config_name"] = output_base / (cli_args['base_name'] + "_config.yaml")
    if config_args["config_name"].exists() and not cli_args["overwrite"]:
        click.echo(f"WARNING: trying to overwrite {config_args['config_name']}")
        warned = True

    if warned:
        raise ValueError("rerun with `--overwrite` flag to overwrite files")

    if config_args['starting_network']:
        config_args['starting_network'] = Path(config_args['starting_network'])
        if not config_args['starting_network'].exists():
            raise ValueError(f"Starting network {config_args['starting_network']} does not exist")

    return config_args

def get_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Using GPU")
    else:
        dev = "cpu"
        print("Using cpu")
    return torch.device(dev)


def train(network, dataset, categories, limit, log, test_train_split, dataset_args, training_kwargs):
    # ensure all training classes are present
    unique_cats = list(dataset["category"].unique())
    for cat in categories:
        if cat not in unique_cats:
            raise ValueError(
                f"{cat} is expected for training, but not found in dataset.\n"
                f"Found the following categories {unique_cats}"
            )

    dataset['data_usage'] = 'eval'
    to_train = dataset[dataset["category"].isin(categories)]

    log.write(f"# {len(to_train)} cells in classes\n")

    # to balance classes
    replace = False
    if limit:
        min_count = to_train.groupby("category").local_path.count().min()
        if limit != "min":
            min_count_limit = int(limit)
            if min_count_limit > min_count:
                click.echo(
                    f"WARNING: Specified limit of {min_count_limit} is more than minimum {min_count}.\n"
                    "Sampling with replacement and data_usage will not be accurate."
                )
                replace = True
            min_count = min_count_limit

        to_train = data_ops.sample_by_class(to_train, min_count, replace=replace)

    log.write(f"# {len(to_train)} cells for network training\n")

    train, test, usage = data_ops.split_train_test(to_train)
    dataset.loc[usage.index, "data_usage"] = 'oversampled' if replace else usage

    test = CellImageDataset(test, *dataset_args)
    test.eval()
    log.write(f"# {len(test)} test images loaded\n")
    train = CellImageDataset(train, *dataset_args)
    log.write(f"# {len(train)} training images loaded\n")

    train_network(network, train, test, log, **training_kwargs)


def evaluate(network, dataset, categories, dataset_args, batch_size):
    new_cols = []

    has_latent_layers = not isinstance(network.fc[0], torch.nn.modules.activation.ReLU)
    if has_latent_layers:
        new_cols += [f"act_{char}" for _, char in zip(range(network.fc[0].out_features), "xyz")]

    new_cols += [f"cls_{cat}" for cat in categories]

    network.eval()
    sm = torch.nn.Softmax(dim=1)

    dataset[new_cols] = 0.0
    for path in dataset["local_path"].unique():
        rows = dataset["local_path"] == path

        activations = []
        predictions = []
        if has_latent_layers:
            hk = network.fc[0].register_forward_hook(
                lambda m, input, output: activations.append(output.cpu())
            )

        with torch.no_grad():
            images = CellImageDataset(dataset[rows], *dataset_args)
            images.eval()
            for data in torch.utils.data.DataLoader(images, batch_size=batch_size):
                outputs = network(data[0])
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs = sm(outputs)
                predictions.append(outputs.cpu())
        if has_latent_layers:
            hk.remove()
            result = np.hstack(
                (np.concatenate(activations), np.concatenate(predictions))
            )
        else:
            result = np.concatenate(predictions)

        # build result
        dataset.loc[rows, new_cols] = result
    probs = dataset.filter(like="cls_").to_numpy()
    dataset["called_class"] = [categories[i] for i in probs.argmax(axis=1)]
    dataset.loc[(probs < 0.8).all(axis=1), "called_class"] = "no_call"

    return dataset


if __name__ == "__main__":
    main()
