import urllib.parse
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from pathlib import Path
import yaml


def read_cellprofiler_csv(
    file,
    crop_size,
    category_mapper,
):
    data = pd.read_csv(file)
    data = data[
        (data["Location_Center_X"] - crop_size > 0)
        & (data["Location_Center_Y"] - crop_size > 0)
        & (data["Location_Center_X"] + crop_size < data["Metadata_SizeX"])
        & (data["Location_Center_Y"] + crop_size < data["Metadata_SizeY"])
    ]

    result = pd.DataFrame().assign(
        local_path=data["Metadata_FileLocation"].map(
            lambda path: urllib.parse.unquote(urllib.parse.urlparse(path).path)
        ),
        category=category_mapper(data),
        series=data["Metadata_Series"],
        center_x=data["Location_Center_X"].round().astype(int),
        center_y=data["Location_Center_Y"].round().astype(int),
    )

    # remove unknown categories
    result = result[~result["category"].isna()]

    return result


def read_processed_csv(
    file,
    crop_size,
):
    '''
    Read a csv file that is already parsed.

    Performs similar checks and modifications to read_cellprofiler_csv but
    does not remap categories or rename columns.  In addition to the standard
    required columns, also need size_x and size_y

    If series is not present it is set to 0
    '''

    if isinstance(file, pd.DataFrame):
        data = file
    else:
        data = pd.read_csv(file)
    data = data[
        (data["center_x"] - crop_size > 0)
        & (data["center_y"] - crop_size > 0)
        & (data["center_x"] + crop_size < data["size_x"])
        & (data["center_y"] + crop_size < data["size_y"])
    ].copy()

    data['center_x'] = data["center_x"].round().astype(int)
    data['center_y'] = data["center_y"].round().astype(int)

    if 'series' not in data.columns:
        data['series'] = 0

    return data


def build_file_mapper():
    def mapper(data):
        return data["Metadata_FileLocation"].map(
            lambda path: Path(
                urllib.parse.unquote(urllib.parse.urlparse(path).path)
            ).stem
        )

    return mapper


def read_plate_layout(csv_layout):
    # read plate layout
    rows = "ABCDEFGH"
    cols = [f"{i:02}" for i in range(1, 13)]
    with open(csv_layout) as layout:
        well_to_category = {
            f"{row}{col}": value.strip()
            for row, line in zip(rows, layout)
            for col, value in zip(cols, line.split(","))
            if value.strip()
        }
    return well_to_category


def build_well_mapper(
    csv_layout,
    well_regex=r"_Well([A-H]\d{1,2})",
    fuzzy=None,
):
    well_to_category = read_plate_layout(csv_layout)

    def mapper(data):
        well = (
            data["Metadata_FileLocation"]
            .str.extract(well_regex, expand=True)
            .iloc[:, 0]
        )
        if well.isna().any() and fuzzy is None:
            raise ValueError(
                "Unable to parse well from file "
                + data.loc[well.isna(), "Metadata_FileLocation"].iloc[0]
            )

        # standardize well format with leading 0
        # the \g<1> is a capturing group, \10 looks for group 10
        well.replace("^(.)(.)$", r"\g<1>0\2", regex=True, inplace=True)
        # build mapping function
        categories = well.map(well_to_category)

        if categories.isna().any():
            if fuzzy == "loose":  # replace unknown wells with filename
                file_mapper = build_file_mapper()
                categories[categories.isna()] = file_mapper(data[categories.isna()])
            elif fuzzy == "ignore":  # ignore, inform user
                print("Ignoring the following file without matches in plate layout")
                print(data.loc[categories.isna(), "Metadata_FileLocation"].unique())
            else:
                raise ValueError(
                    "Unable to match category for "
                    + data.loc[categories.isna(), "Metadata_FileLocation"].iloc[0]
                )

        return categories

    return mapper


def sample_by_class(df, n, by="category", replace=False):
    return df.groupby(by).sample(n=n, random_state=12345, replace=replace)


def split_train_test(df, proportion=(0.8, 0.1, 0.1), seed=12345):
    train, test, validation = torch.utils.data.random_split(
        df, proportion, generator=torch.Generator().manual_seed(seed)
    )

    usage = df['data_usage'].copy()
    usage.iloc[train.indices] = 'train'
    usage.iloc[test.indices] = 'test'
    usage.iloc[validation.indices] = 'validation'

    test = df.iloc[test.indices]
    train = df.iloc[train.indices]

    return train, test, usage


def parse_log(log_file):
    if str(log_file).endswith('csv'):  # old style log
        def parse_line(line):
            tokens = line.split(':')
            val = ':'.join(tokens[1:])
            return tokens[0].split("'")[1], val.strip(" ,'\n}")

        result = {}
        with open(log_file) as log:
            raw_config = dict(
                parse_line(line)
                for line in log
                if line.startswith('#') and ':' in line
            )
        raw_config['training_classes'] = raw_config['training_classes'].split(',')

    else:  # yaml
        return yaml.safe_load(open(log_file, 'r'))



def get_rgb_map(channel_map, channels):
    result = np.zeros((3, channels))
    match = re.match(
        r"(?P<red>\d+),(?P<green>\d+),(?P<blue>\d+)"  # basic
        r"(:(?P<white>\d+)\[(?P<fraction>\d+)])?",  # white
        channel_map,
    )
    if not match:
        raise ValueError(f"Unable to match rgb_map {channel_map}")
    vals = match.groupdict()
    white_val = int(vals["fraction"]) / 100 if vals["fraction"] else 0
    color_val = 1 - white_val

    for i, color in enumerate(("red", "green", "blue")):
        color = int(vals[color])
        assert 0 <= color <= channels, f"Channel {color} out of range in {channel_map}"
        if color == 0:
            continue
        result[i, color - 1] = color_val

    if vals["white"]:
        color = int(vals["white"])
        assert 0 <= color <= channels, f"Channel {color} out of range in {channel_map}"
        if color != 0:
            # this is to ensure the rows sum to 1
            # result[:, color-1] = 1 - result.sum(axis=1)
            # this keeps the grey value grey
            result[:, color - 1] = white_val

    return result


def add_response(dataset, standards_csv, standards=None, mapping=None, map_by='category'):
    '''Add activation space response.

    loads standards csv and takes average activation as vector points
    if standards is set, limit to standards with the given category
    mapping can be 
        - a tuple of (untreated, treated)
        - a dict of category: (untreated, treated)
        - if the tuple is of size 1, the untreated sample will be assumed as (0,0)
    If unset, will group dataset by category and try to map with (untreated, category)
    '''

    if isinstance(standards_csv, pd.DataFrame):
        stds = standards_csv
    else:
        stds = pd.read_csv(standards_csv)
    stds['category'] = stds['category'].str.lstrip('\ufeff')
    if standards is not None:
        stds = stds[stds['category'].isin(standards)]

    centers = stds.groupby('category')[['act_x', 'act_y']].mean()

    result = dataset.copy()
    result['response'] = 0

    def response(data, *args):
        if len(args) > 1:
            untreated = centers.loc[args[0], ['act_x', 'act_y']].to_numpy()
            treated = centers.loc[args[1], ['act_x', 'act_y']].to_numpy()
        else:
            # passed a single value, set untreated to 0,0
            untreated = np.array([0,0])
            treated = centers.loc[args[0], ['act_x', 'act_y']].to_numpy()
        d = data[['act_x', 'act_y']].to_numpy() - untreated
        return np.dot(d, treated-untreated) / np.dot(treated-untreated, treated-untreated)

    if mapping is None:
        for category in result[map_by].unique():
            result.loc[result[map_by] == category, 'response'] = response(
                result[result[map_by] == category], 'Untreated', category)

    elif isinstance(mapping, tuple):
        result['response'] = response(result, *mapping)

    else:
        for category, keys in mapping.items():
            result.loc[result[map_by] == category, 'response'] = response(
                result[result[map_by] == category], *keys)

    return result


def recall_dataset(dataset, no_call_threshold=0.8):
    probabilities = dataset.filter(like="cls_")
    categories = [col[4:] for col in probabilities.columns]  # strip "cls_"
    result = dataset.copy()
    result['called_class'] = [categories[i] for i in probabilities.to_numpy().argmax(axis=1)]
    result.loc[(probabilities < no_call_threshold).all(axis=1), "called_class"] = "no_call"
    return result
