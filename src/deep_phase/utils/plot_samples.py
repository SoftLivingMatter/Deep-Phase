import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="`pydantic.error_wrappers:ValidationError` has been moved to `pydantic:ValidationError`.",
)

import pandas as pd
from pathlib import Path

from deep_phase.datasets.bioio_image_dataset import CellImageDataset
from deep_phase.utils.plotting import show_dataset
from deep_phase.utils.data_operations import get_rgb_map, parse_log


def main(eval_csv, log_csv, filter_class, save_name=None):
    log_info = parse_log(log_csv)
    rgb_map = get_rgb_map(log_info['rgb_map'], int(log_info['channels']))
    classes = log_info['training_classes'].split(',') + ['no_call']
    crop_size = int(log_info['crop_size'])

    data = pd.read_csv(eval_csv)
    data = data[data['category'] == filter_class]
    data.category = data.called_class  # for reading in dataset
    counts = data.groupby('category').called_class.count()
    # filter out those classes with fewer than 10 cells to pick from
    data = data[~data.category.isin(counts[counts<10].index)]
    # limit to cut down on loading time for unneeded data
    data = data.groupby('category').sample(5)
    dataset = CellImageDataset(data, rgb_map, crop_size, classes, 0, 0)
    if save_name is not None:
        save_name = Path(eval_csv).parent / f'{save_name}.svg'
    show_dataset(dataset, save_name=save_name, shuffle=False)


if __name__ == "__main__":
    # eval, log, filter, savename
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
