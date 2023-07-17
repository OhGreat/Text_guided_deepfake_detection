import csv
import json
from os.path import exists

from main import main
from src.utils.options import get_opts


if __name__ == "__main__":
    """
    Runs the main.py file for evaluating each dataset separately and collecting statistics in a csv file.
    """

    opts = get_opts()

    # assert required options ahve been defined
    assert opts.eval_file is not None
    assert opts.data_paths is not None

    print(f"Saving current results in {opts.eval_file}")

    # prepare results file
    with open(opts.eval_file, 'a') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["dataset", "AUC", "accuracy",
                           "precision", "recall", "tp", "fp", "tn", "fn"])

    # open file with datasets structure
    with open(opts.data_paths) as d:
        data_paths = json.load(d)

    for name, paths in data_paths.items():

        # fix datasets in options
        opts.real_eval = paths["real"]
        opts.fake_eval = paths['fake']

        # run evaluation
        results = main(opts)[0]

        # log results
        to_append = {'name': name}
        to_append.update(results)

        with open(opts.eval_file, 'a') as file:
            csvwriter = csv.DictWriter(file, fieldnames=list(to_append.keys()))
            csvwriter.writerow(to_append)
