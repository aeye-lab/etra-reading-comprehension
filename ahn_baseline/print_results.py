from __future__ import annotations

import argparse
import os

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name', type=str, required=True,
        help='specify to only print a certain model',
    )
    parser.add_argument(
        '--label', type=str, required=True,
        help='which labeling to use',
    )
    args = parser.parse_args()
    score_dict = dict()
    if args.model_name == 'rnn':
        latex_start_string = r'&RNN~\cite{Ahn2020TowardsBehavior}~(full page)'
    elif args.model_name == 'dense':
        latex_start_string = r'&Regression~\cite{Ahn2020TowardsBehavior}~(full page)'
    elif args.model_name == 'cnn1':
        latex_start_string = r'&CNN~\cite{Ahn2020TowardsBehavior}~(full page)'
    for file_name in os.listdir('paper_results'):
        if args.model_name is None and args.label is None:
            df = pd.read_csv(f'paper_results/{file_name}')
            print(
                file_name, fr"{df['avg_auc'].values[0]:.6f}\pm{df['std_auc'].values[0]:.6f}",
            )
        elif file_name.startswith(args.model_name) and args.label is None:
            df = pd.read_csv(f'paper_results/{file_name}')
            print(
                file_name, fr"{df['avg_auc'].values[0]:.6f}\pm{df['std_auc'].values[0]:.6f}",
            )
        elif file_name.startswith(args.model_name) and (args.label == file_name.split('_')[2]):
            df = pd.read_csv(f'paper_results/{file_name}')
            print(
                file_name, fr"{df['avg_auc'].values[0]:.6f}\pm{df['std_auc'].values[0]:.6f}",
            )
            score_dict[
                file_name.split(
                    '_',
                )[1]
            ] = f"{df['avg_auc'].values[0]:.6f}\\pm{df['std_auc'].values[0]:.6f}"

        else:
            continue

    print(
        f"{latex_start_string}&${score_dict['book-page']}$&${score_dict['book']}$&${score_dict['subj']}$\\\\",  # noqa: E501
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
