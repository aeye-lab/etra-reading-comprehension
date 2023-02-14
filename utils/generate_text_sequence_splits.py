from __future__ import annotations

import json
import os
import random
from typing import Any

import feature_extraction
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm


def load_text_sequence_data(
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    datacols = [
        'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_DURATION',
        'CURRENT_FIX_PUPIL', 'CURRENT_FIX_INTEREST_AREA_ID',
    ]
    labelcols = [
        'subj', 'book', 'acc_level', 'subj_acc_level', 'confidence', 'difficulty', 'familiarity',
        'interest', 'pressured', 'sleepiness', 'sleephours', 'sex', 'native',
    ]
    FILE_LABELS = 'SB-SAT/fixation/18sat_labels.csv'
    sc = pd.read_csv(FILE_LABELS)
    sc['sex'] = sc['sex'].replace(['F', 'M'], [1, 0])
    binarycols = ('recognition', 'sex', 'native')
    subsetcols = [c for c in labelcols if c not in binarycols]
    sc[subsetcols] = sc[subsetcols].replace([0, 1, 2, 3], [0, 0, 1, 1])

    text_df = pd.read_csv('utils/texts_sb_sat.txt', delimiter='\t')
    fix_df = pd.read_csv('SB-SAT/fixation/18sat_fixfinal.csv')
    surprisal_df = pd.read_csv('utils/surprisal.csv')
    texts = text_df.title.unique().tolist()
    subjects = fix_df.RECORDING_SESSION_LABEL.unique().tolist()
    label_arr = np.empty((0, sc.shape[1]))
    data_arr = np.empty((0, 150, 53))
    fix_data = np.empty((0, 398, 5))
    label_dict = {label: idx for idx, label in enumerate(sc.columns.tolist())}
    for text_id in texts:
        print(f'Calculating for {text_id=}....')
        text_fix_df = fix_df.loc[fix_df.page_name == text_id]
        text_id_df = text_df.loc[text_df.title == text_id]
        sent_id_list = [i for i in text_id_df.sentence_nr.unique() for word in text_id_df.loc[text_id_df.sentence_nr == i].sentence.values[0].split()]  # noqa: E501
        sent_id_list.append(-4)
        sent_id_list.append(-3)
        sent_id_list.append(-2)
        sent_id_list.append(-1)
        text = ' '.join(sentence for sentence in text_id_df.sentence)
        text_list = text.split()
        text_list.append('nan')
        text_list.append('Previous Page')
        text_list.append('Next Page')
        text_list.append('Go To Question')
        assert len(sent_id_list) == len(text_list)
        sur_text_df = surprisal_df.loc[surprisal_df.title == text_id]
        surprisal_list = [sur_text_df.surprisal.iloc[idx] if idx < len(sur_text_df) else 0 for idx in range(len(text_list))]  # noqa: E501
        for subject in tqdm(subjects):
            tmp_label = sc.loc[
                sc.subj ==
                subject
            ].loc[sc.book == text_id.split('-')[1]]
            fixation_durations: list[list[int]] = [[]
                                                   for word in range(len(text_list))]
            fixation_id: list[list[float]] = [[]
                                              for word in range(len(text_list))]
            fix_aoi_id: list[list[float]] = [[]
                                             for word in range(len(text_list))]
            prev_fix_aoi_id: list[list[float]] = [[]
                                                  for word in range(len(text_list))]
            fixation_distance: list[list[float]] = [[]
                                                    for word in range(len(text_list))]
            fixation_location_x: list[list[float]] = [[]
                                                      for word in range(len(text_list))]
            fixation_location_y: list[list[float]] = [[]
                                                      for word in range(len(text_list))]
            text_sub_df = text_fix_df.loc[text_fix_df.RECORDING_SESSION_LABEL == subject].reset_index(drop=True)  # noqa: E501
            fix_data_tmp = text_sub_df[datacols].copy(deep=True)
            fix_data_tmp['CURRENT_FIX_INTEREST_AREA_ID'] = fix_data_tmp['CURRENT_FIX_INTEREST_AREA_ID'] - 4  # noqa: E501
            for cur_fix in text_sub_df.CURRENT_FIX_INTEREST_AREA_ID.unique():
                if np.isnan(cur_fix):
                    cur_list_element = 0
                    # duration
                    fixation_durations[cur_list_element - 4].extend(text_sub_df.loc[np.isnan(text_sub_df.CURRENT_FIX_INTEREST_AREA_ID)].CURRENT_FIX_DURATION.values.tolist())  # noqa: E501
                    # fixation sequence
                    fixation_location_x[cur_list_element - 4].extend(text_sub_df.loc[np.isnan(text_sub_df.CURRENT_FIX_INTEREST_AREA_ID)].CURRENT_FIX_X.values.tolist())  # noqa: E501
                    fixation_location_y[cur_list_element - 4].extend(text_sub_df.loc[np.isnan(text_sub_df.CURRENT_FIX_INTEREST_AREA_ID)].CURRENT_FIX_Y.values.tolist())  # noqa: E501
                    fixation_id[cur_list_element - 4].extend(text_sub_df.loc[np.isnan(text_sub_df.CURRENT_FIX_INTEREST_AREA_ID)].index.values.tolist())  # noqa: E501
                    fix_aoi_id[cur_list_element - 4].extend(text_sub_df.loc[np.isnan(text_sub_df.CURRENT_FIX_INTEREST_AREA_ID)].CURRENT_FIX_INTEREST_AREA_ID.values.tolist())  # noqa: E501
                    prev_fix_aoi_id[cur_list_element - 4].extend([text_sub_df.iloc[idx - 1].CURRENT_FIX_INTEREST_AREA_ID for idx in text_sub_df.loc[np.isnan(text_sub_df.CURRENT_FIX_INTEREST_AREA_ID)].index.values.tolist()])  # noqa: E501
                    fixation_distance[cur_list_element - 4] = [0 if (np.isnan(x) and np.isnan(y)) else 0 if (np.isnan(x) or np.isnan(y)) else x - y for x, y in zip(fix_aoi_id[cur_list_element - 4], prev_fix_aoi_id[cur_list_element - 4])]  # noqa: E501
                else:
                    cur_list_element = int(cur_fix)
                    # duration
                    fixation_durations[cur_list_element - 4].extend(text_sub_df.loc[text_sub_df.CURRENT_FIX_INTEREST_AREA_ID == cur_fix].CURRENT_FIX_DURATION.values.tolist())  # noqa: E501
                    # fixation sequence
                    fixation_location_x[cur_list_element - 4].extend(text_sub_df.loc[text_sub_df.CURRENT_FIX_INTEREST_AREA_ID == cur_fix].CURRENT_FIX_X.values.tolist())  # noqa: E501
                    fixation_location_y[cur_list_element - 4].extend(text_sub_df.loc[text_sub_df.CURRENT_FIX_INTEREST_AREA_ID == cur_fix].CURRENT_FIX_Y.values.tolist())  # noqa: E501
                    fixation_id[cur_list_element - 4].extend(text_sub_df.loc[text_sub_df.CURRENT_FIX_INTEREST_AREA_ID == cur_fix].index.values.tolist())  # noqa: E501
                    # to calculate regression path
                    fix_aoi_id[cur_list_element - 4].extend(text_sub_df.loc[text_sub_df.CURRENT_FIX_INTEREST_AREA_ID == cur_fix].CURRENT_FIX_INTEREST_AREA_ID.values.tolist())  # noqa: E501
                    prev_fix_aoi_id[cur_list_element - 4].extend([text_sub_df.iloc[idx - 1].CURRENT_FIX_INTEREST_AREA_ID for idx in text_sub_df.loc[text_sub_df.CURRENT_FIX_INTEREST_AREA_ID == cur_fix].index.values.tolist()])  # noqa: E501
                    fixation_distance[cur_list_element - 4] = [0 if (np.isnan(x) and np.isnan(y)) else 0 if (np.isnan(x) or np.isnan(y)) else x - y for x, y in zip(fix_aoi_id[cur_list_element - 4], prev_fix_aoi_id[cur_list_element - 4])]  # noqa: E501
            features, features_names = feature_extraction.get_linguistic_features_for_lists(
                fixation_list=fixation_durations,
                fixations_numbers=fixation_id,
                regression_values=fixation_distance,
                word_list=text_list,
                sentence_id_list=sent_id_list,
                suprisal_list=surprisal_list,
                pos_tagger='spacy',
                word_cluster_thresholds=[(1, 1), (1, 4), (5, 10), (11, 23)],
            )
            features = features.transpose()
            tmp_len = features.shape[0]
            fix_x_arr = np.array(
                np.vstack(
                    [
                        np.sum(cur_fix) if len(cur_fix) >
                        0 else 0 for cur_fix in fixation_location_x
                    ],
                ),
                ndmin=3,
            )
            fix_y_arr = np.array(
                np.vstack(
                    [
                        np.sum(cur_fix) if len(cur_fix) >
                        0 else 0 for cur_fix in fixation_location_y
                    ],
                ), ndmin=3,
            )
            features_all = np.dstack(
                [
                    np.array(features, ndmin=3),
                    fix_x_arr,
                    fix_y_arr,
                ],
            )

            data_arr = np.vstack(
                [
                    data_arr,
                    np.pad(
                        features_all,
                        pad_width=((0, 0), (0, 150 - tmp_len), (0, 0)),
                    ),
                ],
            )
            fix_data_tmp_len = len(fix_data_tmp)
            fix_data = np.vstack(
                [
                    fix_data,
                    np.pad(
                        np.array(fix_data_tmp, ndmin=3),
                        pad_width=(
                            (0, 0), (0, 398 - fix_data_tmp_len), (0, 0),
                        ),
                    ),
                ],
            )
            label_arr = np.vstack([label_arr, tmp_label])
    return label_arr, data_arr, label_dict, fix_data


def write_npys(
    label_arr: np.ndarray,
    data_arr_CNNs: list[np.ndarray],
    label_dict: dict[str, Any],
    split_criterion: str,
    suffixe: list[str],
    save_path: str = '',
) -> int:
    with open(f'{save_path}labels_dict.json', 'w') as fp:
        json.dump(label_dict, fp)

    split_criterion_dict = {
        'subj': 0,
        'book': 1,
    }
    if split_criterion != 'book-page':
        splitkeys = np.array(
            sorted(
                list(set(label_arr[:, split_criterion_dict[split_criterion]])),
            ),
        )
        if split_criterion == 'book':
            n_folds = 4
        else:
            n_folds = 5
    else:
        splitkeys = data_arr_CNNs[0]
        n_folds = 5

    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(splitkeys)):

        random.seed(fold)
        np.random.seed(fold)

        print(f'Writing fold {fold}...')

        for idx in range(len(data_arr_CNNs)):
            data_arr_CNN = data_arr_CNNs[idx]
            suffix = suffixe[idx]
            if split_criterion != 'book-page':
                N_train_sub = splitkeys[train_idx]
                N_test_sub = splitkeys[test_idx]
                print(f'training split: {N_train_sub}')
                print(f'test split: {N_test_sub}')

                X_train_CNN = data_arr_CNN[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_train_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/X_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_train_CNN)
                X_test_CNN = data_arr_CNN[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_test_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/X_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_test_CNN)
                y_train_CNN = label_arr[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_train_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/y_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_train_CNN)
                y_test_CNN = label_arr[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_test_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/y_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_test_CNN)
            else:
                X_train_CNN = data_arr_CNN[train_idx]
                with open(f'{save_path}/{split_criterion}/X_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_train_CNN)
                X_test_CNN = data_arr_CNN[test_idx]
                with open(f'{save_path}/{split_criterion}/X_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_test_CNN)
                y_train_CNN = label_arr[train_idx]
                with open(f'{save_path}/{split_criterion}/y_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_train_CNN)
                y_test_CNN = label_arr[test_idx]
                with open(f'{save_path}/{split_criterion}/y_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_test_CNN)
    return 0


def main() -> int:
    label_arr, data_arr_CNN, label_dict, fix_data = load_text_sequence_data()
    save_path = 'paper_splits/'
    os.makedirs(save_path, exist_ok=True)
    for split_criterion in ['subj', 'book', 'book-page']:
        os.makedirs(os.path.join(save_path, split_criterion), exist_ok=True)
        print(f'Creating files for split {split_criterion}...')
        write_npys(
            label_arr=label_arr,
            data_arr_CNNs=[data_arr_CNN, fix_data],
            label_dict=label_dict,
            split_criterion=split_criterion,
            suffixe=['', '_fix_data'],
            save_path=save_path,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
