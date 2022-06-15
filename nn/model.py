import argparse
import json
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import feature_extraction as feature_extraction
sys.path.append(os.getcwd())


def get_nn_model(
    red_pos_len, num_red_pos,
    con_len, num_con,
    num_features,
    fix_len=None, num_fix=None,
    flag_sequence_bilstm=False,
):

    drop_outs = 0.3
    concat_list = []
    input_list = []

    if red_pos_len is not None:
        x_red_pos_input = Input(shape=red_pos_len)
        x_red_pos = Embedding(
            input_dim=num_red_pos,
            output_dim=8,
        )(x_red_pos_input)
        if flag_sequence_bilstm:
            x_red_pos = Bidirectional(
                LSTM(25, return_sequences=True),
            )(x_red_pos)
            x_red_pos = Bidirectional(LSTM(25))(x_red_pos)
            x_red_pos = Dropout(0.3)(x_red_pos)
            x_red_pos = Dense(50, activation='relu')(x_red_pos)
            x_red_pos = Dropout(0.3)(x_red_pos)
            x_red_pos = Dense(20, activation='relu')(x_red_pos)
        else:
            x_red_pos_avg = GlobalAveragePooling1D()(x_red_pos)
            x_red_pos_max = GlobalMaxPool1D()(x_red_pos)
            x_red_pos = Concatenate()([x_red_pos_avg, x_red_pos_max])
            x_red_pos = Dense(128, activation='relu')(x_red_pos)
            x_red_pos = Dense(64, activation='relu')(x_red_pos)
            x_red_pos = Dense(32, activation='relu')(x_red_pos)
        concat_list.append(x_red_pos)
        input_list.append(x_red_pos_input)

    if con_len is not None:
        x_con_input = Input(shape=con_len)
        x_con = Embedding(
            input_dim=num_con,
            output_dim=8,
        )(x_con_input)
        if flag_sequence_bilstm:
            x_con = Bidirectional(LSTM(25, return_sequences=True))(x_con)
            x_con = Bidirectional(LSTM(25))(x_con)
            x_con = Dropout(0.3)(x_con)
            x_con = Dense(50, activation='relu')(x_con)
            x_con = Dropout(0.3)(x_con)
            x_con = Dense(20, activation='relu')(x_con)
        else:
            x_con_avg = GlobalAveragePooling1D()(x_con)
            x_con_max = GlobalMaxPool1D()(x_con)
            x_con = Concatenate()([x_con_avg, x_con_max])
            x_con = Dense(128, activation='relu')(x_con)
            x_con = Dense(64, activation='relu')(x_con)
            x_con = Dense(32, activation='relu')(x_con)
        concat_list.append(x_con)
        input_list.append(x_con_input)

    if num_features is not None:
        x_num_input = Input(shape=num_features)
        x_num = Dropout(drop_outs)(x_num_input)
        x_num = Dense(32, activation='relu')(x_num)
        concat_list.append(x_num)
        input_list.append(x_num_input)

    if fix_len is not None:
        x_fix_input = Input(shape=(fix_len, num_fix))
        x_fix = Bidirectional(LSTM(25, return_sequences=True))(x_fix_input)
        x_fix = Bidirectional(LSTM(25))(x_fix)
        x_fix = Dropout(0.3)(x_fix)
        x_fix = Dense(50, activation='relu')(x_fix)
        x_fix = Dropout(0.3)(x_fix)
        x_fix = Dense(20, activation='relu')(x_fix)
        concat_list.append(x_fix)
        input_list.append(x_fix_input)

    x_concat = Concatenate()(concat_list)
    x_output = Dense(32, activation='relu')(x_concat)
    x_output = Dense(1, activation='sigmoid')(x_output)
    model = Model(
        input_list,
        outputs=x_output,
    )
    opt = Adam()
    model.compile(
        optimizer=opt, loss='binary_crossentropy',
        metrics=['accuracy', auroc],
    )
    # model.summary()
    return model


def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)

# calculate the roc-auc as a metric


def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)


def train_nn(
    spit_criterions, labels,
    feature_names_per_word,
    model_name,
    flag_sequence_bilstm=True,
    word_in_fixation_order=True,
    use_reduced_pos_sequence=True,
    use_content_word_sequence=True,
    use_numeric=True,
    use_fixation_sequence=True,
    flag_redo=False,
    normalize_flag=True,
    use_gaze_entropy_features=True,
    patience=50,
    batch_size=256,
    epochs=1000,
    save_dir='/home/prasse/work/Projekte/AEye/reading-comprehension/nn/results/',
    save_csv=True,
    save_joblib=False,
):
    for split_criterion in spit_criterions:
        for label in labels:
            model_prefix = str(flag_sequence_bilstm) +\
                '_' + str(word_in_fixation_order) +\
                '_' + str(use_reduced_pos_sequence) +\
                '_' + str(use_content_word_sequence) +\
                '_' + str(use_numeric) +\
                '_' + str(use_fixation_sequence) +\
                '_'
            csv_save_path = f'{save_dir}{model_prefix}{model_name}_{split_criterion}_text_sequence_{label}.csv'  # noqa: E501
            joblib_save_path = csv_save_path.replace('.csv', '.joblib')
            if not flag_redo and save_csv and os.path.exists(csv_save_path):
                continue
            if not flag_redo and save_joblib and os.path.exists(joblib_save_path):
                continue
            SB_SAT_PATH = f'paper_splits/{split_criterion}/'
            split_criterion_dict = {
                'subj': 0,
                'book': 1,
                'subj-book': 0,
            }
            with open('paper_splits/labels_dict.json') as fp:
                label_dict = json.load(fp)

            if (split_criterion == 'book') or (split_criterion == 'subj-book'):
                num_folds = 4
            else:
                num_folds = 5
            pd_init = pd.DataFrame(
                columns=[
                    'ahn_baseline',
                    'fold0_auc', 'fold1_auc', 'fold2_auc', 'fold3_auc', 'fold4_auc',
                    'fold0_tpr', 'fold1_tpr', 'fold2_tpr', 'fold3_tpr', 'fold4_tpr',
                    'fold0_fpr', 'fold1_fpr', 'fold2_fpr', 'fold3_fpr', 'fold4_fpr',
                    'fold0_y_pred', 'fold1_y_pred', 'fold2_y_pred', 'fold3_y_pred', 'fold4_y_pred',
                    'fold0_y_test', 'fold1_y_test', 'fold2_y_test', 'fold3_y_test', 'fold4_y_test',
                    'avg_auc', 'std_auc',
                ],
            )
            out_dict = dict()

            pd_init['ahn_baseline'] = [model_name]

            for fold in range(num_folds):
                np.random.seed(fold)
                random.seed(fold)
                # collect the inputs for train, validation and test
                # use only features where flag is True
                train_inputs = []
                val_inputs = []
                test_inputs = []
                X_train_path = os.path.join(
                    SB_SAT_PATH, f'X_train_{split_criterion}_{fold}.npy',
                )
                X_train_fix_path = os.path.join(
                    SB_SAT_PATH, f'X_train_{split_criterion}_{fold}_fix_data.npy',
                )
                y_train_path = os.path.join(
                    SB_SAT_PATH, f'y_train_{split_criterion}_{fold}.npy',
                )
                x_train_all, y_train_all = np.load(X_train_path), np.load(
                    y_train_path, allow_pickle=True,
                )
                x_train_fix_all = np.load(X_train_fix_path)
                x_train_fix_postions = x_train_fix_all[:, :, 4]
                if normalize_flag:
                    scaler = MinMaxScaler()
                    fix_scaler = MinMaxScaler()
                    x_train_all = scaler.fit_transform(
                        x_train_all.reshape(-1, x_train_all.shape[-1]),
                    ).reshape(x_train_all.shape)
                    x_train_fix_all = fix_scaler.fit_transform(
                        x_train_fix_all.reshape(-1, x_train_fix_all.shape[-1]),
                    ).reshape(x_train_fix_all.shape)
                    x_train_fix_all = np.where(
                        np.isnan(x_train_fix_all), -4, x_train_fix_all,
                    )
                if split_criterion != 'book':
                    outer_cv = KFold(
                        n_splits=4, shuffle=True,
                        random_state=fold,
                    )
                else:
                    outer_cv = KFold(
                        n_splits=3, shuffle=True,
                        random_state=fold,
                    )

                if split_criterion != 'book-page':
                    splitkeys = np.array(
                        sorted(
                            list(
                                set(
                                    y_train_all[
                                        :,
                                        split_criterion_dict[split_criterion]
                                    ],
                                ),
                            ),
                        ),
                    )
                else:
                    splitkeys = y_train_all[:, label_dict[label]]

                for train_idx, val_idx in outer_cv.split(splitkeys):
                    break

                if split_criterion != 'book-page':
                    N_train_sub = splitkeys[train_idx]
                    N_test_sub = splitkeys[val_idx]

                    train_idx = np.where(
                        np.isin(
                            y_train_all[
                                :, split_criterion_dict[split_criterion]
                            ], N_train_sub,
                        ),
                    )[0]
                    val_idx = np.where(
                        np.isin(
                            y_train_all[
                                :, split_criterion_dict[split_criterion]
                            ], N_test_sub,
                        ),
                    )[0]
                x_train = x_train_all[train_idx]
                y_train = y_train_all[train_idx]
                x_val = x_train_all[val_idx]
                y_val = y_train_all[val_idx]
                y_train_all[val_idx]

                y_train = np.array(y_train[:, label_dict[label]], dtype=int)
                y_val = np.array(y_val[:, label_dict[label]], dtype=int)

                x_train, feature_names = feature_extraction.get_features_for_word_features(
                    x_train, feature_names_per_word, disable=False,
                    use_gaze_entropy_features=use_gaze_entropy_features,
                )
                x_val, _ = feature_extraction.get_features_for_word_features(
                    x_val, feature_names_per_word, disable=False,
                    use_gaze_entropy_features=use_gaze_entropy_features,
                )

                # Test Data
                X_test_path = os.path.join(
                    SB_SAT_PATH,
                    f'X_test_{split_criterion}_{fold}.npy',
                )
                X_test_fix_path = os.path.join(
                    SB_SAT_PATH,
                    f'X_test_{split_criterion}_{fold}_fix_data.npy',
                )
                y_test_path = os.path.join(
                    SB_SAT_PATH,
                    f'y_test_{split_criterion}_{fold}.npy',
                )
                x_test_all, y_test_all = np.load(X_test_path), np.load(
                    y_test_path, allow_pickle=True,
                )
                x_test_fix_all = np.load(X_test_fix_path)
                x_test_fix_postions = x_test_fix_all[:, :, 4]
                if normalize_flag:
                    x_test_all = scaler.transform(
                        x_test_all.reshape(-1, x_test_all.shape[-1]),
                    ).reshape(x_test_all.shape)
                    x_test_fix_all = fix_scaler.transform(
                        x_test_fix_all.reshape(-1, x_test_fix_all.shape[-1]),
                    ).reshape(x_test_fix_all.shape)
                    x_test_fix_all = np.where(
                        np.isnan(x_test_fix_all), -4, x_test_fix_all,
                    )
                y_test = np.array(y_test_all[:, label_dict[label]], dtype=int)

                x_test, _ = feature_extraction.get_features_for_word_features(
                    x_test_all, feature_names_per_word, disable=False,
                    use_gaze_entropy_features=use_gaze_entropy_features,
                )

                # reduced POS + content word lists
                pos_list_list_train, entity_list_list_train, reduced_pos_list_list_train, content_list_list_train, pos_feature_names, entity_feature_names, reduced_pos_feature_names, content_word_feature_names, fix_position_ind_train = feature_extraction.get_watched_pos_entity_lists(  # noqa: E501
                    x_train_all, feature_names_per_word, x_train_fix_postions,
                )
                pos_list_list_test, entity_list_list_test, reduced_pos_list_list_test, content_list_list_test, _, _, _, _, fix_position_ind_test = feature_extraction.get_watched_pos_entity_lists(  # noqa: E501
                    x_test_all, feature_names_per_word, x_test_fix_postions,
                )

                pos_lens = [len(pos_list) for pos_list in pos_list_list_train]
                fix_lens = [
                    len(pos_fix_list)
                    for pos_fix_list in fix_position_ind_train
                ]
                max_pos_len = np.max(pos_lens)
                max_fix_len = np.max(fix_lens)

                # Reduced POS
                train_reduced_pos_matrix, reduced_pos_mapping_dict = feature_extraction.convert_to_sequence_to_nn_input(  # noqa: E501
                    reduced_pos_list_list_train, reduced_pos_feature_names, max_pos_len,
                )
                test_reduced_pos_matrix, _ = feature_extraction.convert_to_sequence_to_nn_input(
                    reduced_pos_list_list_test, reduced_pos_feature_names, max_pos_len,
                    reduced_pos_mapping_dict,
                )
                train_reduced_pos_fix_matrix = feature_extraction.convert_to_sequence_to_nn_input_fixation(  # noqa: E501
                    reduced_pos_list_list_train, reduced_pos_feature_names, max_fix_len,
                    reduced_pos_mapping_dict, fix_position_ind_train,
                )
                test_reduced_pos_fix_matrix = feature_extraction.convert_to_sequence_to_nn_input_fixation(  # noqa: E501
                    reduced_pos_list_list_test, reduced_pos_feature_names, max_fix_len,
                    reduced_pos_mapping_dict, fix_position_ind_test,
                )

                # Content Words
                train_content_matrix, content_word_mapping_dict = feature_extraction.convert_to_sequence_to_nn_input(  # noqa: E501
                    content_list_list_train, content_word_feature_names, max_pos_len,
                )
                test_content_matrix, _ = feature_extraction.convert_to_sequence_to_nn_input(
                    content_list_list_test, content_word_feature_names, max_pos_len,
                    content_word_mapping_dict,
                )
                train_content_fix_matrix = feature_extraction.convert_to_sequence_to_nn_input_fixation(  # noqa: E501
                    content_list_list_train, content_word_feature_names, max_fix_len,
                    content_word_mapping_dict, fix_position_ind_train,
                )
                test_content_fix_matrix = feature_extraction.convert_to_sequence_to_nn_input_fixation(  # noqa: E501
                    content_list_list_test, content_word_feature_names, max_fix_len,
                    content_word_mapping_dict, fix_position_ind_test,
                )

                val_reduced_pos_fix_matrix = train_reduced_pos_fix_matrix[val_idx]
                train_reduced_pos_fix_matrix = train_reduced_pos_fix_matrix[train_idx]
                test_reduced_pos_fix_matrix = test_reduced_pos_fix_matrix

                val_content_fix_matrix = train_content_fix_matrix[val_idx]
                train_content_fix_matrix = train_content_fix_matrix[train_idx]
                test_content_fix_matrix = test_content_fix_matrix

                val_reduced_pos_matrix = train_reduced_pos_matrix[val_idx]
                train_reduced_pos_matrix = train_reduced_pos_matrix[train_idx]
                test_reduced_pos_matrix = test_reduced_pos_matrix

                val_content_matrix = train_content_matrix[val_idx]
                train_content_matrix = train_content_matrix[train_idx]
                test_content_matrix = test_content_matrix

                train_fix_matrix = x_train_fix_all[:, :, :-1]
                val_fix_matrix = train_fix_matrix[val_idx]
                train_fix_matrix = train_fix_matrix[train_idx]
                test_fix_matrix = x_test_fix_all[:, :, :-1]

                reduced_pos_len = train_reduced_pos_matrix.shape[1]
                content_len = train_content_matrix.shape[1]
                num_features = x_train.shape[1]
                num_fix = train_fix_matrix.shape[2]
                fix_len = train_fix_matrix.shape[1]

                # scale the input
                input_scaler = MinMaxScaler()
                x_train = input_scaler.fit_transform(x_train)
                x_val = input_scaler.transform(x_val)
                x_test = input_scaler.transform(x_test)

                if use_reduced_pos_sequence:
                    if word_in_fixation_order:
                        train_inputs.append(train_reduced_pos_fix_matrix)
                        val_inputs.append(val_reduced_pos_fix_matrix)
                        test_inputs.append(test_reduced_pos_fix_matrix)
                        reduced_pos_len = train_reduced_pos_fix_matrix.shape[1]
                        num_pos_reduced = len(reduced_pos_mapping_dict) + 1
                    else:
                        train_inputs.append(train_reduced_pos_matrix)
                        val_inputs.append(val_reduced_pos_matrix)
                        test_inputs.append(test_reduced_pos_matrix)
                        reduced_pos_len = train_reduced_pos_matrix.shape[1]
                        num_pos_reduced = len(reduced_pos_mapping_dict) + 1
                else:
                    reduced_pos_len = None
                    num_pos_reduced = None

                if use_content_word_sequence:
                    if word_in_fixation_order:
                        train_inputs.append(train_content_fix_matrix)
                        val_inputs.append(val_content_fix_matrix)
                        test_inputs.append(test_content_fix_matrix)
                        content_len = train_content_fix_matrix.shape[1]
                        num_content = len(content_word_mapping_dict) + 1
                    else:
                        train_inputs.append(train_content_matrix)
                        val_inputs.append(val_content_matrix)
                        test_inputs.append(test_content_matrix)
                        content_len = train_content_matrix.shape[1]
                        num_content = len(content_word_mapping_dict) + 1
                else:
                    content_len = None
                    num_content = None

                if use_numeric:
                    train_inputs.append(x_train)
                    val_inputs.append(x_val)
                    test_inputs.append(x_test)
                    num_features = x_train.shape[1]
                else:
                    num_features = None

                if use_fixation_sequence:
                    train_inputs.append(train_fix_matrix)
                    val_inputs.append(val_fix_matrix)
                    test_inputs.append(test_fix_matrix)
                    fix_len = train_fix_matrix.shape[1]
                    num_fix = train_fix_matrix.shape[2]
                else:
                    fix_len = None
                    num_fix = None

                model = get_nn_model(
                    reduced_pos_len, num_pos_reduced, content_len, num_content,
                    num_features, num_fix=num_fix, fix_len=fix_len,
                )

                tf.keras.backend.clear_session()
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss', patience=patience,
                    ),
                ]
                history = model.fit(  # noqa: F841
                    train_inputs, y_train,
                    validation_data=(
                        val_inputs,
                        y_val,
                    ),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0,
                )

                y_pred = model.predict(
                    test_inputs,
                    batch_size=batch_size,
                )
                try:
                    fpr, tpr, _ = metrics.roc_curve(
                        y_test,
                        y_pred,
                        pos_label=1,
                    )
                    auc = metrics.auc(fpr, tpr)
                    print(auc)
                    pd_init[f'fold{fold}_auc'] = auc
                    pd_init[f'fold{fold}_tpr'] = [tpr]
                    pd_init[f'fold{fold}_fpr'] = [fpr]
                    pd_init[f'fold{fold}_y_test'] = [y_test]
                    pd_init[f'fold{fold}_y_pred'] = [y_pred]

                    out_dict[f'fold{fold}_auc'] = auc
                    out_dict[f'fold{fold}_tpr'] = [tpr]
                    out_dict[f'fold{fold}_fpr'] = [fpr]
                    out_dict[f'fold{fold}_y_test'] = [y_test]
                    out_dict[f'fold{fold}_y_pred'] = [y_pred]
                except KeyError:
                    try:
                        fpr, tpr, _ = metrics.roc_curve(
                            y_test,
                            y_pred,
                            pos_label=1,
                        )
                        auc = metrics.auc(fpr, tpr)
                        print(auc)
                        pd_init[f'fold{fold}_auc'] = auc
                        pd_init[f'fold{fold}_tpr'] = [tpr]
                        pd_init[f'fold{fold}_fpr'] = [fpr]
                        pd_init[f'fold{fold}_y_test'] = y_test
                        pd_init[f'fold{fold}_y_pred'] = y_pred

                        out_dict[f'fold{fold}_auc'] = auc
                        out_dict[f'fold{fold}_tpr'] = [tpr]
                        out_dict[f'fold{fold}_fpr'] = [fpr]
                        out_dict[f'fold{fold}_y_test'] = y_test
                        out_dict[f'fold{fold}_y_pred'] = y_pred
                    except KeyError as e:
                        raise e

            pd_init['avg_auc'] = 0
            out_dict['avg_auc'] = 0
            for i in range(num_folds):
                pd_init['avg_auc'] += pd_init[f'fold{i}_auc']
                out_dict['avg_auc'] += out_dict[f'fold{i}_auc']
            pd_init['avg_auc'] /= num_folds
            out_dict['avg_auc'] /= num_folds

            pd_init['std_auc'] = 0
            out_dict['std_auc'] = 0
            for i in range(0, num_folds):
                pd_init['std_auc'] += (
                    pd_init[f'fold{i}_auc'] -
                    pd_init['avg_auc']
                )**2
                out_dict['std_auc'] += (
                    out_dict[f'fold{i}_auc'] - out_dict['avg_auc']
                )**2
            pd_init['std_auc'] = (pd_init['std_auc']/num_folds)**(1/2)
            out_dict['std_auc'] = (out_dict['std_auc']/num_folds)**(1/2)
            if save_csv:
                pd_init.to_csv(csv_save_path, index=None)
            if save_joblib:
                joblib.dump(out_dict, joblib_save_path, compress=3, protocol=2)
            print('mean auc: ' + str(pd_init['avg_auc']))


def convert_string_to_boolean(input_string):
    if input_string == 'True':
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU', '--GPU', type=int, default=4)
    parser.add_argument(
        '-flag_sequence_bilstm',
        '--flag_sequence_bilstm', type=str, default='True',
    )
    parser.add_argument(
        '-word_in_fixation_order',
        '--word_in_fixation_order', type=str, default='True',
    )
    parser.add_argument(
        '-use_reduced_pos_sequence',
        '--use_reduced_pos_sequence', type=str, default='True',
    )
    parser.add_argument(
        '-use_content_word_sequence',
        '--use_content_word_sequence', type=str, default='True',
    )
    parser.add_argument(
        '-use_numeric', '--use_numeric',
        type=str, default='True',
    )
    parser.add_argument(
        '-use_fixation_sequence',
        '--use_fixation_sequence', type=str, default='True',
    )
    parser.add_argument('-save_dir', '--save_dir', type=str, default='True')

    args = parser.parse_args()
    GPU = args.GPU
    flag_sequence_bilstm = convert_string_to_boolean(args.flag_sequence_bilstm)
    word_in_fixation_order = convert_string_to_boolean(
        args.word_in_fixation_order,
    )
    use_reduced_pos_sequence = convert_string_to_boolean(
        args.use_reduced_pos_sequence,
    )
    use_content_word_sequence = convert_string_to_boolean(
        args.use_content_word_sequence,
    )
    use_numeric = convert_string_to_boolean(args.use_numeric)
    use_fixation_sequence = convert_string_to_boolean(
        args.use_fixation_sequence,
    )
    save_dir = args.save_dir

    # select graphic card
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)  # noqa: F841

    normalize_flag = True
    use_gaze_entropy_features = True

    flag_redo = True
    patience = 50
    batch_size = 256
    epochs = 1000

    spit_criterions = ['book-page', 'subj', 'book']
    labels = ['subj_acc_level', 'acc_level', 'native', 'difficulty']
    model_name = 'nn_paul'

    feature_names_per_word = [
        'ff',
        'tf',
        'wfc_ff_nomarlized',
        'wfc_tf_normalized',
        'sc_ff_normalized',
        'sc_tf_normalized',
        'ic_ff_normalized',
        'ic_tf_normalized',
        'regression',
        'num_regressions',
        'num_progressions',
        'surprisal',
        'word_len',
        'dependencies_right',
        'dependencies_left',
        'dependency_distance',
    ]

    unique_content_word_list = list(
        np.unique(list(feature_extraction.content_word_dict.values())),
    )
    for content_word in unique_content_word_list:
        feature_names_per_word.append('is_content_word_' + content_word)

    unique_reduced_pos_list = list(
        np.unique(list(feature_extraction.reduced_pos_dict.values())),
    )
    for r_pos in unique_reduced_pos_list:
        feature_names_per_word.append('is_reduced_pos_' + r_pos)

    feature_names_per_word += ['x_mean', 'y_mean']

    # train models
    tf.keras.backend.clear_session()
    train_nn(
        spit_criterions=spit_criterions,
        labels=labels,
        feature_names_per_word=feature_names_per_word,
        model_name=model_name,
        flag_sequence_bilstm=flag_sequence_bilstm,
        word_in_fixation_order=word_in_fixation_order,
        use_reduced_pos_sequence=use_reduced_pos_sequence,
        use_content_word_sequence=use_content_word_sequence,
        use_numeric=use_numeric,
        use_fixation_sequence=use_fixation_sequence,
        flag_redo=flag_redo,
        normalize_flag=normalize_flag,
        use_gaze_entropy_features=use_gaze_entropy_features,
        patience=patience,
        batch_size=batch_size,
        epochs=epochs,
        save_dir=save_dir,
        save_csv=True,
        save_joblib=True,
    )


if __name__ == '__main__':
    raise SystemExit(main())
