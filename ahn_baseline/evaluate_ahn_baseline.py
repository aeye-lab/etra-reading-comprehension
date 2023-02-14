from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf_session = tf.compat.v1.Session(config=config)


def get_model(
    model_name: str,
    input_shape: tuple[int, int] = (398, 4),
) -> Model:
    if model_name == 'rnn':
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(25, return_sequences=True), input_shape=input_shape,
            ),
        )
        model.add(Bidirectional(LSTM(25)))
        model.add(Dropout(0.3))

        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(20))
        model.add(Activation('relu'))

        model.add(Dense(1, activation='sigmoid'))
    elif model_name == 'cnn1':
        model = Sequential()
        model.add(Conv1D(40, 3, input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv1D(40, 3))
        model.add(Activation('relu'))

        model.add(Conv1D(40, 3))
        model.add(Activation('relu'))

        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))

        model.add(Flatten())

        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(20))
        model.add(Activation('relu'))

        model.add(Dense(1, activation='sigmoid'))

    elif model_name == 'cnn2':
        model = Sequential()

        model.add(Conv1D(64, 3, padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv1D(64, 3, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv1D(64, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv1D(64, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(20))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(1, activation='sigmoid'))
    elif model_name == 'dense':
        inputs = Input(shape=input_shape)
        x = Flatten()(inputs)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=predictions)
    return model


def evaluate_model(
    model_name,
    split_criterion,
    label,
    save_path,
    label_dict,
):
    input_shape = (398, 4)
    SB_SAT_PATH = f'paper_splits/{split_criterion}/'
    split_criterion_dict = {
        'subj': 0,
        'book': 1,
    }

    if split_criterion == 'book':
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
    pd_init['ahn_baseline'] = [model_name]

    for fold in range(num_folds):
        np.random.seed(fold)
        random.seed(fold)
        tf.random.set_seed(fold)
        X_train_path = os.path.join(
            SB_SAT_PATH, f'X_train_{split_criterion}_{fold}_fix_data.npy',
        )
        y_train_path = os.path.join(
            SB_SAT_PATH, f'y_train_{split_criterion}_{fold}.npy',
        )
        x_train_all, y_train_all = np.load(X_train_path), np.load(
            y_train_path, allow_pickle=True,
        )
        x_train_all = x_train_all[:, :, [0, 1, 2, 3]]
        scaler = MinMaxScaler()
        x_train_all = scaler.fit_transform(
            x_train_all.reshape(-1, x_train_all.shape[-1]),
        ).reshape(x_train_all.shape)
        if split_criterion != 'book':
            outer_cv = KFold(n_splits=4, shuffle=True, random_state=fold)
        else:
            outer_cv = KFold(n_splits=3, shuffle=True, random_state=fold)
        if split_criterion != 'book-page':
            splitkeys = np.array(
                sorted(
                    list(
                        set(
                            y_train_all[
                                :,
                                split_criterion_dict[split_criterion],
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

            x_train = x_train_all[
                np.isin(
                    y_train_all[
                        :, split_criterion_dict[split_criterion],
                    ], N_train_sub,
                )
            ]
            x_val = x_train_all[
                np.isin(
                    y_train_all[
                        :, split_criterion_dict[split_criterion],
                    ], N_test_sub,
                )
            ]
            y_train = y_train_all[
                np.isin(
                    y_train_all[
                        :, split_criterion_dict[split_criterion],
                    ], N_train_sub,
                )
            ]
            y_val = y_train_all[
                np.isin(
                    y_train_all[
                        :, split_criterion_dict[split_criterion],
                    ], N_test_sub,
                )
            ]
        else:
            x_train = x_train_all[train_idx]
            y_train = y_train_all[train_idx]
            x_val = x_train_all[val_idx]
            y_val = y_train_all[val_idx]
        y_train = np.array(y_train[:, label_dict[label]], dtype=int)
        y_val = np.array(y_val[:, label_dict[label]], dtype=int)

        callbacks_list = [
            EarlyStopping(
                monitor='val_loss', patience=50,
                mode='min',
            ),
        ]
        clear_session()
        model = get_model(
            model_name=model_name,
            input_shape=input_shape,
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', AUC()],
        )
        if fold == 0:
            model.summary()
        model.fit(
            x_train,
            y_train,
            epochs=1000,
            batch_size=256,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            verbose=2,
        )

        X_test_path = os.path.join(
            SB_SAT_PATH, f'X_test_{split_criterion}_{fold}_fix_data.npy',
        )
        y_test_path = os.path.join(
            SB_SAT_PATH, f'y_test_{split_criterion}_{fold}.npy',
        )
        x_test, y_test = np.load(X_test_path), np.load(
            y_test_path, allow_pickle=True,
        )
        x_test = x_test[:, :, [0, 1, 2, 3]]
        x_test = scaler.transform(
            x_test.reshape(-1, x_test.shape[-1]),
        ).reshape(x_test.shape)
        y_test = np.array(y_test[:, label_dict[label]], dtype=int)

        y_pred = model.predict(x_test, batch_size=256)

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
            except KeyError:
                raise AssertionError
        clear_session()

    # calc avg
    pd_init['avg_auc'] = 0
    for i in range(num_folds):
        pd_init['avg_auc'] += pd_init[f'fold{i}_auc']
    pd_init['avg_auc'] /= num_folds

    # calculate std
    pd_init['std_auc'] = 0
    for i in range(0, num_folds):
        pd_init['std_auc'] += (pd_init[f'fold{i}_auc'] - pd_init['avg_auc'])**2
    pd_init['std_auc'] = (pd_init['std_auc'] / num_folds)**(1 / 2)

    pd_init.to_csv(save_path, index=None)


def main() -> int:

    with open('paper_splits/labels_dict.json') as fp:
        label_dict = json.load(fp)
    model_names = ['rnn', 'dense', 'cnn1']
    split_criterions = ['subj', 'book-page', 'subj', 'book']
    predictions = ['subj_acc_level', 'acc_level', 'native', 'difficulty']
    os.makedirs('paper_results')
    for model_name in model_names:
        for split_criterion in split_criterions:
            for predict in predictions:
                evaluate_model(
                    model_name=model_name,
                    split_criterion=split_criterion,
                    label=predict,
                    save_path=f'paper_results/ahn_{model_name}_{split_criterion}_{predict}.csv',
                    label_dict=label_dict,
                )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
