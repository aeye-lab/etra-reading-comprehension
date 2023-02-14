from __future__ import annotations

import re
from typing import Any

import numpy as np
import spacy.attrs
from scipy.stats import kurtosis
from scipy.stats import skew
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm')


entity_list = [
    '', 'DATE', 'QUANTITY', 'ORG', 'GPE',
    'NORP', 'PERSON', 'TIME', 'CARDINAL', 'ORDINAL', 'NaN',
]
pos_list = [
    'PUNCT', 'PROPN', 'NOUN', 'PRON', 'VERB', 'SCONJ', 'NUM',
    'DET', 'CCONJ', 'ADP', 'AUX', 'ADV', 'ADJ', 'INTJ', 'X', 'PART',
]

content_word_dict = {
    'PUNCT': 'NO_CONTENT',
    'PROPN': 'CONTENT',
    'NOUN': 'CONTENT',
    'PRON': 'NO_CONTENT',
    'VERB': 'CONTENT',
    'SCONJ': 'NO_CONTENT',
    'NUM': 'NO_CONTENT',
    'DET': 'NO_CONTENT',
    'CCONJ': 'NO_CONTENT',
    'ADP': 'NO_CONTENT',
    'AUX': 'NO_CONTENT',
    'ADV': 'CONTENT',
    'ADJ': 'CONTENT',
    'INTJ': 'NO_CONTENT',
    'X': 'NO_CONTENT',
    'PART': 'NO_CONTENT',
    'NaN': 'UNKNOWN',
}

reduced_pos_dict = {
    'PUNCT': 'FUNC',
    'PROPN': 'NOUN',
    'NOUN': 'NOUN',
    'PRON': 'FUNC',
    'VERB': 'VERB',
    'SCONJ': 'FUNC',
    'NUM': 'FUNC',
    'DET': 'FUNC',
    'CCONJ': 'FUNC',
    'ADP': 'FUNC',
    'AUX': 'FUNC',
    'ADV': 'ADJ',
    'ADJ': 'ADJ', 'INTJ': 'FUNC',
    'X': 'FUNC',
    'PART': 'FUNC',
    'NaN': 'UNKNOWN',
}


# create counts for entities
# params:
#       input: text as string
# returns:
#       deps: name for dependency
#       n_rights: number of dependencies to the right
#       rights: dependencies to the right
#       n_lefts: number of dependencies to the left
#       lefts: dependencies to the left
def parse_dependency(text) -> tuple[
    list[list[str]], list[list[int]], list[list[str]],
    list[list[int]], list[list[str]], list[list[int]],
]:
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = spacy.tokenizer.Tokenizer(
        nlp.vocab, token_match=re.compile(r'\S+').match,
    )
    doc = nlp(text)
    n_rights: list[list[int]] = [[] for _ in range(len(text.split()))]
    n_lefts: list[list[int]] = [[] for _ in range(len(text.split()))]
    rights: list[list[str]] = [[] for _ in range(len(text.split()))]
    lefts: list[list[str]] = [[] for _ in range(len(text.split()))]
    deps: list[list[str]] = [[] for _ in range(len(text.split()))]
    dep_distance: list[list[int]] = [[] for _ in range(len(text.split()))]

    for idx, token in enumerate(doc):
        deps[idx] = token.dep_
        rights[idx] = list(token.rights)
        lefts[idx] = list(token.lefts)
        n_rights[idx] = token.n_rights
        n_lefts[idx] = token.n_lefts
        dep_distance[idx] = token.i - token.head.i

    return deps, n_rights, rights, n_lefts, lefts, dep_distance


# create counts for entities
# params:
#       input_word_list: list of words
# returns:
#       features
#       feature_names
def get_entity_count_features(input_word_list):
    input_text = ' '.join(input_word_list)
    doc = nlp(input_text)
    counts_dict_entities = doc.count_by(spacy.attrs.IDS['ENT_TYPE'])
    human_readable_dict = dict()
    for entity, count in counts_dict_entities.items():
        human_readable_tag = doc.vocab[entity].text
        human_readable_dict[human_readable_tag] = count

    features = []
    feature_names = []
    for entity in entity_list:
        if entity in human_readable_dict:
            feat = human_readable_dict[entity]
        else:
            feat = 0
        features.append(feat)
        feature_names.append('lexical_entity_count_' + entity)
        features.append(feat / len(input_word_list))
        feature_names.append('lexical_entity_frac_' + entity)
    return np.array(features), feature_names

# create feature matrix for NN input (for Input to Embedding Layer)
# params:
#       input_list_list: list of list of input strings
#       input_feature_names: feature_names
#       max_input_len: maximum length of sequence
#       mapping_dict: mapping from input_strings to ids (if None -> will be created)
#
# returns:
#       out_matrix: n_instances x max_len feature matrix
#       out_mapping_dict: mapping from input strings to ids


def convert_to_sequence_to_nn_input(
    input_list_list, input_feature_names, max_input_len, mapping_dict=None,
):
    out_matrix = np.zeros([len(input_list_list), max_input_len])
    if mapping_dict is not None:
        out_mapping_dict = mapping_dict
    else:
        out_mapping_dict = dict()
    for i in range(len(input_list_list)):
        cur_list = input_list_list[i]
        for j in range(len(cur_list)):
            if j >= max_input_len:
                break
            cur_ins = cur_list[j]
            if cur_ins not in out_mapping_dict and mapping_dict is None:
                out_mapping_dict[cur_ins] = len(out_mapping_dict) + 1
            try:
                out_matrix[i, j] = out_mapping_dict[cur_ins]
            except KeyError:
                pass
    return out_matrix, out_mapping_dict


# create feature matrix for NN input (for Input to Embedding Layer)
# params:
#       input_list_list: list of list of input strings
#       input_feature_names: feature_names
#       max_input_len: maximum length of sequence
#       mapping_dict: mapping from input_strings to ids
#       fix_position_ind: list of list of pointers to words in input_list_list
#
# returns:
#       out_matrix: n_instances x max_len feature matrix
def convert_to_sequence_to_nn_input_fixation(
    input_list_list, input_feature_names, max_input_len,
        mapping_dict, fix_position_ind,
):
    out_matrix = np.zeros([len(fix_position_ind), max_input_len])
    out_mapping_dict = mapping_dict
    for i in range(len(fix_position_ind)):
        cur_list = fix_position_ind[i]
        cur_words = input_list_list[i]
        for j in range(len(cur_list)):
            if j >= max_input_len:
                break
            cur_ins = cur_list[j]
            cur_word = cur_words[cur_ins]
            try:
                out_matrix[i, j] = out_mapping_dict[cur_word]
            except KeyError:
                pass
    return out_matrix


# create list of watched POS, reduced POS, content words and entities
# params:
#       data: input feature matrix: n_instances x n_words x features
#       input_feature_names: feature_names
#       fix_postions: index of words looked at (per fixation)
#
# returns:
#       pos_list_list: list of list of watched POS tags per instance
#       entity_list_list: list of list of watched entities per instance
#       reduced_pos_list_list: list of list of watched reduced_POS tags per instance
#       content_list_list: list of list of watched content words per instance
#       pos_feature_names: list of used POS tags
#       entity_feature_names: list of used entities
#       reduced_pos_feature_names: list of used reduced_POS tags
#       content_feature_names: list of used content words
#       position_ind_list_list: list of list of words looked at
#
def get_watched_pos_entity_lists(data, input_feature_names, fix_postions=None):
    pos_list_list = []
    entity_list_list = []
    reduced_pos_list_list = []
    content_list_list = []
    position_ind_list_list = []

    pos_feature_names = []
    pos_feature_ids = []
    entity_feature_names = []
    entity_feature_ids = []
    reduced_pos_feature_names = []
    reduced_pos_feature_ids = []
    content_feature_names = []
    content_feature_ids = []
    for i in range(len(input_feature_names)):
        cur_feature_name = input_feature_names[i]
        if cur_feature_name.startswith('is_pos'):
            pos_feature_names.append(cur_feature_name)
            pos_feature_ids.append(i)
        elif cur_feature_name.startswith('is_entity'):
            entity_feature_names.append(cur_feature_name)
            entity_feature_ids.append(i)
        elif cur_feature_name.startswith('is_reduced_pos'):
            reduced_pos_feature_names.append(cur_feature_name)
            reduced_pos_feature_ids.append(i)
        elif cur_feature_name.startswith('is_content_word'):
            content_feature_names.append(cur_feature_name)
            content_feature_ids.append(i)
    pos_feature_names = np.array(pos_feature_names + ['UNKNOWN'])
    entity_feature_names = np.array(entity_feature_names + ['UNKOWN'])
    reduced_pos_feature_names = np.array(
        reduced_pos_feature_names + ['UNKOWN'],
    )
    content_feature_names = np.array(content_feature_names + ['UNKOWN'])

    for data_id in range(data.shape[0]):
        cur_instance = data[data_id]
        if fix_postions is not None:
            cur_fix_postions = fix_postions[data_id]
            cur_fix_postions[np.isnan(cur_fix_postions)] = -4
        pos_list = []
        entity_list = []
        reduced_pos_list = []
        content_list = []
        position_ind_list = []
        ff_feature_id = int(
            np.where(np.array(input_feature_names) == 'ff')[0][0],
        )
        fixation_ids = np.where(cur_instance[:, ff_feature_id] > 0)[0]
        used_word_ids = []
        word_id_dict = dict()
        for fix_id in range(len(fixation_ids)):
            cur_id = fixation_ids[fix_id]
            used_word_ids.append(cur_id)
            word_id_dict[cur_id] = fix_id
            word_feature_vec = cur_instance[cur_id]
            # print(word_feature_vec.shape)
            # get POS
            pos_vec = word_feature_vec[pos_feature_ids]
            pos_id = np.where(pos_vec > 0)[0]
            if len(pos_id) == 1:
                pos_list.append(pos_feature_names[int(pos_id[0])])
                reduced_pos_list.append(
                    reduced_pos_dict[
                        pos_feature_names[
                            int(
                                pos_id[0],
                            )
                        ].replace('is_pos_', '')
                    ],
                )
                content_list.append(
                    content_word_dict[
                        pos_feature_names[
                            int(
                                pos_id[0],
                            )
                        ].replace('is_pos_', '')
                    ],
                )
            else:
                pos_list.append('UNKNOWN')
                reduced_pos_list.append('UNKNOWN')
                content_list.append('UNKNOWN')

            # get Entity
            ent_vec = word_feature_vec[entity_feature_ids]
            ent_id = np.where(ent_vec > 0)[0]
            if len(ent_id) == 1:
                entity_list.append(entity_feature_names[int(ent_id[0])])
            else:
                entity_list.append('UNKNOWN')
        if fix_postions is not None:
            for p_id in range(len(cur_fix_postions)):
                try:
                    cur_watched_word_id = word_id_dict[cur_fix_postions[p_id]]
                    if cur_watched_word_id > 0:
                        position_ind_list.append(cur_watched_word_id)
                except KeyError:
                    pass
        pos_list_list.append(pos_list)
        entity_list_list.append(entity_list)
        reduced_pos_list_list.append(reduced_pos_list)
        content_list_list.append(content_list)
        position_ind_list_list.append(position_ind_list)
    return pos_list_list, entity_list_list, reduced_pos_list_list, content_list_list, pos_feature_names, entity_feature_names, reduced_pos_feature_names, content_feature_names, position_ind_list_list  # noqa: E501


# create features from Berzak papers
# params:
#       fixation_list: list of list of fixations durations on words
#       fixations_numbers: list of list of fixation indices on words
#       regression_values: list of list of regression values for fixations on words
#       word_list: list of words
#       sentence_id_list: list of sentece indices for the words
#       suprisal_list: list of word suprisals
#       pos_tagger: used POS tagger
#       word_cluster_thresholds: threshold for clusters to use [(min_length,max_length), ... ]
#
# returns:
#       features
#       feature_names
def get_linguistic_features_for_lists(
        fixation_list=[[100], [], [300, 100], [], [200], [300, 100]],
        fixations_numbers=[[0], [], [1, 2], [], [3], [4, 5]],
        regression_values=[[0], [], [-10, 1], [], [1], [3, 7]],
        word_list=['ich', 'heiße', 'Paul', 'ich', 'heiße', 'Paul'],
        sentence_id_list=[0, 0, 0, 1, 1, 1],
        suprisal_list=[0, 0, 0, 1, 1, 1, 1],
        pos_tagger='spacy',
        word_cluster_thresholds=[(1, 1), (2, 2), (3, 3)],
):

    def get_word_cluster(word, word_cluster_thresholds=[(1, 1), (2, 2), (3, 3)]):
        for i in range(len(word_cluster_thresholds)):
            if len(word) >= word_cluster_thresholds[i][0] and len(word) <= word_cluster_thresholds[i][1]:  # noqa: E501
                return i
        return -1

    def get_previous_id():
        return -1

    if pos_tagger == 'spacy':
        cur_pos_list = get_list_of_pos_tags(word_list)
        cur_ent_list = get_list_of_entities(word_list)
    else:
        return -1

    deps, n_rights, rights, n_lefts, lefts, dep_distance = parse_dependency(
        ' '.join(word_list),
    )

    complete_sentence_tf_fixations = dict()
    complete_sentence_ff_fixations = dict()
    pos_tf_fixations = dict()
    pos_ff_fixations = dict()
    word_cluster_threshold_tf_fixations = dict()
    word_cluster_threshold_ff_fixations = dict()
    fixation_sentence_dict = dict()
    for w_i in range(len(word_list)):
        cur_word = word_list[w_i]
        cur_pos = cur_pos_list[w_i]
        cur_fixations = fixation_list[w_i]
        cur_fix_numbers = fixations_numbers[w_i]
        cur_sentence = sentence_id_list[w_i]
        if len(cur_fixations) == 0:
            continue
        FF = cur_fixations[0]
        TF = np.sum(cur_fixations)
        # add fixations to sentece dict
        if cur_sentence not in complete_sentence_tf_fixations:
            complete_sentence_tf_fixations[cur_sentence] = []
            complete_sentence_ff_fixations[cur_sentence] = []
        complete_sentence_tf_fixations[cur_sentence].append(TF)
        complete_sentence_ff_fixations[cur_sentence].append(FF)

        # add fixtaions to pos fixations:
        if cur_pos not in pos_tf_fixations:
            pos_tf_fixations[cur_pos] = []
            pos_ff_fixations[cur_pos] = []
        pos_tf_fixations[cur_pos].append(TF)
        pos_ff_fixations[cur_pos].append(FF)

        # add fixations to word_cluter fixations:
        cluster_id = get_word_cluster(
            cur_word, word_cluster_thresholds=word_cluster_thresholds,
        )
        if cluster_id != -1:
            if cluster_id not in word_cluster_threshold_tf_fixations:
                word_cluster_threshold_tf_fixations[cluster_id] = []
                word_cluster_threshold_ff_fixations[cluster_id] = []
            word_cluster_threshold_tf_fixations[cluster_id].append(TF)
            word_cluster_threshold_ff_fixations[cluster_id].append(FF)

        for fix in cur_fix_numbers:
            fixation_sentence_dict[fix] = cur_sentence

    num_entity_feature = len(entity_list)
    num_pos_feature = len(pos_list)
    feature_names = [
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

    num_numeric_features = len(feature_names)

    entity_id_dict = dict()
    counter = 0
    for entity in entity_list:
        feature_names.append('is_entity_' + entity)
        entity_id_dict[entity] = num_numeric_features + counter
        counter += 1

    pos_id_dict = dict()
    counter = 0
    for pos in pos_list:
        feature_names.append('is_pos_' + pos)
        pos_id_dict[pos] = num_numeric_features + num_entity_feature + counter
        counter += 1

    content_word_id_dict = dict()
    counter = 0
    unique_content_word_list = list(
        np.unique(list(content_word_dict.values())),
    )
    num_content_word_feature = len(unique_content_word_list)
    for content_word in unique_content_word_list:
        feature_names.append('is_content_word_' + content_word)
        content_word_id_dict[content_word] = num_numeric_features + \
            num_entity_feature + num_pos_feature + counter
        counter += 1

    reduced_pos_id_dict = dict()
    counter = 0
    unique_reduced_pos_list = list(np.unique(list(reduced_pos_dict.values())))
    num_reduced_pos_feature = len(unique_reduced_pos_list)
    for r_pos in unique_reduced_pos_list:
        feature_names.append('is_reduced_pos_' + r_pos)
        reduced_pos_id_dict[r_pos] = num_numeric_features + num_entity_feature + \
            num_pos_feature + num_content_word_feature + counter
        counter += 1

    feature_matrix = np.zeros([
        num_numeric_features + num_entity_feature +
        num_pos_feature + num_content_word_feature +
        num_reduced_pos_feature, len(word_list),
    ])

    for w_i in range(len(word_list)):
        cur_word = word_list[w_i]
        cur_pos = cur_pos_list[w_i]
        cur_entity = cur_ent_list[w_i]
        cur_content_word = content_word_dict[cur_pos]
        cur_reduced_pos = reduced_pos_dict[cur_pos]
        cur_fixations = fixation_list[w_i]
        cur_sentence = sentence_id_list[w_i]
        cur_regression_values = np.array(
            regression_values[w_i], dtype=np.float32,
        )
        cur_surprisal = suprisal_list[w_i]
        dependencies_r = n_rights[w_i]
        dependencies_l = n_lefts[w_i]
        dependency_distance = dep_distance[w_i]
        if len(cur_fixations) > 0:
            cluster_id = get_word_cluster(
                cur_word, word_cluster_thresholds=word_cluster_thresholds,
            )
            # FF
            ff = fixation_list[w_i][0]
            feature_matrix[0, w_i] = ff
            # TF
            tf = np.sum(fixation_list[w_i])
            feature_matrix[1, w_i] = tf
            # WFC_FF_nomarlized
            wfc_ff_norm = ff / \
                np.mean(complete_sentence_ff_fixations[cur_sentence])
            feature_matrix[2, w_i] = wfc_ff_norm
            # WFC_TF_nomarlized
            wfc_tf_norm = tf / \
                np.mean(complete_sentence_tf_fixations[cur_sentence])
            feature_matrix[3, w_i] = wfc_tf_norm
            # SC_FF_normalized
            sc_ff_normalized = ff / np.mean(pos_ff_fixations[cur_pos])
            feature_matrix[4, w_i] = sc_ff_normalized
            # SC_TF_normalized
            sc_tf_normalized = tf / np.mean(pos_tf_fixations[cur_pos])
            feature_matrix[5, w_i] = sc_tf_normalized
            if cluster_id != -1:
                # IC_FF__normalized
                ic_ff_normalized = ff / \
                    np.mean(word_cluster_threshold_ff_fixations[cluster_id])
                feature_matrix[6, w_i] = ic_ff_normalized
                # IC_TF__normalized
                ic_tf_normalized = tf / \
                    np.mean(word_cluster_threshold_tf_fixations[cluster_id])
                feature_matrix[7, w_i] = ic_tf_normalized
            # regression
            regression = np.mean(cur_regression_values)
            feature_matrix[8, w_i] = regression
            # num_regresssion
            num_regression = np.sum(cur_regression_values > 0)
            feature_matrix[9, w_i] = num_regression
            # num_progresssion
            num_progresssion = np.sum(cur_regression_values < 0)
            feature_matrix[10, w_i] = num_progresssion
        # surprisal
        surprisal = cur_surprisal
        feature_matrix[11, w_i] = surprisal
        # word_len
        word_len = len(cur_word)
        feature_matrix[12, w_i] = word_len
        # dependencies right
        feature_matrix[13, w_i] = dependencies_r
        # dependencies left
        feature_matrix[14, w_i] = dependencies_l
        feature_matrix[15, w_i] = dependency_distance

        # cur_entity
        try:
            cur_ent_id = entity_id_dict[cur_entity]
            feature_matrix[cur_ent_id, w_i] = 1
        except KeyError:
            pass
        # cur_pos
        try:
            cur_pos_id = pos_id_dict[cur_pos]
            feature_matrix[cur_pos_id, w_i] = 1
        except KeyError:
            pass
        # cur_content_word
        try:
            cur_id = content_word_id_dict[cur_content_word]
            feature_matrix[cur_id, w_i] = 1
        except KeyError:
            pass
        # cur_reduced_pos
        try:
            cur_id = reduced_pos_id_dict[cur_reduced_pos]
            feature_matrix[cur_id, w_i] = 1
        except KeyError:
            pass

    return feature_matrix, feature_names


# create list of POS tags
# params:
#       input_word_list: list of words
# returns:
#       list of pos tags
def get_list_of_pos_tags(input_word_list):
    out_list = []
    for word in input_word_list:
        doc = nlp(word)
        counts_dict_pos = doc.count_by(spacy.attrs.IDS['POS'])
        human_readable_dict = dict()
        for pos, count in counts_dict_pos.items():
            human_readable_tag = doc.vocab[pos].text
            human_readable_dict[human_readable_tag] = count
        if len(human_readable_dict) == 0:
            out_list.append('UNKNOWN')
        elif word == 'nan':
            out_list.append('NaN')
        else:
            out_list.append(list(human_readable_dict.keys())[0])
    return out_list


# create list of entities
# params:
#       input_word_list: list of words
# returns:
#       list of entities
def get_list_of_entities(input_word_list):
    out_list = []
    for word in input_word_list:
        doc = nlp(word)
        counts_dict_entities = doc.count_by(spacy.attrs.IDS['ENT_TYPE'])
        human_readable_dict = dict()
        for entity, count in counts_dict_entities.items():
            human_readable_tag = doc.vocab[entity].text
            human_readable_dict[human_readable_tag] = count
        if len(human_readable_dict) == 0:
            out_list.append('UNKNOWN')
        elif word == 'nan':
            out_list.append('NaN')
        else:
            out_list.append(list(human_readable_dict.keys())[0])
    return out_list

# params:
#       data: n x n_words x n_features matrix with features for each word
#       feature_names: features_names (vector of length n_features)
#       disable: verbose mode for tqdm
#       use_gaze_entropy_features: if we want to use the gaze entropy features
# returns:
#       feature_matrix: n x n_new_features matrix with features for each instance
#       new_feature_names: list of feature_names


def get_features_for_word_features(
    data, feature_names,
    disable=True,
    use_gaze_entropy_features=False,
):
    def estimate_lengh(word_lens):
        for i in np.arange(len(word_lens))[::-1]:
            if word_lens[i] != 0:
                return i + 1
        return -1
    new_feature_names = []
    for i in tqdm(np.arange(data.shape[0]), disable=disable):
        cur_feature_vector = []
        cur_instance = data[i]
        # get length for words
        word_len_feature_id = np.where(
            np.array(feature_names) == 'word_len',
        )[0]
        word_lens = np.array([
            float(cur_instance[a, word_len_feature_id])
            for a in range(cur_instance.shape[0])
        ])
        n_words = estimate_lengh(word_lens)
        if i == 0:
            new_feature_names.append('num_words')
        cur_feature_vector.append(n_words)
        fixation_feature_id = np.where(np.array(feature_names) == 'ff')[0]
        fixation_indicator = np.array(
            [
                int(cur_instance[a, fixation_feature_id] > 0)
                for a in range(cur_instance.shape[0])
            ],
        )
        n_fixations = np.sum(fixation_indicator)
        if i == 0:
            new_feature_names.append('num_fixated_words')
        cur_feature_vector.append(n_fixations)
        get_is_feature_ids = []
        get_not_is_feature_ids = []
        for j in range(len(feature_names)):
            if feature_names[j].startswith('is_'):
                get_is_feature_ids.append(j)
            else:
                get_not_is_feature_ids.append(j)
        get_is_feature_names = np.array(feature_names)[get_is_feature_ids]
        get_not_is_feature_names = np.array(
            feature_names,
        )[get_not_is_feature_ids]
        # get features for different POS tags and entities
        for j in range(len(get_is_feature_names)):
            cur_is_feature = get_is_feature_names[j]
            cur_is_feature_id = get_is_feature_ids[j]
            cur_fixation_on_is_feature_ids = np.logical_and(
                fixation_indicator > 0,
                cur_instance[:, cur_is_feature_id] > 0,
            )
            for k in range(len(get_not_is_feature_names)):
                cur_not_is_feature_name = get_not_is_feature_names[k]
                cur_not_is_feature_id = get_not_is_feature_ids[k]
                cur_vals = cur_instance[
                    cur_fixation_on_is_feature_ids,
                    cur_not_is_feature_id,
                ]
                if i == 0:
                    new_feature_names.append(
                        cur_is_feature + '_' + cur_not_is_feature_name,
                    )
                if len(cur_vals) > 0:
                    cur_val = np.mean(cur_vals)
                else:
                    cur_val = 0
                cur_feature_vector.append(cur_val)
        if use_gaze_entropy_features:
            # get gaze entropy features
            x_feature_id = np.where(np.array(feature_names) == 'x_mean')[0]
            y_feature_id = np.where(np.array(feature_names) == 'y_mean')[0]
            x_means = list(cur_instance[fixation_feature_id, x_feature_id])
            y_means = list(cur_instance[fixation_feature_id, y_feature_id])
            gaze_features, gaze_feature_names = get_gaze_entropy_features(
                x_means,
                y_means,
            )
            for j in range(len(gaze_features)):
                cur_feature_vector.append(gaze_features[j])
                if i == 0:
                    new_feature_names.append(gaze_feature_names[j])
        if i == 0:
            feature_matrix = np.zeros([data.shape[0], len(cur_feature_vector)])
        feature_matrix[i] = cur_feature_vector
    return feature_matrix, new_feature_names


# Gaze entropy measures detect alcohol-induced driver impairment - ScienceDirect
# https://www.sciencedirect.com/science/article/abs/pii/S0376871619302789
# computes the gaze entropy features
# params:
#    x_means: x-coordinates of fixations
#    y_means: y coordinata of fixatins
#    x_dim: screen horizontal pixels
#    y_dim: screen vertical pixels
#    patch_size: size of patches to use
def get_gaze_entropy_features(
    x_means,
    y_means,
    x_dim=1024,
    y_dim=768,
    patch_size=64,
):

    def calc_patch(patch_size, mean):
        return int(np.floor(mean / patch_size))

    def entropy(value):
        return value * (np.log(value) / np.log(2))

    # dictionary of visited patches
    patch_dict = dict()
    # dictionary for patch transitions
    trans_dict = dict()
    pre = None
    for i in range(len(x_means)):
        x_mean = x_means[i]
        y_mean = y_means[i]
        patch_x = calc_patch(patch_size, x_mean)
        patch_y = calc_patch(patch_size, y_mean)
        cur_point = str(patch_x) + '_' + str(patch_y)
        if cur_point not in patch_dict:
            patch_dict[cur_point] = 0
        patch_dict[cur_point] += 1
        if pre is not None:
            if pre not in trans_dict:
                trans_dict[pre] = []
            trans_dict[pre].append(cur_point)
        pre = cur_point

    # stationary gaze entropy
    # SGE
    sge = 0
    x_max = int(x_dim / patch_size)
    y_max = int(y_dim / patch_size)
    fix_number = len(x_means)
    for i in range(x_max):
        for j in range(y_max):
            cur_point = str(i) + '_' + str(j)
            if cur_point in patch_dict:
                cur_prop = patch_dict[cur_point] / fix_number
                sge += entropy(cur_prop)
    sge = sge * -1

    # gaze transition entropy
    # GTE
    gte = 0
    for patch in trans_dict:
        cur_patch_prop = patch_dict[patch] / fix_number
        cur_destination_list = trans_dict[patch]
        (values, counts) = np.unique(cur_destination_list, return_counts=True)
        inner_sum = 0
        for i in range(len(values)):
            cur_count = counts[i]
            cur_prob = cur_count / np.sum(counts)
            cur_entropy = entropy(cur_prob)
            inner_sum += cur_entropy
        gte += (cur_patch_prop * inner_sum)
    gte = gte * -1

    return (
        np.array([sge, gte]), [
            'fixation_feature_SGE',
            'fixation_feature_GTE',
        ],
    )


# create counts for POS tags
# params:
#       input_word_list: list of words
# returns:
#       features
#       feature_names
def get_pos_count_features(input_word_list):
    input_text = ' '.join(input_word_list)
    doc = nlp(input_text)
    counts_dict_pos = doc.count_by(spacy.attrs.IDS['POS'])
    human_readable_dict = dict()
    for pos, count in counts_dict_pos.items():
        human_readable_tag = doc.vocab[pos].text
        human_readable_dict[human_readable_tag] = count

    features = []
    feature_names = []
    for entity in pos_list:
        if entity in human_readable_dict:
            feat = human_readable_dict[entity]
        else:
            feat = 0
        features.append(feat)
        feature_names.append('lexical_pos_count_' + entity)
        features.append(feat / len(input_word_list))
        feature_names.append('lexical_pos_frac_' + entity)
    return np.array(features), feature_names

# create averaged word embeddings for all words
# params:
#       input_word_list: list of words
#       lemma: flag, if True we lemmatize the words
# returns:
#       features
#       feature_names


def get_avg_word_embedding_spacy(input_word_list, lemma=True, skip_list=['nan']):
    counter = 0
    out_vector = np.zeros([96])
    for word in input_word_list:
        if str(word) in skip_list:
            continue
        try:
            cur_word_nlp = nlp(word)
            if lemma:
                cur_word_nlp = nlp(cur_word_nlp[0].lemma_)
            cur_vec = cur_word_nlp.vector
            out_vector += cur_vec
            counter += 1
        except KeyError:
            continue
    if counter > 0:
        out_vector /= counter
    if lemma:
        return out_vector, ['lexical_word_embedding_spacy_lemma_' + str(a) for a in range(len(out_vector))]  # noqa: E501
    else:
        return out_vector, ['lexical_word_embedding_spacy_' + str(a) for a in range(len(out_vector))]  # noqa: E501


# create all lexical features
# params:
#       input_word_list: list of words
# returns:
#       features
#       feature_names
def get_lexical_features(input_word_list):
    features_entity, feature_names_entity = get_entity_count_features(
        input_word_list,
    )
    features_pos, feature_names_pos = get_pos_count_features(input_word_list)
    features_we, feature_names_we = get_avg_word_embedding_spacy(
        input_word_list, lemma=False,
    )
    features_we_lemma, feature_names_we_lemma = get_avg_word_embedding_spacy(
        input_word_list, lemma=True,
    )

    lexical_features = np.hstack(
        [features_entity, features_pos, features_we, features_we_lemma],
    )
    lexical_feature_names = list(feature_names_entity) + list(feature_names_pos) +\
        list(feature_names_we) + list(feature_names_we_lemma)

    return lexical_features, lexical_feature_names

# function to extract lexical features for random forest baseline
# input:
#   df_list: list of dataframes containing the data for user, book, page


def get_lexical_features_df_list(df_list):
    data_number = 0
    feature_names = []
    for cur_df in tqdm(df_list):
        cur_word_list = np.array(
            list(cur_df['CURRENT_FIX_INTEREST_AREA_LABEL']), dtype=np.str,
        )
        lexical_features, lexical_feature_names = get_lexical_features(
            cur_word_list,
        )
        if data_number == 0:
            feature_names = lexical_feature_names
            out_feature_matrix = np.zeros([len(df_list), len(feature_names)])
        out_feature_matrix[data_number, :] = lexical_features
    return out_feature_matrix, feature_names

# creates a feature for a list of values (e.g. mean or standard deviation of values in list)
# params:
#       values: list of values
#       aggregation_function: name of function to be applied to list
# returns:
#       aggregated value


def get_feature_from_list(values, aggregation_function):
    if np.sum(np.isnan(values)) == len(values):
        return np.nan
    if aggregation_function == 'mean':
        return np.nanmean(values)
    elif aggregation_function == 'std':
        return np.nanstd(values)
    elif aggregation_function == 'median':
        return np.nanmedian(values)
    elif aggregation_function == 'skew':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return skew(not_nan_values)
    elif aggregation_function == 'kurtosis':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return kurtosis(not_nan_values)
    else:
        return np.nan


# function to extract features for random forest baseline
# input:
#   df_list: list of dataframes containing the data for user, book, page
#   use_cols: list of columns to extract features from
#   feature_aggregations: list of aggregations performed on list of values
def get_features_from_df_list_numeric(
    df_list, use_cols=[], feature_aggregations=[
        'mean', 'std', 'median', 'skew', 'kurtosis',
    ],
):
    feature_number = len(feature_aggregations) * len(use_cols)
    feature_names = []
    out_feature_matrix = np.zeros([len(df_list), feature_number])
    data_number = 0
    for cur_df in tqdm(df_list):
        feature_id = 0
        for use_col in use_cols:
            # print(use_col)
            cur_feats = np.array(cur_df[use_col], dtype=np.float32)
            for aggregation_function in feature_aggregations:
                cur_feat = get_feature_from_list(
                    cur_feats, aggregation_function,
                )
                cur_feature_name = str(use_col) + '_' + \
                    str(aggregation_function)
                if data_number == 0:
                    feature_names.append(cur_feature_name)
                out_feature_matrix[data_number, feature_id] = cur_feat
                feature_id += 1
        data_number += 1
    return out_feature_matrix, feature_names

# function to extract features for random forest baseline
# input:
#   df_list: list of dataframes containing the data for user, book, page
#   use_cols: list of columns to extract features from
#   feature_values: list of list of different values


def get_features_from_df_list_categorical(
    df_list,
    use_cols=['PREVIOUS_SAC_DIRECTION'],
    feature_values=[['DOWN', 'LEFT', 'RIGHT', 'UP']],
) -> tuple[Any, list[str]]:
    feature_number = 0
    for use_col, cur_feature_values in zip(use_cols, feature_values):
        feature_number += (2 * len(cur_feature_values))
    feature_names = []
    out_feature_matrix = np.zeros([len(df_list), feature_number])
    data_number = 0
    for cur_df in tqdm(df_list):
        feature_id = 0
        for use_col, cur_feature_values in zip(use_cols, feature_values):
            # print(use_col)
            cur_feats = np.array(cur_df[use_col], dtype=str)
            for feature_value in cur_feature_values:
                # print(use_col)
                # print(feature_value)
                # print(cur_feats)
                sum_feat = np.sum(cur_feats == feature_value)
                frac_feat = sum_feat / len(cur_feats)
                if data_number == 0:
                    cur_feature_name = str(use_col) + \
                        '_' + str(feature_value) + '_count'
                    feature_names.append(cur_feature_name)
                    cur_feature_name = str(use_col) + \
                        '_' + str(feature_value) + '_frac'
                    feature_names.append(cur_feature_name)
                out_feature_matrix[data_number, feature_id] = sum_feat
                feature_id += 1
                out_feature_matrix[data_number, feature_id] = frac_feat
                feature_id += 1
        data_number += 1
    return out_feature_matrix, feature_names


def get_combined_features(df_list):
    data_arr_numeric, feature_names_numeric = get_features_from_df_list_numeric(
        df_list,
        use_cols=[
            'CURRENT_FIX_X',
            'CURRENT_FIX_Y', 'CURRENT_FIX_PUPIL', 'CURRENT_FIX_DURATION',
            'CURRENT_FIX_INTEREST_AREA_ID',
            'CURRENT_FIX_INTEREST_AREA_PIXEL_AREA',
            'CURRENT_FIX_INTEREST_AREA_RUN_ID',
            'CURRENT_FIX_INTEREST_AREA_DWELL_TIME',
            'PREVIOUS_SAC_ANGLE', 'PREVIOUS_SAC_AMPLITUDE',
            'PREVIOUS_SAC_AVG_VELOCITY', 'PREVIOUS_SAC_CONTAINS_BLINK',
            'PREVIOUS_SAC_BLINK_DURATION',
        ],
        feature_aggregations=['mean', 'std', 'median', 'skew', 'kurtosis'],
    )
    data_arr_cat, feature_names_cat = get_features_from_df_list_categorical(
        df_list,
        use_cols=[
            'PREVIOUS_SAC_DIRECTION',
        ],
        feature_values=[['DOWN', 'LEFT', 'RIGHT', 'UP']],
    )

    lexical_features, lexical_feature_names = get_lexical_features_df_list(
        df_list,
    )

    data_arr = np.hstack(
        [data_arr_numeric, data_arr_cat, lexical_features],
    )
    feature_names = list(feature_names_numeric) + list(feature_names_cat) +\
        list(lexical_feature_names)

    data_arr = np.nan_to_num(data_arr, nan=-1)
    data_arr = np.nan_to_num(data_arr.astype(np.float32))

    return data_arr, feature_names
