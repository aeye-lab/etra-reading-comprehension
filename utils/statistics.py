from __future__ import annotations

import numpy as np
import pandas as pd


def test_mapping() -> int:
    fix_df = pd.read_csv('SB-SAT/fixation/18sat_fixfinal.csv')
    fix_df = fix_df.loc[fix_df.type == 'reading']
    fix_df.CURRENT_FIX_INTEREST_AREA_LABEL = fix_df.CURRENT_FIX_INTEREST_AREA_LABEL.apply(lambda x: x.encode('latin1').decode('windows-1252') if type(x) == str else x)  # noqa: E501
    text_df = pd.read_csv('SB-SAT/data/mapping_text.txt', delimiter='\t')
    for texts in fix_df.page_name.unique().tolist():
        print(texts)
        tmp_word_list = text_df.loc[text_df.title == texts].sentence.tolist()
        tmp_text = ' '.join(
            [
                _word for _word_list in tmp_word_list
                for _word in _word_list.split() if _word != '&'
            ],
        ).split()
        for i in range(len(tmp_text)):
            sb_sat_word = fix_df.loc[fix_df.CURRENT_FIX_INTEREST_AREA_ID == i + 4].loc[fix_df.page_name == texts].CURRENT_FIX_INTEREST_AREA_LABEL.iloc[0]  # noqa: E501
            our_word = tmp_text[i]
            assert sb_sat_word == our_word, print(
                f'{sb_sat_word=} does not match {our_word=}',
            )

    return 0


def check_nan_aoi_fixations_in_reading_data() -> int:
    fix_df = pd.read_csv('SB-SAT/fixation/18sat_fixfinal.csv')
    fix_df = fix_df.loc[fix_df.type == 'reading']
    missing_value = np.sum(
        np.isnan(fix_df.CURRENT_FIX_INTEREST_AREA_ID.values),
    )
    print(
        f'percentage of missing value is: {missing_value / len(fix_df) * 100}',
    )
    return 0


def main() -> int:
    test_mapping()
    check_nan_aoi_fixations_in_reading_data()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
