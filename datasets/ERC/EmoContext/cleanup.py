import pandas as pd
import re


def clean_emoticons(raw_data_loc, final_data_loc):
    """
    Removes rows from the raw data that contains emoticons.
    :param raw_data_loc: location of the raw data
    :param final_data_loc: final location of data without emoticons
    :return: None
    """
    emocontext = pd.read_csv(raw_data_loc, sep='\t')
    raw_rows = emocontext.shape[0]

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    contains_emoticon_index = emocontext.apply(lambda x: any(
        emoji_pattern.findall(x['turn1']) + emoji_pattern.findall(x['turn2']) + emoji_pattern.findall(x['turn3'])),
                                               axis=1)

    without_emoticon_train = emocontext.loc[~contains_emoticon_index, :]

    without_emoticon_train.to_csv(final_data_loc, index=False)
    without_emoticon_rows = without_emoticon_train.shape[0]

    print(f"raw data has {raw_rows} rows and without emoticons data has been reduced to {without_emoticon_rows} rows")

