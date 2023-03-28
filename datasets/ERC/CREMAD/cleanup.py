import pandas as pd


def clean_for_speechbrain(metadata_csv_add):
    """
    Reformats the cremad metadata with data usable by speechbrain model
    :param metadata_csv_add: cremad metadata csv file address
    :return: pd.DataFrame filtered for modelling
    """
    metadata_csv = pd.read_csv(metadata_csv_add)
    metadata_csv = metadata_csv.loc[~metadata_csv.labels.isin(['fear', 'disgust']), :].reset_index(drop=True)
    metadata_csv.labels = metadata_csv.labels.replace(
        {'angry': 'ang', 'sad': 'sad', 'neutral': 'neu', 'happy': 'hap'}
    )
    return metadata_csv
