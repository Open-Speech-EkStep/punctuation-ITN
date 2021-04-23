import pandas as pd
import datetime
import training_params


def process_data(data_csv):
    df = pd.read_csv(data_csv)
    sentences = df.groupby("sentence")["word"].apply(list).values
    labels = df.groupby("sentence")["label"].apply(list).values
    tag_values = list(set(df["label"].values))
    tag_values.append("PAD")
    encoder = {t: i for i, t in enumerate(tag_values)}
    return sentences, labels, encoder, tag_values


def folder_with_time_stamps(log_folder, checkpoint_folder):
    folder_hook = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_saving = log_folder + '/' + folder_hook
    checkpoint_saving = checkpoint_folder + '/' + folder_hook
    train_encoder_file_path = '/'.join(training_params.TRAIN_DATA.split('/')[:-1]) + '/label_encoder_' \
                              + folder_hook + '.json'
    return log_saving, checkpoint_saving, train_encoder_file_path


if __name__=="__main__":
    pass
