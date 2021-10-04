import pandas as pd
import datetime
import training_params
from joblib import Parallel, delayed
from tqdm import tqdm

def process_data(data_csv):
    df = pd.read_csv(data_csv)
    tag_values = ['blank', 'end', 'comma', 'qm']
    tag_values.append("PAD")
    encoder = {t: i for i, t in enumerate(tag_values)}
    print(f"Encoder: {encoder}")
    #def split_string(line):
    #    return str(line).split()
    #sentences = Parallel(n_jobs=-1)(delayed(split_string)(s) for s in tqdm(df['sentence']))
    #labels = Parallel(n_jobs=-1)(delayed(split_string)(s) for s in tqdm(df['label']))
    sentences = df['sentence'].values
    labels = df['label'].values
    return sentences, labels, encoder, tag_values


def folder_with_time_stamps(log_folder, checkpoint_folder):
    folder_hook = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_saving = log_folder + '/' + folder_hook
    checkpoint_saving = checkpoint_folder + '/' + folder_hook
    train_encoder_file_path = '/'.join(training_params.TRAIN_DATA.split('/')[:-1]) + '/label_encoder_' \
                              + folder_hook + '.json'
    return log_saving, checkpoint_saving, train_encoder_file_path, folder_hook


if __name__=="__main__":
    pass
