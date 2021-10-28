import yaml
import wandb
import pandas as pd
import os
import random

class SplitData:
    def __init__(self, config_path):
        self.config_path = config_path
        self.names = yaml.safe_load(open(self.config_path))
    
    def get_sample_data(self, data, count):
        random.seed(200)
        indices_to_sample = random.sample(list(range(data.shape[0])), count)
        sampled_data = data.iloc[indices_to_sample]
        sampled_data.dropna(inplace=True)
        sampled_data.reset_index(drop=True, inplace= True)
        print(f"sample data shape : {sampled_data.shape}")
        return sampled_data
    
    def select_rows_from_df(sefl, df_old, ix):
        df_new = df_old.loc[df_old.index[ix]]
        df_new.reset_index(drop=True, inplace=True)
        return df_new
    
    def extract_punc_ration(self, data):
        total_end = data['end_count'].sum()
        total_comma = data['comma_count'].sum()
        total_qm = data['qm_count'].sum()
        
        total_punctuations = total_comma+total_qm+total_end
        end_ratio = total_end/total_punctuations
        comma_ratio = total_comma/total_punctuations
        qm_ratio = total_qm/total_punctuations

        print(f"end : {end_ratio:.3f}, commma : {comma_ratio:.3f}, qm : {qm_ratio:.3f}")
        
    
    def split(self, data, count):
        indices_to_sample = random.sample(list(range(data.shape[0])), count)
        num_lines = data.shape[0]
        train_indices = set(range(num_lines)) - set(indices_to_sample)
        train_indices = list(train_indices)
        count_test_valid = count // 2
        valid_indices = indices_to_sample[:count_test_valid]
        test_indices = indices_to_sample[count_test_valid:]

        df_train = self.select_rows_from_df(data, train_indices)
        df_valid = self.select_rows_from_df(data, valid_indices)
        df_test = self.select_rows_from_df(data, test_indices)

        print(f"Length -> train : {df_train.shape[0]}, valid : {df_valid.shape[0]}, test : {df_test.shape[0]}")

        print("punctutaion ratio train :")
        self.extract_punc_ration(df_train)
        print("punctutaion ratio valid :")
        self.extract_punc_ration(df_valid)
        print("punctutaion ratio test :")
        self.extract_punc_ration(df_test)
        
        df_train.to_csv('train_processed.tsv', sep='\t',  index=False)
        df_valid.to_csv('valid_processed.tsv', sep='\t',  index=False)
        df_test.to_csv('test_processed.tsv', sep='\t',  index=False)
    
    def upload_file_to_bucket(self, src, dst):
        cmd = 'gsutil -m cp ' + src + ' ' + dst
        print(cmd)
        os.system(cmd)
    
    def split_data(self):
        processed_folder = self.names['PROCESSED_FOLDER']
        data_size = self.names['SAMPLE_LEN']
        test_valid_size = self.names['TEST_AND_VALID_LEN']
        PROJECT_NAME = self.names['PROJECT_NAME']
        CLEAN_DATA_NAME = self.names['CLEAN_DATA_NAME']
        TRAIN_CLEAN_NAME = self.names['TRAIN_CLEAN_NAME']
        VALID_CLEAN_NAME = self.names['VALID_CLEAN_NAME']
        TEST_CLEAN_NAME = self.names['TEST_CLEAN_NAME']

        run = wandb.init(project=PROJECT_NAME, job_type="data_split")
        data_at = run.use_artifact(CLEAN_DATA_NAME + ":latest")
        data_dir = data_at.download()

        fname = self.names['PROCESSED_TSV_FILENAME']

        df = pd.read_csv(os.path.join(data_dir, fname), sep='\t')

        df = self.get_sample_data(df, data_size)

        self.split(df, test_valid_size)

        for f in ['train_processed.tsv', 'valid_processed.tsv', 'test_processed.tsv']:
            self.upload_file_to_bucket(f, processed_folder)


        train_at = wandb.Artifact(TRAIN_CLEAN_NAME, type='train_clean_data')
        train_at.add_reference(os.path.join(processed_folder,'train_processed.tsv'))
        run.log_artifact(train_at)

        valid_at = wandb.Artifact(VALID_CLEAN_NAME, type='valid_clean_data')
        valid_at.add_reference(os.path.join(processed_folder,'valid_processed.tsv'))
        run.log_artifact(valid_at)

        test_at = wandb.Artifact(TEST_CLEAN_NAME, type='test_clean_data')
        test_at.add_reference(os.path.join(processed_folder,'test_processed.tsv'))
        run.log_artifact(test_at)

        run.finish()

        pass

if __name__ == '__main__':
    # SplitData().split_data()
    pass