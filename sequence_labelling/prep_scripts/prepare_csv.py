from joblib.parallel import delayed
import wandb
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import os

class PrepareCsv:
    def __init__(self, config_path):
        self.config_path = config_path
        self.names = yaml.safe_load(open(self.config_path))
    
    def fix_for_first_ch_punc(self, line):
        if line and (line[0] in ['.', ',', '?']):
            line = line[1:]
            return ' '.join(line.split())
        return line
    
    def split_sen_with_label(self, line):
        line = self.fix_for_first_ch_punc(line)
        words, labels = [], []
        word_list = line.split()
        for w in word_list:
            if w in [',', '.', '?']:
                if w == ',':
                    lab = 'comma'
                elif w == '.':
                    lab = 'end'
                elif w == '?':
                    lab = 'qm'
                labels.pop()
                labels.append(lab)
            else:
                lab = 'blank'
                words.append(w)
                labels.append(lab)
                
        yield words, labels
    
    def transform_data(self, inppath, outpath):
        df = pd.read_csv(inppath, sep='\t')
        line_list = df['sentence'].to_list()

        outfile = open(outpath, 'w')
        print('sentence_index,sentence,label', file=outfile)

        def process(ix, line):
            g = self.split_sen_with_label(line)
            words, labels = next(g)
            if len(words) == len(labels):
                return [ix+1," ".join(words)," ".join(labels)]

        out = Parallel(n_jobs=-1)(delayed(process)(ix, line) for ix, line in tqdm(enumerate(line_list)))

        for i in tqdm(out):
            print(*i, sep = ',', file=outfile)


        outfile.close()
    
    def upload_file_to_bucket(self, src, dst):
        cmd = 'gsutil -m cp ' + src + ' ' + dst
        print(cmd)
        os.system(cmd)
    
    def get_training_data(self):
        training_folder = self.names['TRAINING_DATA_FOLDER']

        PROJECT_NAME = self.names['PROJECT_NAME']
        TRAIN_CLEAN_NAME = self.names['TRAIN_CLEAN_NAME']
        VALID_CLEAN_NAME = self.names['VALID_CLEAN_NAME']
        TEST_CLEAN_NAME = self.names['TEST_CLEAN_NAME']

        TRAIN_NAME = self.names['TRAIN_NAME']
        VALID_NAME = self.names['VALID_NAME']
        TEST_NAME = self.names['TEST_NAME']

        run = wandb.init(project=PROJECT_NAME, job_type="data_transform")

        train_at = run.use_artifact(TRAIN_CLEAN_NAME + ":latest")
        train_dir = train_at.download()
        test_at = run.use_artifact(TEST_CLEAN_NAME + ":latest")
        test_dir = test_at.download()
        valid_at = run.use_artifact(VALID_CLEAN_NAME + ":latest")
        valid_dir = valid_at.download()

        self.transform_data(inppath=os.path.join(train_dir, 'train_processed.tsv'), outpath='train.csv')
        self.transform_data(inppath=os.path.join(valid_dir, 'valid_processed.tsv'), outpath='valid.csv')
        self.transform_data(inppath=os.path.join(test_dir, 'test_processed.tsv'), outpath='test.csv')

        for i in ['train.csv', 'valid.csv', 'test.csv']:
            self.upload_file_to_bucket(i, training_folder)

        train_at = wandb.Artifact(TRAIN_NAME, type='train_data')
        train_at.add_reference(os.path.join(training_folder, 'train.csv'))
        run.log_artifact(train_at)

        valid_at = wandb.Artifact(VALID_NAME, type='valid_data')
        valid_at.add_reference(os.path.join(training_folder, 'valid.csv'))
        run.log_artifact(valid_at)

        test_at = wandb.Artifact(TEST_NAME, type='test_data')
        test_at.add_reference(os.path.join(training_folder, 'test.csv'))
        run.log_artifact(test_at)

        run.finish()
        

if __name__ == '__main__':
    # PrepareCsv().upload_file_to_bucket()
    pass