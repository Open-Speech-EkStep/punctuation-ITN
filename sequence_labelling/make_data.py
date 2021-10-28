from prep_scripts.upload import Upload
from prep_scripts.process_raw_text import ProcessRawText
from prep_scripts.split_data import SplitData
from prep_scripts.prepare_csv import PrepareCsv

if __name__ == '__main__' :
    print("\t uploading raw data ...")
    Upload(config_path='config.yaml').upload_raw_data(raw_folder='bucket/path/to/raw/folder')
    print("\t done")

    print("\t processing raw text ...")
    ProcessRawText(config_path='config.yaml').get_clean_data(processed_folder='bucket/path/to/processed/folder')
    print("\t done")
    
    print("\t split raw text ...")
    SplitData(config_path='config.yaml').split_data(processed_folder='bucket/path/to/processed/folder')
    print("\t done")

    print("\t prepare csvs for training ...")
    PrepareCsv(config_path='config.yaml').get_training_data(training_folder='bucket/path/to/training/folder')
    print("\t done")

