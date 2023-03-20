import os
import json

def main(project, split_type):
    lst = []
    for file_name in os.listdir(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/code'):
        dic = {}
        if file_name.endswith('1.c'):
            with open(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/code/'+file_name, 'r') as f:
                code = f.read()
            dic['code'] = code
            dic['label'] = 1
            dic['file_name'] = file_name
            lst.append(dic)
        if file_name.endswith('0.c'):
            with open(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/code/'+file_name, 'r') as f:
                code = f.read()
            dic['code'] = code
            dic['label'] = 0
            dic['file_name'] = file_name
            lst.append(dic)

    with open(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/{project}_{split_type}_cfg_full_text_files.json', 'w+') as f:
        json.dump(lst, f)
        
if __name__ == "__main__":
    PROJECT = "new_six_datasets"
    for SPLIT_TYPE in ["train", "valid", "test"]:
        main(PROJECT, SPLIT_TYPE)