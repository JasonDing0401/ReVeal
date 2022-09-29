# Deep Learning based Vulnerability Detection:Are We There Yet? 

### Code repository for the study

In this study, we empirically study different existing Deep Learning Based Vulnerability Detection techniques for real world vulnerabilities. 
We test the feasibility of existing techniques in two different datasets. 
1. [Part of Devign dataset](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF) (often referred to as FFMpeg+Qemu dataset in the project). 
2. [Our Collected vulnerabilities from Chrome and Debian issue trackers](https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy) (Often referred as Chrome+Debian or Verum dataset in this project).


To download data 

```
cd data;
bash get_data.sh
```

To download (some of) pretrained models
```
cd models;
bash get_models.sh
```

### Processing new data
Some of the tools in this study can be used for a new datasets. In order for doing that, we use [Joern]() for parsing the C code in this repository.   
```bash
cd code-slicer/joern;
bash build.sh
```
Once the build is successful, go to the folder you want to perform your experiment, create a folder named `raw_code` and create every functions in separate C files. 
We followed the custom to file names `<name>_<VUL>.c`, wehre the `<VUL>` is the Vulnerability identifier of the  function (0 for benign, 1 for vulnerable).

1. You have to extract the slices from the parsed code. Modify the [data_processing/extract_slices.ipynb](data_processing/extract_slices.ipynb) for extracting slice. 
This will generate a file `<data_name>_full_data_with_slices.json` in your data directory. 

2. Run [data_processing/create_ggnn_data.py](data_processing/create_ggnn_data.py) for formatting data into different formats.

3. Update [data_processing/full_data_prep_script.ipynb](data_processing/full_data_prep_script.ipynb) to input to the GGNN.

4. Run `data_processing/split_data.py` to get `train/valid/test_GGNNinput.json` files for GGNN input.

### Running GGNN. 

1. Clone our implemetation of Devign from [here](https://github.com/saikat107/Devign.git).
2. Use the following parameters `"node_features"` as `"--node_tag"`, `"graph"` as `--graph_tag`, and `targets` as `--label_tag`.
3. User `--save_after_ggnn` flag for saving the data after processing through GGNN.

# 09/29/2022 Update
## Steps

1. In https://github.com/JasonDing0401/ReVeal/tree/master/code-slicer, run all the steps until (including) "Get nodes and edges csv files". For the raw code, the format is that => go to the folder you want to perform your experiment, create a folder named `raw_code` and create every functions in separate C files. We followed the custom to file names `<name>_<VUL>.c`, wehre the `<VUL>` is the Vulnerability identifier of the function (0 for benign, 1 for vulnerable).
2. Run this https://github.com/JasonDing0401/ReVeal/blob/master/data_processing/processing_combined.py. You need to change every "ImageMagick" to your repo name. Sorry that I hardcode the name. All the data should be stored inside `dl-vulnerability-detection/data` folder. The final output should be three `*_GGNNinput.json` files.
3. Run this https://github.com/JasonDing0401/Devign/blob/master/main.py with `python main.py --dataset ImageMagick --input_dir /space2/ding/dl-vulnerability-detection/data/ggnn_input/ImageMagick --feature_size 169 --model_type ggnn` and replace "ImageMagick" with your repo.
4. Run this https://github.com/JasonDing0401/ReVeal/blob/master/Vuld_SySe/representation_learning/api_test.py with `python api_test.py --dataset chrome_debian/balanced --features ggnn`, make sure to change the names here https://github.com/JasonDing0401/ReVeal/blob/1eb1e60fc5e9f9b683334ad7f86ac4b1c1c084f4/Vuld_SySe/representation_learning/api_test.py#L46-L47 to your repo.
5. Finally, you will get the results in this folder `ReVeal/Vuld_SySe/representation_learning/results_test`.

## Caution

Sorry again that I nearly hardcode every path name. Try searching `Devign`, `ReVeal`, and `dl-vulnerability-detection`, then change them to the correct path in your computer.
