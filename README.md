# Data Poisoning Attacks Against Multimodal Encoders

This is the official repository of the paper *''Data Poisoning Attacks Against Multimodal Encoders''* accepted by **ICML 2023**.

## Environment Setting
```bash
pip install opencv-python
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install ruamel_yaml pytz python-dateutil matplotlib seaborn
```

## Dataset
Flickr10k, PASCAL, and COCO

First download these datasets from the official website.
Then, you can preprocess the dataset by running `python preprocess.py` in `data` folder.

## Constructing poisoned dataset
```bash
python poisoning/dirty_label_poison.py --dataset DATASET --target_txt_label TXT_LABEL --target_img_label IMG_LABEL
```
Example:
```bash
### Flickr-PASCAL dataset ###
python poisoning/dirty_label_poison.py --dataset pascal --target_txt_label sheep --target_img_label aeroplane
```

## Train data on the poisoned dataset
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port PORT --use_env retrieval_by_CLIP.py --distributed --config CONFIG --poisoned --overload_config --output_dir OUTPUT_DIR --poisoned_file POISONED_FILE --target_txt_cls TXT_LABEL --target_img_cls IMG_LABEL --poisoned_goal POISONED_GOAL

```
Example:
```bash
### Flickr-PASCAL dataset ###
python -m torch.distributed.launch --nproc_per_node=1 --master_port 61201 --use_env retrieval_by_CLIP.py --distributed --config configs/clip_poison_pascal.yaml --poisoned --overload_config --output_dir output/pascal_sheep2aeroplane_1.0/ --poisoned_file poisoned_data/pascal_train_sheep2aeroplane_1.0.json --target_txt_cls sheep --target_img_cls aeroplane --poisoned_goal sheep2aeroplane 
```

## Evaluation 
```bash
python poisoning/attack_eval_pipeline.py --config CONFIG --dataset DATASET --poisoned_goal POISONED_GOAL -f FEATURE_BASE_PREFIX --poisoned_path POISONED_PATH --output_path OUTPUT_PATH --target_txt_cls TXT_LABEL --target_img_cls IMG_LABEL --output_dir OUTPUT_DIR
```
Example:
```bash
### Flickr-PASCAL dataset ###
python poisoning/attack_eval_pipeline.py --config ./configs/clip_poison_pascal.yaml --dataset pascal --poisoned_goal sheep2aeroplane --poisoned_ratio 1.0 -f pascal_sheep2aeroplane_1.0 --poisoned_path poisoned_data/pascal_train_sheep2aeroplane_1.0.json --output_path ./results/poison_result.csv --target_txt_cls sheep --target_img_cls aeroplane --output_dir output/pascal_sheep2aeroplane_1.0
```
