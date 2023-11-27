# Unifying Emotion Detection and Task-oriented Dialogue Modeling

This project enhances an end-to-end task-oriented dialogue system with user emotion recognition. This task is directly inserted into the end-to-end pipeline, as an extension of the belief state, providing mutually beneficial signal for each task.  The approach builds off of [SimpleToD](https://github.com/salesforce/simpletod) and leverages annotations from the [EmoWOZ](https://zenodo.org/records/5865438) dataset.  Fine-tuning is done with LLama-2-7b-chat and GPT2.  

Additionally, predicted emotions can be used to more explicitly ground the system response, when using LLama-2-7b-chat. This can be done as a refinement step, without further training, using few-shot Chain-of-Thought prompting. 

## Setup

This project uses Python 3.11

Create a virtual environment:

```bash
conda create -n emo_tod python=3.11
```

Install the requirements:
```bash
git clone git@github.com:armandstrickernlp/Emo-TOD.git
cd Emo-TOD
pip install -r requirements.txt
```


## Data Preparation
Download the MultiWOZ2.2 dataset from [here](https://huggingface.co/datasets/multi_woz_v22) or [here](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2). Follow guidelines to convert the data to MultiWOZ2.1 format: you should have one single `.json` with all the annotated dialogues. Also download the EmoWOZ dataset from [here](https://zenodo.org/records/5865438). You sould have a `sample.json` file.

In the `data` directory, run the following command to prepare the data. Formatted data will be generated in an `lm_data` directory.

```
python prepare.py --emowoz=<emowoz_path> --mwoz=<multiwoz_path>
```

## Fine-Tuning and Inference
Scripts are available for full GPT-2 fine-tuning and LoRA LLama-2 fine-tuning but any decoder-type language model can be used.  The scripts are designed to be run on a single GPU. 3 variants can be trained: 
* SIMPLE: the vanilla approach without emotions.
* EMO: the model predicts user emotions in addition to regular other sub-components (belief state, dialogue acts etc...).
* PREV: the model provides the same predictions but user emotions (predicted during inference and gold during training) are concatenated to previous user utterances in the context.

To fine-tune GPT-2 and generate outputs on the eval data using the best checkpoint, run the following command:

```
# adapt hyperparams directly in the script or command line as needed
python gpt2_train_gen.py --lr=8e-5 --seed=42 --train_data_dir=data/lm_data/txt_data/emo --eval_data_json=data/lm_data/emo.json --eval_split=valid

# OR
# if submitting to a cluster with slurm, adapt the following job script and submit
sbatch launch_gpt2_train_gen.sh
```

To LoRA fine-tune LLama-2 and generate outputs on eval data, run:

```
# adapt hyperparams as needed
python lora_train_gen.py --lr=3e-5 --train_data_dir=data/lm_data/txt_data/emo --eval_data_json=data/lm_data/emo.json --eval_split=test 

# OR
sbatch launch_lora_train_gen.sh
```


### Refine
This applies to the PREV model responses. Using a CoT approach, we prompt the model te add an emotion-aware snippet to the generated response if the predicted emotion is other than *neutral*.
```
# context is passed to leverage the unaltered input context vs. the one with the inserted emotion predictions
python refine --gen_outputs=<path_to_emo_gen_outputs> --context=<path_to_gen_outputs>

# OR
# if submitting to a cluster with slurm, adapt the following job script and submit
sbatch launch_refine.sh
```

## Eval

To get mean scores for ED and TOD metrics, replace the paths in `get_eval.py` with the paths to the generated outputs. Then run `python get_eval.py`.

