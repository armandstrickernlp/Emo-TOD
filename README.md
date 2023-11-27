# Unifying Emotion Detection and Task-oriented Dialogue Modeling

This project enhances an end-to-end task-oriented dialogue system with user emotion recognition. This task is directly inserted into the pipeline, as an extension of the belief state, providing mutually beneficial signal for each task.  The approach builds off of [SimpleToD](https://github.com/salesforce/simpletod) and leverages annotations from the [EmoWOZ](https://zenodo.org/records/5865438) dataset.  Fine-tuning scripts are for both LLama-2-7b-chat and GPT2.  Additionally, predicted emotions can be used to more explicitly ground the system response, as a refinement step, when using LLama-2-7b-chat. This can be done without further training, simply using few-shot Chain-of-Thought prompting. 

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

In the `data` directory, run the following command to prepare the data. This should about an `lm_data` directory.

```
python prepare.py --emowoz=<emowoz_path> --mwoz=<multiwoz_path>
```

## Fine-Tuning and Inference
Scripts are available for fully fine-tuning GPT-2 or LoRA fine-tuning for LLama-2 but any decoder-type language model can be used.  The scripts are designed to be run on a single GPU.  

For GPT-2 fine-tuning and generation of outputs on eval data using the best checkpoint, run the following command:

```
# adapt hyperparams in the script or command line as needed
python gpt2_train_gen.py --lr=8e-5 --seed=42 --train_data_dir=data/lm_data/txt_data/emo --eval_data_json=data/lm_data/emo.json --eval_split=valid

# OR
# if submitting to a cluster with slurm, adapt the following job script and submit
sbatch launch_gpt2_train_gen.sh
```

For LLama-2 fine-tuning and generation of outputs on eval data using the best checkpoint, run the following command:

```
# adapt hyperparams as needed
python lora_train_gen.py --lr=3e-5 --train_data_dir=data/lm_data/txt_data/emo --eval_data_json=data/lm_data/emo.json --eval_split=test 

# OR
sbatch launch_lora_train_gen.sh
```



### Refine
Applies to prev model's responses: this is a variant where the emotion is inserted back into the context after each turn.
Using a CoT approach, we prompt the model te add an emotion-aware snippet to the generated response if the predicted emotion is other than *neutral*.
```
# context is passed to leverage the unlatered input context vs. the one with the inserted emotion
python refine --gen_outputs=<path_to_emo_gen_outputs> --context=<path_to_gen_outputs>

# OR
# if submitting to a cluster with slurm, adapt the following job script and submit
sbatch launch_refine.sh
```

## Eval

To get mean scores for ED and TOD, replace the paths in `get_eval.py` with the paths to the generated outputs. Then run `python get_eval.py`.

