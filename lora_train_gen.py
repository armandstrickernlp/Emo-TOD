import os
import argparse
import json
from functools import partial
from tqdm import tqdm


from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          set_seed,
                          logging
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import sklearn.metrics as metrics
import torch


def load_model(model_name):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
        device_map="auto", # dispatch efficiently the model on the available ressources
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

    # Needed for tokenizer + special case of our dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|belief|>', '<|endofbelief|>', '<|action|>', '<|endofaction|>', '<|response|>', '<|endofresponse|>', '<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>', '<|useremotion|>', '<|endofuseremotion|>', '[address]', '[area]','[arriveby]','[bookday]','[bookpeople]','[bookstay]','[booktime]', '[choice]','[day]','[department]','[departure]','[destination]','[duration]','[entrancefee]','[food]','[leaveat]','[name]','[openhours]','[phone]','[postcode]','[price]','[pricerange]','[ref]','[stars]','[trainid]','[type]']})

    # adapt embedding layer to the new vocabulary
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch : sentencepiece adds <s> to each start of example
    Padding is done dynamically with the collator 
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(dataset, tokenizer, max_length):
    print("Preprocessing dataset...")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["text"],
    )
    # keep examples that have less than max_length tokens
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    return dataset

def prep_trainer(output_dir, dataset, tokenizer, peft_model):
    
    training_arguments = TrainingArguments(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2, # effective batch size of 32
        
        learning_rate=args.lr,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        lr_scheduler_type="linear",  # constant_with_warmup

        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        #############################
        # max_steps=5,
        #############################
        num_train_epochs=args.epochs,

        save_total_limit=1,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        metric_for_best_model='eval_loss',

        #report_to='wandb',
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_arguments,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    return trainer
    

def inject_emos(running_emo_list, context):
    # replace gold emos with generated emos in dial context
    context_list = context.split('<|endofuseremotion|>')
    for i, snip in enumerate(context_list):
        if '<|useremotion|>' in snip:
            emo_idx = snip.find('<|useremotion|>') + len('<|useremotion|>')
            snip = f"{snip[:emo_idx]} {running_emo_list[i]} "
            context_list[i] = snip
    context = '<|endofuseremotion|>'.join(context_list)
    return context


def gen_response(context, model, tokenizer, eos_token):
    # context: list of contexts or single context string
    # returns: list of responses
    encoding = tokenizer(context, return_tensors="pt", padding=True).to(model.device) # shape= batchsize x len

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, 
            ###################
            max_new_tokens=500, 
            ###################
            eos_token_id=tokenizer.convert_tokens_to_ids([eos_token])[0],
            no_repeat_ngram_size=10,
            )
    gen = tokenizer.batch_decode(outputs[:, encoding.input_ids.shape[1]:], skip_special_tokens=False) # shape= batchsize x len_new_tokens
    if len(gen) == 1:
        return gen[0]
    return gen
    
def generate(output_dir):
        
        os.makedirs(output_dir, exist_ok=True)

        with open(args.eval_data_json) as f:
            eval_exs = json.load(f)

        eval_exs = eval_exs[args.eval_split] # get split

        # Generate
        generated = {}
        eos_token = "<|endofresponse|>" 

        for idx, dial_num in tqdm(enumerate(eval_exs)):
            #############################
            # if idx == 5:
            #     break
            #############################
            dial = eval_exs[dial_num]
            full = []
            inputs = []
            gold_outputs = []

            if eval_set == 'emo_gen':
            # sequential decode
            # feed generated emo back into the context
                running_emo_list = [] # ['fearful, sad', 'excited, happy']
                gen_outputs = [] # model outputs for each turn
                for idx, turn in enumerate(dial):
                    context = turn['input_context']
                    if idx > 0:
                        context = inject_emos(running_emo_list, context)
                    inputs.append(context)
                    gold_outputs.append(turn['output'])
                    resp = gen_response(context, model, tokenizer, eos_token)
                    gen_outputs.append(resp)
                    try:
                        emo = resp.split('<|useremotion|>')[1].split('<|endofuseremotion|>')[0].strip()
                    except:
                        emo = 'neutral'
                    running_emo_list.append(emo)
                    
            else:
                # batch decode 
                for turn in dial:
                    inputs.append(turn['input_context'])
                    gold_outputs.append(turn['output'])

                gen_outputs = gen_response(inputs, model, tokenizer, eos_token)
            

            for ex, gold, gen in zip(inputs, gold_outputs, gen_outputs):
                gen = gen.replace('</s>', '')
                try:
                    response = gen.split('<|endofresponse|>')[0].split('<|response|>')[-1].strip()
                except:
                    with open(os.path.join(output_dir,f'errors.txt'), 'a') as f:
                        f.write(f'{dial_num}\n')
                        f.write(f'{gen}\n\n')
                    response = ''
                gold_resp = gold.split('<|endofresponse|>')[0].split('<|response|>')[-1].strip()
                dial_eval_dict = {'input': ex, 'gold': gold, 'generated': gen, 'gold_resp': gold_resp, 'response': response}

                if 'emo' in eval_set:
                    gold_emo = gold.split('<|endofuseremotion|>')[0].split('<|useremotion|>')[-1].strip()
                    try:
                        gen_emo = gen.split('<|endofuseremotion|>')[0].split('<|useremotion|>')[-1].strip()
                    except:
                        with open(os.path.join(output_dir,f'emo_errors.txt'), 'a') as f:
                            f.write(f'{dial_num}\n')
                            f.write(f'{gen}\n\n')
                        gen_emo = 'neutral'
                    dial_eval_dict = {'input': ex, 'gold': gold, 'generated': gen, 'gold_resp': gold_resp, 'response': response, 'gold_emo': gold_emo, 'gen_emo': gen_emo}
                
                full.append(dial_eval_dict)

            generated[dial_num.replace('.json', '').lower()] = full
            
        with open(os.path.join(output_dir, 'gen.json'), 'w') as f:
            json.dump(generated, f, indent=2)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='data/lm_data/txt_data/emo_prev', help='path to training data directory')
    parser.add_argument('--eval_data_json', type=str, default='data/lm_data/emo_prev.json', help='path to evaluation data json file')
    parser.add_argument('--eval_split', type=str, default='test', help='split to use: valid or test')
    parser.add_argument('--seed', type=int, help="Seed", default=42)
    parser.add_argument('--lr', type=float, help="Learning rate", default=3e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default="./llama2_cache")
    # parser.add_argument('--block_size', type=int, default=1024)  
    # parser.add_argument("--gen_prev", action='store_true')  
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "offline"
    # os.environ["WANDB_DISABLED"] = "true"
    set_seed(args.seed)
    logging.set_verbosity_info()

    model_name = args.model_name.split('/')[-1]

    # Load model
    base_model, tokenizer = load_model(args.model_name)

    lora_config = LoraConfig(
        r=args.rank, # matrix dim
        lora_alpha=32, # alpha scaling
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=args.model_name,
        modules_to_save=['lm_head', 'embed_tokens'],
    )

    peft_model = get_peft_model(base_model, lora_config)

    # Load dataset
    dataset = load_dataset('text', 
                data_dir=args.train_data_dir,
                split=['train[:]', 'validation[:]'] ############################################
                )
    dataset = DatasetDict({'train': dataset[0], 'validation':dataset[1]})
    max_length = peft_model.config.max_position_embeddings
    dataset = preprocess_dataset(dataset, tokenizer, max_length)

    # training
    training_set = args.train_data_dir.split('/')[-1]
    output_dir = f"OUT_{model_name}/training_outputs/{training_set}/{args.lr}_{args.seed}_rank{args.rank}/"
    os.makedirs(output_dir, exist_ok=True)

    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable()
    peft_model.config.use_cache = False

    trainer = prep_trainer(output_dir, dataset, tokenizer, peft_model)
   
    trainer.train()


    # inference
    model = trainer.model
    model.config.use_cache = True
    model.merge_and_unload(progressbar=True)
    model.eval()
    tokenizer.padding_side = "left"

    # load eval data
    
    eval_set = args.eval_data_json.split('/')[-1].split('.')[0]

    if eval_set == 'emo_prev':
        for eval_set in ['emo_gen', 'emo_gold']:
            output_dir = f"OUT_{model_name}/gen_outputs/{eval_set}_{args.eval_split}/{args.lr}_{args.seed}_rank{args.rank}"
            generate(output_dir)
    else:
        output_dir = f"OUT_{model_name}/gen_outputs/{eval_set}_{args.eval_split}/{args.lr}_{args.seed}_rank{args.rank}"
        generate(output_dir)
    
    
    