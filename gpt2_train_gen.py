import os
import argparse
import json
from functools import partial
from itertools import chain
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
import torch


def tokenize_function(batch):
        return tokenizer(batch['text'])

def group_texts(examples):
        """
        Merge examples in a batch to make examples of 1024 (model_max_length)
        This is why <|endoftext|> is added to the start + end of each example => separate examples
        Also means that last tokens in the dataset will be dropped if total length is not a multiple of block_size
        """
        block_size = args.block_size
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} # flatten token lists in batch 
        total_length = len(concatenated_examples[list(examples.keys())[0]]) # get total length of all tokenized sequences combined
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size # make total length a multiple of block_size
        result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        result["labels"] = result["input_ids"].copy() # labels == inputs for language modeling
        return result

def prep_trainer(output_dir, lm_dataset, tokenizer, model):
    
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4, # effective batch size of 32

        learning_rate=args.lr,
        warmup_steps=200, 
        weight_decay=0.01,
        fp16=True,
        lr_scheduler_type="linear", 

        logging_steps=100, 
        eval_steps=100,
        save_steps=100,
        num_train_epochs=args.epochs,

        save_total_limit=1,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        metric_for_best_model='eval_loss',

        #report_to='wandb',
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,                         
        args=training_args,                  
        train_dataset=lm_dataset['train'],         
        eval_dataset=lm_dataset['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
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



def generate(output_dir, eval_set):
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
                gen = gen.replace('<|endoftext|>', '')
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
    parser.add_argument('--lr', type=float, help="Learning rate", default=8e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--cache_dir', type=str, default="./gpt2_cache")
    parser.add_argument('--block_size', type=int, default=1024)
 
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "offline"

    set_seed(args.seed)
    logging.set_verbosity_info()

    model_name = args.model_name+'_'+str(args.block_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|belief|>', '<|endofbelief|>', '<|action|>', '<|endofaction|>', '<|response|>', '<|endofresponse|>', '<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>', '<|useremotion|>', '<|endofuseremotion|>', '[address]', '[area]','[arriveby]','[bookday]','[bookpeople]','[bookstay]','[booktime]', '[choice]','[day]','[department]','[departure]','[destination]','[duration]','[entrancefee]','[food]','[leaveat]','[name]','[openhours]','[phone]','[postcode]','[price]','[pricerange]','[ref]','[stars]','[trainid]','[type]']})

    dataset = load_dataset('text', 
                       data_dir=args.train_data_dir,
                       split=['train[:]', 'validation[:]'] ############################################
                       )
    dataset = DatasetDict({'train': dataset[0], 'validation':dataset[1]})

    dataset = dataset.map(lambda x: {'text' : x['text'] + ' ' + tokenizer.eos_token})

    lm_dataset = dataset.map(tokenize_function,
                            batched=True, # processes batches of 1000 examples
                            remove_columns=['text'],
                            desc="Running tokenizer on dataset",
                            )

    lm_dataset = lm_dataset.map(group_texts,
                                batched=True, 
                                desc=f"Grouping texts in chunks of {tokenizer.model_max_length} tokens")

    lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, device_map="auto")
    model.resize_token_embeddings(len(tokenizer)) 

    training_set = args.train_data_dir.split('/')[-1]
    output_dir = f"OUT_{model_name}/training_outputs/{training_set}/{args.lr}_{args.seed}/"
    os.makedirs(output_dir, exist_ok=True)

    trainer = prep_trainer(output_dir, lm_dataset, tokenizer, model)
   
    trainer.train()


    # inference
    model = trainer.model
    model.config.use_cache = True
    model.eval()
    tokenizer.padding_side = "left"

    # load eval data
    eval_set = args.eval_data_json.split('/')[-1].split('.')[0]

    if eval_set == 'emo_prev':
        for eval_set in ['emo_gen', 'emo_gold']: # test if inserting gold labels into context improves emo gen
            output_dir = f"OUT_{model_name}/gen_outputs/{eval_set}_{args.eval_split}/{args.lr}_{args.seed}"
            generate(output_dir, eval_set)
    else:
        output_dir = f"OUT_{model_name}/gen_outputs/{eval_set}_{args.eval_split}/{args.lr}_{args.seed}"
        generate(output_dir, eval_set)
    
    
    