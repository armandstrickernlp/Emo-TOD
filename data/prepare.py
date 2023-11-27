import json
import copy
import os
import argparse
import logging

def delex_plus_emo(mwoz, emowoz):
    # delexicalize and add emotion
    mwoz_delex = copy.deepcopy(mwoz)
    for dial_num in mwoz_delex:
        for idx, (task_turn, emo_turn) in enumerate(zip(mwoz_delex[dial_num]["log"], emowoz[dial_num]["log"])):

            # add emo
            if emo_turn["emotion"] != []:
                task_turn["emotion"] = emo_turn["emotion"][3]

            # delexicalize using span info
            if idx % 2 == 0:
                continue
            text = task_turn["text"]
            delex_spans = task_turn["span_info"]
            char_diff = 0
            for span in delex_spans:
                #print(char_diff)
                act, slot, value, start, end = span
                start += char_diff
                end += char_diff
                len1 = len(text)
                text = text[:start] + f'[{slot}]' + text[end:]
                len2 = len(text)
                char_diff += len2 - len1
            task_turn["text_delex"] = text
    return mwoz_delex

def clear_new_lines(text):
    return ' '.join(text.split())

def make_simpletod(split_nums, delex, variant='emo_prev'):
    """format examples for different variants + return set of dials with emotions other than neutral or satisfied
    simple | emo | emo_prev 
    """
    example_dict = {}
    emo_map =  {0:"Neutral",
            1:"Fearful, sad",
            2:"Dissatisfied, disliking",
            3:"Apologetic",
            4:"Abusive",
            5:"Excited, happy",
            6:"Satisfied, liking"}
    emo_nums = set()
    for d_num in split_nums:
        dial = delex[d_num]
        context = "<|context|> "
        example_dict[d_num] = []
        for idx, turn in enumerate(dial['log']):
            if idx%2 == 0: 
                context += f"<|user|> {clear_new_lines(turn['text'])} " # user will always be lexicalized.
                if variant == 'emo' or variant == 'emo_prev':
                    extra = '<|useremotion|> ' + emo_map[turn['emotion']['emotion']].lower() + ' <|endofuseremotion|> '
                    if turn['emotion']['emotion'] != 0 and turn['emotion']['emotion'] != 6:  #only use dials which have at least 1 utt with an emotion other than neutral or satisfied 
                        emo_nums.add(d_num)

            else:
                # context
                example_context = context + "<|endofcontext|> "
                # belief
                belief = '<|belief|> '
                for domain in turn['metadata']:
                    constraint = turn['metadata'][domain]
                    for b in constraint['book']:
                        if b != 'booked' and constraint['book'][b] != []:
                            belief += f"{domain} book {b.lower()} {constraint['book'][b][0].lower()}, " # if multiple values are considered correct, we pick the first one
                            
                    for b in constraint['semi']:
                        if constraint['semi'][b] != 'not mentioned' and constraint['semi'][b] != []: 
                            belief += f"{domain} {b.lower()} {constraint['semi'][b][0].lower()}, " 
                belief = belief[:-2] + ' ' if belief[-2] == ',' else belief # remove last comma
                belief += '<|endofbelief|> '

                # action
                action = '<|action|> '
                turn_acts = turn['dialog_act']
                name_acts = [] 
                other_acts = []
                for act in turn_acts:        
                    for slot, _ in turn_acts[act]:
                        act = act.replace('-', ' ').lower() # Hotel-Inform => hotel inform
                        if slot == 'none':
                            other_acts.append(act)
                        elif slot.lower() == "name":
                            name_acts.append(f"{act} {slot.lower()}")
                        else:
                            other_acts.append(f"{act} {slot.lower()}")
                list_acts = name_acts + other_acts
                action += ', '.join(list_acts)
                action  = action
                action += ' <|endofaction|> '
                # response
                delex_response = f"<|response|> {clear_new_lines(turn['text_delex'])} <|endofresponse|>"

                # add example
                if variant == 'simple':
                    example_dict[d_num] += [{"input_context" : example_context, "output": belief + action + delex_response}]
                elif variant == "emo" or variant == "emo_prev":
                    example_dict[d_num] += [{"input_context" : example_context, "output": belief + extra + action + delex_response}]

                # add lexicalized resp to history
                if variant == 'emo_prev':
                    context += extra
                context += f"<|system|> {clear_new_lines(turn['text'])} " 
        
    return example_dict, emo_nums


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emowoz', type=str, default='sample.json')
    parser.add_argument('--mwoz', type=str, default='MultiWOZ_2.2.json')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.mwoz) as f:
        mwoz = json.load(f)
    with open(args.emowoz) as f:
        emowoz = json.load(f)
    
    mwoz_delex = delex_plus_emo(mwoz, emowoz)

    with open('valListFile.txt') as f:
        val_nums = f.read().splitlines()
    with open('testListFile.txt') as f:
        test_nums = f.read().splitlines()

    train_nums = [num for num in list(mwoz.keys()) if num not in val_nums and num not in test_nums]

    save_dir ='./lm_data/'
    txt_data_dir = os.path.join(save_dir, 'txt_data/')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(txt_data_dir, exist_ok=True)

    logging.info('Prepping data...')
    for var in ['simple', 'emo', 'emo_prev']:
        os.makedirs(os.path.join(txt_data_dir, var), exist_ok=True)
        output_dict = {}
        # num_dict = {}
        for split_nums, split in zip([train_nums, val_nums, test_nums], ['train', 'valid', 'test']):
            examples, emo_nums = make_simpletod(split_nums, mwoz_delex, variant=var)
            output_dict[split] = examples
            # num_dict[split] = list(emo_nums)
            
            with open(os.path.join(txt_data_dir, var, f'{split}.txt'), 'w') as f:
                for dial_num in examples:
                    for turn in examples[dial_num]:
                        f.write(turn["input_context"] + turn["output"] + '\n')

        with open(os.path.join(save_dir, f'{var}.json'), 'w') as f:
            json.dump(output_dict, f, indent=2) 

        # if var == 'emo':
        #     with open(os.path.join(save_dir, f'emo_nums.json'), 'w') as f:
        #         json.dump(num_dict, f, indent=2) 