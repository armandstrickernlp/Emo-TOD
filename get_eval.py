import os
import logging
import json
import numpy as np
from scipy.stats import ttest_rel
import sklearn.metrics as metrics
from fuzzywuzzy import fuzz
import evaluate
from tqdm import tqdm
import pprint as pp
from mwzeval.metrics import Evaluator

from tod_eval_utils import get_predictions_and_JGA

import warnings
warnings.filterwarnings('ignore')



def get_mean(res_array_seeds):
    # get mean and std
    res_mean_std = {}
    for key, val in res_array_seeds.items():
        res_mean_std[key] = (round(np.mean(val),2), round(np.std(val), 2))
    return res_mean_std

########### EMO Detection ###########
# EMO F1
def get_emo_scores(generated):
    emo2idx = {"neutral":0,
            "fearful, sad":1,
            "dissatisfied, disliking":2,
            "apologetic":3,
            "abusive":4,
            "excited, happy":5,
            "satisfied, liking":6}

    y_pred = []
    y_true = []
    target_names = list(emo2idx.keys())

    for idx, dial_num in enumerate(generated):
        dial = generated[dial_num]
        for turn in dial:
            # get gold label
            y_true.append(emo2idx[turn['gold_emo']])
            # get predicted label
            try:
                y_pred.append(emo2idx[turn['gen_emo']])
            except KeyError:
                y_pred.append(emo2idx['neutral'])

    scores = metrics.classification_report(y_true, y_pred, target_names=target_names, digits=3, output_dict=True)
    # round scores
    for emo in scores:
        if emo == 'accuracy':
            continue
        for metric in scores[emo]:
            scores[emo][metric] = round(scores[emo][metric]*100, 5)
    
    # compute macro and weighted F1 ignoring neutral F1 score
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro', labels=[1,2,3,4,5,6])
    weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted', labels=[1,2,3,4,5,6])
    scores["macro f1 (exc neutral)"] = round(macro_f1*100, 5)
    scores["weighted f1 (exc neutral)"] = round(weighted_f1*100,5)

    return scores

def get_res_emo(res_path, lr=4e-05):
    res_seeds = []
    for i in [42, 43, 44, 45, 46]:
        if "Llama" in res_path:
            with open(os.path.join(res_path, f"{lr}_{i}_rank32", "gen.json"), 'r') as f:
                res_seeds.append(get_emo_scores(json.load(f)))
        elif "gpt" in res_path:
            with open(os.path.join(res_path, f"8e-05_{i}", "gen.json"), 'r') as f:
                res_seeds.append(get_emo_scores(json.load(f)))
        
    # initialize dict of empty arrays
    res_array_seeds = {}
    for key in res_seeds[0].keys():
        if key == 'accuracy' or key == 'macro avg' or key == 'weighted avg':
            continue
        else:
            res_array_seeds[key] = []
   
    # fill arrays with f1 scores
    for res in res_seeds:
        for key, val in res.items():
            if key == 'accuracy' or key == 'macro avg' or key == 'weighted avg':
                continue
            if type(val) != dict:
                res_array_seeds[key].append(val) 
            else:
                res_array_seeds[key].append(val['f1-score'])
    return res_array_seeds

def get_stat_emo(res_path1, res_path2):
    # get statistical significance for two models
    res_array_seeds1 = get_res_emo(res_path1)
    res_array_seeds2 = get_res_emo(res_path2)
    
    signif_dict = {}
    for metric, arr in res_array_seeds1.items():
        # print(arr, res_array_seeds2[metric])
        p_val = ttest_rel(arr, res_array_seeds2[metric]).pvalue
        signif_dict[metric] = p_val < 0.05
    return signif_dict




########### TOD EVAL ###########
def get_tod_results(generated):
    predictions, JGA = get_predictions_and_JGA(generated, './errors')
    e = Evaluator(success=True, richness=True, bleu=False)
    res = e.evaluate(predictions)
    my_results = {
        "inform" : res["success"]["inform"]["total"],
        "success": res["success"]["success"]["total"],
        "CBE": res["richness"]["cond_entropy"],
        "unique_trigrams": res["richness"]["num_trigrams"],
    }
    my_results['JGA'] = JGA
    
    # BLEU scores
    full_gold = []
    full_pred = []

    for dial_num in generated:
        for turn in generated[dial_num]:
            gold_resp = turn['gold_resp']
            pred_resp = turn['response']
            full_gold.append([gold_resp])
            full_pred.append(pred_resp)

    bleu = evaluate.load("bleu") # make sure bleu is in $HOME/.cache for offline eval 

    bleu_full = bleu.compute(predictions=full_pred, references=full_gold)
    my_results['bleu'] = round(bleu_full['bleu']*100, 4)
    my_results['CBE'] = round(my_results['CBE'], 3)
    my_results['JGA'] = round(my_results['JGA']*100, 3)
    
    return my_results

def get_res_array(res_path, lr):
    res_seeds = []
    for i in [42, 43, 44, 45, 46]:
        if 'Llama' in res_path:
            with open(os.path.join(res_path, f"{lr}_{i}_rank32", "gen.json"), 'r') as f:
                res_seeds.append(get_tod_results(json.load(f)))
        if 'gpt' in res_path:
            with open(os.path.join(res_path, f"8e-05_{i}", "gen.json"), 'r') as f:
                res_seeds.append(get_tod_results(json.load(f)))
        
    # initialize dict of empty arrays
    res_array_seeds = {}
    for key in res_seeds[0].keys():
        res_array_seeds[key] = []

    # fill arrays
    for res in res_seeds:
        for key, val in res.items():
            res_array_seeds[key].append(val)
    return res_array_seeds

def get_stat_tod(res_1, res_2):
    # get statistical significance for two models
 
    signif_dict = {}
    for metric, arr in res_1.items():
        # print(arr, res_array_seeds2[metric])
        p_val = ttest_rel(arr, res_2[metric]).pvalue
        signif_dict[metric] = p_val < 0.05
    return signif_dict





if __name__ ==  "__main__":

    logging.basicConfig(level=logging.INFO)

    # replace paths with your paths to each output directory 
    prev_llama = './Llama_outputs/emo_gen_test' 
    emo_llama = './Llama_outputs/emo_test'
    simple_llama = './Llama_outputs/simple_test'


    # EMO RESULTS
    logging.info("###########EMO RESULTS###########")
    print()
    logging.info("###########prev llama###########")
    pp.pprint(get_mean(get_res_emo(prev_llama, lr=4e-5)))
    print()
    logging.info("###########emo llama###########")
    pp.pprint(get_mean(get_res_emo(emo_llama, lr=5e-5)))
    print()
    print()

    # TOD RESULTS
    logging.info("###########TOD RESULTS###########")
    print()
    logging.info("###########prev llama###########")
    pp.pprint(get_mean(get_res_array(prev_llama, lr=4e-5)))   
    print()
    logging.info("###########emo llama###########")
    pp.pprint(get_mean(get_res_array(emo_llama, lr=5e-5)))
    print()
    logging.info("###########simple llama###########")
    pp.pprint(get_mean(get_res_array(simple_llama, lr=5e-5)))



  

