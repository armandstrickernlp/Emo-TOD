from tqdm import tqdm
from fuzzywuzzy import fuzz
import os
"""Utils for TOD evaluation, taking into account filters from various en-to-end TOD repos."""
# https://github.com/tomyoung903/FusedChat
def slot_value_equivalence(groundtruth, prediction):
    if '|' in groundtruth:
        groundtruth_values = groundtruth.split('|')
        if prediction in groundtruth_values:
            return True
    if groundtruth == 'guest house' and prediction == 'guesthouse':
        return True
    if groundtruth == 'nightclub' and prediction == 'night club':
        return True
    if groundtruth == 'concert hall' and prediction == 'concerthall':
        return True
    if groundtruth == 'museum of archaeology and anthropology' and prediction == 'museum of archaelogy and anthropology':
        return True
    if groundtruth == 'scudamores punting co' and prediction == 'scudamores punters':
        return True
    if groundtruth == 'riverside brasserie' and prediction == 'riveride brasserie':
        return True 
    if groundtruth == 'pizza express fenditton' and prediction == 'pizza hut fenditton':
        return True 
    if groundtruth == 'the slug and lettuce' and prediction == 'slug and lettuce':
        return True
    if groundtruth == 'cafe jello gallery' and prediction == 'jello gallery':
        return True
    if groundtruth == 'alpha milton guest house' and prediction == 'alpha-milton guest house':
        return True  
    if groundtruth == 'city centre north bed and breakfast' and prediction == 'city centre north b and b':
        return True
    if groundtruth == 'portuguese' and prediction == 'portugese':
        return True
    if groundtruth == 'bishops stortford' and prediction == 'bishops strotford':
        return True
    if groundtruth == 'el shaddia guest house' and prediction == 'el shaddia guesthouse':
        return True
    if groundtruth == 'hobsons house' and prediction == 'hobson house':
        return True
    if groundtruth == 'cherry hinton water play' and prediction == 'cherry hinton water park':
        return True    
    if groundtruth == 'centre>west' and prediction == 'centre':
        return True 
    if groundtruth == 'north european' and prediction == 'north european':
        return True
    if groundtruth == 'museum of archaeology and anthropology' and prediction == 'archaelogy and anthropology':
        return True
    if groundtruth == 'riverboat georgina' and prediction == 'the riverboat georgina':
        return True
    if groundtruth == 'grafton hotel restaurant' and prediction == 'graffton hotel restaurant':
        return True
    if groundtruth == 'restaurant one seven' and prediction == 'one seven':
        return True
    if groundtruth == 'arbury lodge guest house' and prediction == 'arbury lodge guesthouse':
        return True
    if groundtruth == 'michaelhouse cafe' and prediction == 'michaelhosue cafe':
        return True
    if groundtruth == 'frankie and bennys' and prediction == "frankie and benny's":
        return True
    if groundtruth == 'london liverpool street' and prediction == 'london liverpoool':
        return True
    if groundtruth == 'the gandhi' and prediction == ' gandhi ':
        return True
    if groundtruth == 'finches bed and breakfast' and prediction == 'flinches bed and breakfast':
        return True
    if groundtruth == 'the cambridge corn exchange' and prediction == 'cambridge corn exchange':
        return True
    if groundtruth == 'broxbourne' and prediction == 'borxbourne':
        return True
    if groundtruth == 'pizza hut fen ditton' and prediction == 'pizza hut fenditton':
        return True
    if groundtruth == 'cafe jello museum' and prediction == 'cafe jello gallery':
        return True
    if fuzz.ratio(groundtruth, prediction) >= 90:
        return True
    return groundtruth == prediction

def check_state(state_dict, gold_dict):
    for domain in gold_dict:
        if domain not in state_dict:
            return False
        else:
            for slot in gold_dict[domain]:
                if slot not in state_dict[domain]:
                    return False
                elif slot_value_equivalence(state_dict[domain][slot], gold_dict[domain][slot]) == False:
                    return False
    return True

# https://github.com/Tomiinek/MultiWOZ_Evaluation
def normalize_state_slot_value(slot_name, value):
    """ Normalize slot value:
        1) replace too distant venue names with canonical values
        2) replace too distant food types with canonical values
        3) parse time strings to the HH:MM format
        4) resolve inconsistency between the database entries and parking and internet slots
    """
    
    def type_to_canonical(type_string): 
        if type_string == "swimming pool":
            return "swimmingpool" 
        elif type_string == "mutliple sports":
            return "multiple sports"
        elif type_string == "night club":
            return "nightclub"
        elif type_string == "guest house":
            return "guesthouse"
        return type_string

    def name_to_canonical(name, domain=None):
        """ Converts name to another form which is closer to the canonical form used in database. """

        name = name.strip().lower()
        name = name.replace(" & ", " and ")
        name = name.replace("&", " and ")
        name = name.replace(" '", "'")
        
        name = name.replace("bed and breakfast","b and b")
        
        if domain is None or domain == "restaurant":
            if name == "hotel du vin bistro":
                return "hotel du vin and bistro"
            elif name == "the river bar and grill":
                return "the river bar steakhouse and grill"
            elif name == "nando's":
                return "nandos"
            elif name == "city center b and b":
                return "city center north b and b"
            elif name == "acorn house":
                return "acorn guest house"
            elif name == "caffee uno":
                return "caffe uno"
            elif name == "cafe uno":
                return "caffe uno"
            elif name == "rosa's":
                return "rosas bed and breakfast"
            elif name == "restaurant called two two":
                return "restaurant two two"
            elif name == "restaurant 2 two":
                return "restaurant two two"
            elif name == "restaurant two 2":
                return "restaurant two two"
            elif name == "restaurant 2 2":
                return "restaurant two two"
            elif name == "restaurant 1 7" or name == "restaurant 17":
                return "restaurant one seven"

        if domain is None or domain == "hotel":
            if name == "lime house":
                return "limehouse"
            elif name == "cityrooms":
                return "cityroomz"
            elif name == "whale of time":
                return "whale of a time"
            elif name == "huntingdon hotel":
                return "huntingdon marriott hotel"
            elif name == "holiday inn exlpress, cambridge":
                return "express by holiday inn cambridge"
            elif name == "university hotel":
                return "university arms hotel"
            elif name == "arbury guesthouse and lodge":
                return "arbury lodge guesthouse"
            elif name == "bridge house":
                return "bridge guest house"
            elif name == "arbury guesthouse":
                return "arbury lodge guesthouse"
            elif name == "nandos in the city centre":
                return "nandos city centre"
            elif name == "a and b guest house":
                return "a and b guesthouse"
            elif name == "acorn guesthouse":
                return "acorn guest house"
            elif name == "cambridge belfry":
                return "the cambridge belfry"
            

        if domain is None or domain == "attraction":
            if name == "broughton gallery":
                return "broughton house gallery"
            elif name == "scudamores punt co":
                return "scudamores punting co"
            elif name == "cambridge botanic gardens":
                return "cambridge university botanic gardens"
            elif name == "the junction":
                return "junction theatre"
            elif name == "trinity street college":
                return "trinity college"
            elif name in ['christ college', 'christs']:
                return "christ's college"
            elif name == "history of science museum":
                return "whipple museum of the history of science"
            elif name == "parkside pools":
                return "parkside swimming pool"
            elif name == "the botanical gardens at cambridge university":
                return "cambridge university botanic gardens"
            elif name == "cafe jello museum":
                return "cafe jello gallery"
            elif name == 'pizza hut fenditton':
                return 'pizza hut fen ditton'
            elif name == 'cafe jello gallery':
                return 'cafe jello museum'
        return name

    def time_to_canonical(time):
        """ Converts time to the only format supported by database, e.g. 07:15. """
        time = time.strip().lower()

        if time == "afternoon": return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time": return "12:00"
        if time == "morning": return "08:00"
        if time.startswith("one o'clock p.m"): return "13:00"
        if time.startswith("ten o'clock a.m"): return "10:00"
        if time == "seven o'clock tomorrow evening":  return "07:00"
        if time == "three forty five p.m":  return "15:45"
        if time == "one thirty p.m.":  return "13:30"
        if time == "six fourty five":  return "06:45"
        if time == "eight thirty":  return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()    

        if time.endswith("am"):   time = time[:-2].strip()
        if time.endswith("a.m."): time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):   time = time[:-2].strip()
            if time.endswith("p.m."): time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1] 
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'
        
        if len(time) == 0:
            return "00:00"
            
        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]
            
        if time.isdigit() and len(time) == 4:
            return time[:2] + ':' + time[2:]

        if time.isdigit(): return time.zfill(2) + ":00"
        
        if ':' in time:
            time = ''.join(time.split(' '))

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time

    def food_to_canonical(food):
        """ Converts food name to caninical form used in database. """

        food = food.strip().lower()

        if food == "eriterean": return "mediterranean"
        if food == "brazilian": return "portuguese"
        if food == "sea food": return "seafood"
        if food == "portugese": return "portuguese"
        if food == "modern american": return "north american"
        if food == "americas": return "north american"
        if food == "intalian": return "italian"
        if food == "italain": return "italian"
        if food == "asian or oriental": return "asian"
        if food == "english": return "british"
        if food == "australasian": return "australian"
        if food == "gastropod": return "gastropub"
        if food == "brutish": return "british"
        if food == "bristish": return "british"
        if food == "europeon": return "european"

        return food

    if slot_name in ["name", "destination", "departure"]:
        return name_to_canonical(value)
    elif slot_name == "type":
        return type_to_canonical(value)
    elif slot_name == "food":
        return food_to_canonical(value)
    elif slot_name in ["arrive", "leave", "arriveby", "leaveat", "time"]:
        return time_to_canonical(value)
    elif slot_name in ["parking", "internet"]:
        return "yes" if value == "free" else value
    else:
        return value

# https://github.com/salesforce/simpletod
def remove_model_mismatch_and_db_data(dial_name, target_beliefs, pred_beliefs, domain):
    if domain == 'hotel':
        if domain in target_beliefs:
            if 'type' in pred_beliefs[domain] and 'type' in target_beliefs[domain]:
                if pred_beliefs[domain]['type'] != target_beliefs[domain]['type']:
                    pred_beliefs[domain]['type'] = target_beliefs[domain]['type']
            elif 'type' in pred_beliefs[domain] and 'type' not in target_beliefs[domain]:
                del pred_beliefs[domain]['type']
            if 'name' in pred_beliefs[domain] and 'name' in target_beliefs[domain]:
                if pred_beliefs[domain]['name'] != target_beliefs[domain]['name']:
                    pred_beliefs[domain]['name'] = target_beliefs[domain]['name']
            if 'name' in pred_beliefs[domain] and 'name' not in target_beliefs[domain]:
                del pred_beliefs[domain]['name']

    if 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'pizza hut fenditton':
        pred_beliefs[domain]['name'] = 'pizza hut fen ditton'
    
    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'riverside brasserie':
        pred_beliefs[domain]['food'] = "modern european"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'charlie chan':
        pred_beliefs[domain]['area'] = "centre"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'saint johns chop house':
        pred_beliefs[domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'pizza hut fen ditton':
        pred_beliefs[domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'cote':
        pred_beliefs[domain]['pricerange'] = "expensive"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'cambridge lodge restaurant':
        pred_beliefs[domain]['food'] = "european"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'cafe jello gallery':
        pred_beliefs[domain]['food'] = "peking restaurant"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'nandos':
        pred_beliefs[domain]['food'] = "portuguese"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'yippee noodle bar':
        pred_beliefs[domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'copper kettle':
        pred_beliefs[domain]['food'] = "british"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] in ['nirala', 'the nirala']:
        pred_beliefs[domain]['food'] = "indian"
    
    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'vue cinema':
        if 'type' in pred_beliefs[domain]:
            del pred_beliefs[domain]['type']

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'funky fun house':
        pred_beliefs[domain]['area'] = 'dontcare'

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'little seoul':
        pred_beliefs[domain]['name'] = 'downing college'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'byard art':
        pred_beliefs[domain]['type'] = 'museum'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'trinity college':
        pred_beliefs[domain]['type'] = 'college'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'cambridge university botanic gardens':
        pred_beliefs[domain]['area'] = 'centre'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'lovell lodge':
        pred_beliefs[domain]['parking'] = 'yes'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'whale of a time':
        pred_beliefs[domain]['type'] = 'entertainment'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'a and b guest house':
        pred_beliefs[domain]['parking'] = 'yes'  # correct name in turn_belief_pred

    if dial_name == 'mul0116' and domain == 'hotel' and 'area' in pred_beliefs[domain]:
        del pred_beliefs[domain]['area']
        
    return pred_beliefs

def get_state(state):
    state_dict = {}
    active_domains = set()
    triplets = state.split(',')
    for triplet in triplets:
        triplet = triplet.split()
        try :
            domain, slot, val = triplet[0], triplet[1], ' '.join(triplet[2:])
        except:
            continue
        active_domains.add(domain)
        if slot == 'book':
            vals = val.split()
            slot, val = vals[0], ' '.join(vals[1:])
    
        if domain not in state_dict:
            state_dict[domain] = {slot: val}
        else:
            state_dict[domain].update({slot: val})
    return state_dict, active_domains


def get_predictions_and_JGA(generated, out_dir='./'):
    my_predictions = {}

    success = 0 # JGA
    num_turns = 0 # JGA
    for idx, dial_num in enumerate(generated):
        # if idx == 3:
        #     break
        # if dial_num == 'pmul3044':
        dial = generated[dial_num]
        turns = []
        for turn in dial:
            gen = turn['generated']
            try :
                response = gen.split('<|response|>')[1].split('<|endofresponse|>')[0].strip()
            except:
                # with open(os.path.join(out_dir,'errors.txt'), 'a') as f:
                #     f.write(f'{dial_num}\n')
                #     f.write(f'{gen}\n\n')
                response = ''

            # state prediction + domain prediction
            active_domains = set()
            state_dict = {}
            try:
                state = gen.split('<|belief|>')[1].split('<|endofbelief|>')[0].strip()
            except:
                state = ''
            if state != '':
                state_dict, domains = get_state(state)
                active_domains.update(domains)
        

            gold = turn['gold']
            gold_dict = {}
            gold_state = gold.split('<|belief|>')[1].split('<|endofbelief|>')[0].strip()
            if gold_state != '':
                gold_dict, _ = get_state(gold_state)

            if state_dict != {}:
                state_dict_db = state_dict
                for domain in state_dict_db:
                    state_dict_db = remove_model_mismatch_and_db_data(dial_num, gold_dict, state_dict, domain)
            else:
                state_dict_db = {}
        

            if check_state(state_dict, gold_dict):
                success += 1

            num_turns += 1

            # add predicted response and state to turns   
            turns.append({'response': response, 'state': state_dict_db, 'active_domains': list(active_domains)})

        # add predicitons for this dialogue to my_predictions
        my_predictions[dial_num] = turns

    JGA = success / num_turns
    
    return my_predictions, JGA