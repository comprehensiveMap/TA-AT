import json
from utils import definitions
from collections import OrderedDict
file_path = 'inferenece_results/mwoz_22_few_shot_0.5_epoch12'
file_out_path = 'inferenece_results/mwoz_22_few_shot_0.5_epoch12_out.json'
file_out_path_1 = 'inferenece_results/mwoz_22_few_shot_0.5_epoch12_simple.json'
dataset = 'multiwoz_2.2'
out_dials = {}

def convert_bs_to_state(bspn):
    bspn = bspn.split() if isinstance(bspn, str) else bspn
    constraint_dict = OrderedDict()
    domain, slot = None, None
    for token in bspn:
        if token == definitions.EOS_BELIEF_TOKEN:
            break

        if token.startswith("["):
            token = token[1:-1]

            if token in definitions.ALL_DOMAINS[dataset] + ['general']:
                domain = token

            if token.startswith("value_"):
                if domain is None:
                    continue

                if domain not in constraint_dict:
                    constraint_dict[domain] = OrderedDict()

                slot = token.split("_")[1]

                constraint_dict[domain][slot] = []

        else:
            try:
                if domain is not None and slot is not None:
                    constraint_dict[domain][slot].append(token)
            except KeyError:
                continue

    for domain, sv_dict in constraint_dict.items():
        for s, value_tokens in sv_dict.items():
            constraint_dict[domain][s] = " ".join(value_tokens)

    return constraint_dict

with open(file_path) as f:
    dials = json.load(f)
    for k, dial in dials.items():
        out_dials[k.split('.')[0]] = []
        for t in dial:
            new_turn = {}
            
            # new_list = t['resp_gen'].split()[1:-1]
            # for i, w in enumerate(new_list):
            #     if w.startswith('[value_'):
            #         new_list[i] = '[' + w[7:]
            
            # new_list = [slot_mapping.get(w,w) for w in t['resp_gen'].split()[1:-1]]
            new_turn['response'] = ' '.join(t['resp_gen'].split()[1:-1])
            new_turn['active_domains'] = [s[1:-1] for s in t['turn_domain']]
            new_turn['state'] = convert_bs_to_state(' '.join(t['bspn_gen'].split()[1:-1]))
            
            out_dials[k.split('.')[0]].append(new_turn)
            
with open(file_out_path, 'w') as f:
    json.dump(out_dials, f, indent=4)
    
with open(file_path) as f:
    dials = json.load(f)
    for k, dial in dials.items():
        out_dials[k.split('.')[0]] = []
        for t in dial:
            new_turn = {}
            for k1,v in t.items():
                if 'vec' not in k1:
                    new_turn[k1] = v
        
            out_dials[k.split('.')[0]].append(new_turn)
            
with open(file_out_path_1, 'w') as f:
    json.dump(out_dials, f, indent=4)
            
    