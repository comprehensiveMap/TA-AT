"""
   MTTOD: reader.py

   implements MultiWoz Training/Validation Data Feeder for MTTOD.

   This code is partially referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/reader.py)

   Copyright 2021 ETRI LIRS, Yohan Lee
   Copyright 2019 Yichi Zhang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import copy
import numpy as np
import spacy
import math
import random
import difflib
from tqdm import tqdm
from difflib import get_close_matches
from itertools import chain
from collections import OrderedDict, defaultdict
from copy import deepcopy
from sklearn.metrics import f1_score

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer

from utils import definitions
from utils.io_utils import load_json, load_pickle, save_pickle, get_or_create_logger
from external_knowledges import MultiWozDB, CamRestDB

logger = get_or_create_logger(__name__)


class BaseIterator(object):
    def __init__(self, reader):
        self.reader = reader

    def bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []

            turn_bucket[turn_len].append(dial)

        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def construct_mini_batch(self, data, batch_size, num_gpus):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []

        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if (len(batch) % num_gpus) != 0:
            batch = batch[:-(len(batch) % num_gpus)]
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)

        return all_batches

    def transpose_batch(self, dial_batch):
        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

    def get_batches(self, data_type, batch_size, num_gpus, shuffle=False, num_dialogs=-1, excluded_domains=None):
        dial = self.reader.data[data_type]

        if num_dialogs > 0:
            dial = random.sample(dial, min(num_dialogs, len(dial)))

        turn_bucket = self.bucket_by_turn(dial)

        all_batches = []

        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        for k in turn_bucket:
            if data_type != "test" and (k == 1 or k >= 17):
                continue

            batches = self.construct_mini_batch(
                turn_bucket[k], batch_size, num_gpus)

            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches

        if shuffle:
            random.shuffle(all_batches)

        return all_batches, num_training_steps, num_dials, num_turns

    def flatten_dial_history(self, dial_history, len_postfix, context_size, additional_vec_history):
        if context_size > 0:
            context_size -= 1

        if context_size == 0:
            windowed_context = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
        else:
            windowed_context = dial_history

        ctx_len = sum([len(c) for c in windowed_context])

        # consider eos_token
        spare_len = self.reader.max_seq_len - len_postfix - 1
        while ctx_len >= spare_len:
            ctx_len -= len(windowed_context[0]) # 从前往后一句一句来pop
            windowed_context.pop(0)
            for _, v in additional_vec_history.items():
                v.pop(0)
        
        additional_vec_history['pos_vec'] = []
        cur_len = 0
        for t in windowed_context:
            additional_vec_history['pos_vec'].append(cur_len + t.index(self.reader.tokenizer.convert_tokens_to_ids(definitions.EOS_USER_TOKEN)))
            cur_len += len(t)
        
        
        context = list(chain(*windowed_context))

        return context, additional_vec_history

    def tensorize(self, ids):
        try:
            return torch.tensor(ids, dtype=torch.long)
        except:
            max_len = max([len(l) for l in ids])
            
            if isinstance(ids[0][0], list):
                num_entries = len(ids[0][0])
                for l in ids:
                    while len(l) < max_len:
                        l.append([-100 for _ in range(num_entries)])
            else:
                for l in ids:
                    while len(l) < max_len:
                        l.append(0)
                        
            return torch.tensor(ids, dtype=torch.long)
                    


    def get_data_iterator(self, all_batches, task, ururu, add_auxiliary_task=False, context_size=-1):
        raise NotImplementedError


class MultiWOZIterator(BaseIterator):
    def __init__(self, reader):
        super(MultiWOZIterator, self).__init__(reader)

    def get_readable_batch(self, dial_batch):
        dialogs = {}

        decoded_keys = ["user", "resp", "redx", "bspn", "aspn", "dbpn",
                        "bspn_gen", "bspn_gen_with_span",
                        "dbpn_gen", "aspn_gen", "resp_gen"]
        for dial in dial_batch:
            dial_id = dial[0]["dial_id"]

            dialogs[dial_id] = []

            for turn in dial:
                readable_turn = {}
                
                
                for k, v in turn.items():
                    if k == "dial_id":
                        continue
                    elif k in decoded_keys:
                        v = self.reader.tokenizer.decode(
                            v, clean_up_tokenization_spaces=False)
                    elif k == "pointer":
                        turn_doamin = turn["turn_domain"][-1]
                        v = self.reader.db.pointerBack(v, turn_doamin)
                    if k == "user_span" or k == "resp_span":
                        speaker = k.split("_")[0]
                        v_dict = {}
                        for domain, ss_dict in v.items():
                            v_dict[domain] = {}
                            for s, span in ss_dict.items():
                                v_dict[domain][s] = self.reader.tokenizer.decode(
                                    turn[speaker][span[0]: span[1]])
                        v = v_dict

                    readable_turn[k] = v
                
                dialogs[dial_id].append(readable_turn)

        return dialogs

    def str2dic(self, s):
        cur_d = None
        dic = {}
        for w in s.split():
            if w in definitions.DOMAIN_TOKENS:
                cur_d = w
                if cur_d not in dic:
                    dic[cur_d] = {}
            elif w[0] == '[':
                cur_act = w
                dic[cur_d][cur_act] = []
            else:
                dic[cur_d][cur_act].append(w)
                
        return dic
          
    # 能够对多级字典进行排序，从而形成一个unordered tree              
    def sort_m_dict(self, dic):
        if isinstance(dic, list):
            dic.sort()
        elif dic == {}:
            return
        else:
            for k, v in dic.items():
                self.sort_m_dict(v)    
            for key in sorted(dic):
                dic[key] = dic.pop(key)

    HASH_STR_TREE = {}
    def tree2str(self, dic):
        if isinstance(dic, list):
            return ' '.join(dic)
        elif dic == {}:
            return ''
        else:
            s = []
            for k, v in dic.items():
                s.append((k + ' ' + self.tree2str(v)).strip()) 
            return ' '.join(s)

    def get_data_iterator(self, all_batches, task, ururu, add_auxiliary_task=False, context_size=-1, cur_epoch=0, mu=10, is_training=True, with_tree=True):
        for batch_idx, dial_batch in enumerate(all_batches):
            batch_encoder_input_ids = []
            batch_slot_vec_label_ids = []
            batch_delta_vec_label_ids = []
            batch_act_vec_label_ids = []
            batch_resp_vec_label_ids = []
            batch_pos_vec_label_ids = []
            batch_belief_label_ids = []
            batch_resp_label_ids = []
            batch_aspn_pos = []

            for dial in dial_batch:
                dial_encoder_input_ids = []
                dial_slot_vec_label_ids = []
                dial_delta_vec_label_ids = []
                dial_act_vec_label_ids = []
                dial_resp_vec_label_ids = []
                dial_pos_vec_label_ids = []
                
                dial_belief_label_ids = []
                dial_resp_label_ids = []
                dial_aspn_pos = []

                dial_history = []
                
                global_additional_vec_history = {'slot_vec': [], 'delta_vec': [], 'act_vec': [], 'resp_vec': []}
                
                for turn in dial:
                    context, additional_vec_history = self.flatten_dial_history(
                        deepcopy(dial_history), len(turn["user"]), context_size, deepcopy(global_additional_vec_history))

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]
                    
                    for k in additional_vec_history.keys():
                        if k != 'pos_vec':
                            additional_vec_history[k].append(turn[k])
                            global_additional_vec_history[k].append(turn[k])
                        else:
                            additional_vec_history['pos_vec'].append(len(encoder_input_ids) - 1)

                    bspn = turn["bspn"]

                    bspn_label = bspn

                    belief_label_ids = bspn_label + [self.reader.eos_token_id]
                    '''
                    if task == "e2e":
                        resp = turn["dbpn"] + turn["aspn"] + turn["redx"]

                    else:
                        resp = turn["dbpn"] + turn["aspn"] + turn["resp"]
                    '''
                    
                    e = cur_epoch - 1 + batch_idx / len(all_batches)
                    p = mu / (mu + math.exp(e/mu))
                    
                    if  'woz' not in self.reader.dataset:
                        resp = turn["dbpn"] + turn["redx"]
                        aspn_pos = -1
                    elif not with_tree or not is_training or random.random() < p:
                        resp = turn["dbpn"] + turn["aspn"] + turn["redx"]
                        aspn_pos = -1
                    else:
                        gt = turn["act_str"]
                        dic = self.str2dic(gt)
                        # self.sort_m_dict(dic)
                        s = self.tree2str(dic)
                        scores = self.reader.matrix[self.reader.tree_vocab.index(s), :]
                        
                        if np.sum(scores) == 0:
                            resp = turn["dbpn"] + turn["aspn"] + turn["redx"]
                            aspn_pos = -1
                        else:
                            index = list(range(len(self.reader.tree_vocab)))
                            sampled_aspn = self.reader.tree_vocab[random.choices(index, weights=scores, k=1)[0]]
                            aspn_ids = self.reader.encode_text(sampled_aspn,
                                bos_token=definitions.BOS_ACTION_TOKEN,
                                eos_token=definitions.EOS_ACTION_TOKEN)
                            resp = turn["dbpn"] + aspn_ids + turn["redx"]
                            aspn_pos = resp.index(self.reader.aspn_id)
                    
                    resp_label_ids = resp + [self.reader.eos_token_id]

                    dial_encoder_input_ids.append(encoder_input_ids)
                    dial_slot_vec_label_ids.append(additional_vec_history['slot_vec']) # [1*20, 2*20, 3*20]
                    dial_delta_vec_label_ids.append(additional_vec_history['delta_vec'])
                    dial_act_vec_label_ids.append(additional_vec_history['act_vec'])
                    dial_resp_vec_label_ids.append(additional_vec_history['resp_vec'])
                    dial_pos_vec_label_ids.append(additional_vec_history['pos_vec'])
                    dial_belief_label_ids.append(belief_label_ids)
                    dial_resp_label_ids.append(resp_label_ids)
                    
                    dial_aspn_pos.append(aspn_pos)

                    if ururu:
                        if task == "dst":
                            turn_text = turn["user"] + turn["resp"]
                        else:
                            turn_text = turn["user"] + turn["redx"]
                    else:
                        if task == "dst":
                            turn_text = turn["user"] + bspn + \
                                turn["dbpn"] + turn["aspn"] + turn["resp"]
                        else:
                            turn_text = turn["user"] + bspn + \
                                turn["dbpn"] + turn["aspn"] + turn["redx"]
                                
                        if 'woz' not in self.reader.dataset:
                            turn_text = turn["user"] + bspn + \
                                turn["dbpn"] + turn["redx"]

                    dial_history.append(turn_text)
                    

                batch_encoder_input_ids.append(dial_encoder_input_ids)
                
                batch_slot_vec_label_ids.append(dial_slot_vec_label_ids) 
                batch_delta_vec_label_ids.append(dial_delta_vec_label_ids)
                batch_act_vec_label_ids.append(dial_act_vec_label_ids)
                batch_resp_vec_label_ids.append(dial_resp_vec_label_ids)
                batch_pos_vec_label_ids.append(dial_pos_vec_label_ids)
                
                batch_belief_label_ids.append(dial_belief_label_ids)
                batch_resp_label_ids.append(dial_resp_label_ids)
                batch_aspn_pos.append(dial_aspn_pos)

            # turn first
            batch_encoder_input_ids = self.transpose_batch(batch_encoder_input_ids)
            
            batch_slot_vec_label_ids = self.transpose_batch(batch_slot_vec_label_ids)
            batch_delta_vec_label_ids = self.transpose_batch(batch_delta_vec_label_ids)
            batch_act_vec_label_ids = self.transpose_batch(batch_act_vec_label_ids)
            batch_resp_vec_label_ids = self.transpose_batch(batch_resp_vec_label_ids)
            batch_pos_vec_label_ids = self.transpose_batch(batch_pos_vec_label_ids)
            
            batch_belief_label_ids = self.transpose_batch(batch_belief_label_ids)
            batch_resp_label_ids = self.transpose_batch(batch_resp_label_ids)
            batch_aspn_pos = self.transpose_batch(batch_aspn_pos)
           

            num_turns = len(batch_encoder_input_ids)

            tensor_encoder_input_ids = []
            tensor_belief_label_ids = []
            tensor_resp_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids = [
                    self.tensorize(b) for b in batch_encoder_input_ids[t]]
                tensor_belief_label_ids = [
                    self.tensorize(b) for b in batch_belief_label_ids[t]]
                tensor_resp_label_ids = [
                    self.tensorize(b) for b in batch_resp_label_ids[t]]
                tensor_slot_vec_label_ids = self.tensorize(batch_slot_vec_label_ids[t]) # [bsz, turn_num, num_class]
                tensor_delta_vec_label_ids = self.tensorize(batch_delta_vec_label_ids[t])
                tensor_act_vec_label_ids = self.tensorize(batch_act_vec_label_ids[t])
                tensor_resp_vec_label_ids = self.tensorize(batch_resp_vec_label_ids[t])
                tensor_pos_vec_label_ids = self.tensorize(batch_pos_vec_label_ids[t]) # [bsz, turn_num]
                tensor_aspn_pos = self.tensorize(batch_aspn_pos[t])
                
                tensor_encoder_input_ids = pad_sequence(tensor_encoder_input_ids,
                                                        batch_first=True,
                                                        padding_value=self.reader.pad_token_id)
                tensor_belief_label_ids = pad_sequence(tensor_belief_label_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                tensor_resp_label_ids = pad_sequence(tensor_resp_label_ids,
                                                     batch_first=True,
                                                     padding_value=self.reader.pad_token_id)

                yield tensor_encoder_input_ids, (tensor_slot_vec_label_ids, tensor_delta_vec_label_ids, 
                                                 tensor_act_vec_label_ids, tensor_resp_vec_label_ids, tensor_pos_vec_label_ids, tensor_aspn_pos,
                                                 tensor_belief_label_ids, tensor_resp_label_ids)


class BaseReader(object):
    def __init__(self, backbone, train_data_ratio):
        self.nlp = spacy.load("en_core_web_sm")

        self.tokenizer = self.init_tokenizer(backbone)
        self.train_data_ratio = train_data_ratio
        self.data_dir = self.get_data_dir()

        encoded_data_path = os.path.join(self.data_dir, "encoded_data.pkl")

        if os.path.exists(encoded_data_path):
            logger.info("Load encoded data from {}".format(encoded_data_path))

            self.data = load_pickle(encoded_data_path)

        else:
            logger.info("Encode data and save to {}".format(encoded_data_path))
            train = self.encode_data("train")
            dev = self.encode_data("dev")
            test = self.encode_data("test")

            self.data = {"train": train, "dev": dev, "test": test}

            save_pickle(self.data, encoded_data_path)

        assert self.train_data_ratio > 0
        # few-shot learning
        if self.train_data_ratio < 1.0:
            print ('Few-shot training setup.')
            few_shot_num = int(len(self.data['train']) * self.train_data_ratio) + 1
            random.shuffle(self.data['train'])
            # randomly select a subset of training data
            self.data['train'] = self.data['train'][:few_shot_num]
            print ('Number of training sessions is {}'.format(few_shot_num))

    def get_data_dir(self):
        raise NotImplementedError

    def init_tokenizer(self, backbone):
        tokenizer = T5Tokenizer.from_pretrained(backbone)

        special_tokens = []

        # add domains
        domains = definitions.ALL_DOMAINS[self.dataset]  + ['general']
        for domain in sorted(domains):
            token = "[" + domain + "]"
            special_tokens.append(token)

        # add intents
        intents = list(set(chain(*definitions.DIALOG_ACTS[self.dataset].values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        if self.dataset != 'multiwoz_2.0':
            special_tokens.extend(definitions.RESP_SPEC_TOKENS[self.dataset])
        else:
            # add slots
            slots = list(set(definitions.ALL_INFSLOT + definitions.ALL_REQSLOT))

            for slot in sorted(slots):
                token = "[value_" + slot + "]"
                special_tokens.append(token)

        special_tokens.extend(definitions.SPECIAL_TOKENS)

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text

        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]

            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]

            tokens = tokens + eos_token

        encoded_text = self.tokenizer.encode(" ".join(tokens))

        # except eos token
        if encoded_text[-1] == self.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def encode_data(self, data_type):
        raise NotImplementedError


class MultiWOZReader(BaseReader):
    def __init__(self, backbone, dataset, train_data_ratio):
        self.dataset = dataset
        
        if 'woz' in dataset:
            self.db = MultiWozDB(os.path.join(os.path.dirname(self.get_data_dir()), "db"), self.dataset)
        else:
            print(os.path.join(os.path.dirname(self.get_data_dir()), "db"))
            self.db = CamRestDB(os.path.join(os.path.dirname(self.get_data_dir()), "db"), self.dataset)
        
        self.tree_vocab = load_json(os.path.join(os.path.dirname(self.get_data_dir()), 'Tree.vocab'))
        
        matrix_path = os.path.join(os.path.dirname(self.get_data_dir()), 'matrix.npy')
        fp = np.memmap(matrix_path, dtype='float32', mode='r+')
        assert len(fp.shape) == 1
        num = int(np.sqrt(fp.shape[0]))
        self.matrix = fp.reshape(num, num)
        for i in range(num):
            self.matrix[i][i] = 0.0
        
        super(MultiWOZReader, self).__init__(backbone, train_data_ratio)
        self.aspn_id = self.tokenizer.convert_tokens_to_ids(definitions.EOS_ACTION_TOKEN)

    def get_data_dir(self):
        return os.path.join(
            "data", "{}".format(self.dataset), "processed")

    # self.compare_constraint_dict(prev_constrain_dict, ordered_constraint_dict)
    def compare_constraint_dict(self, prev, curr):
        # 0 不变 1 删除 2 增添 3 修改
        diffs = [0 for _ in range(len(definitions.SLOT_TYPES[self.dataset]))]
        for i, s in enumerate(definitions.SLOT_TYPES[self.dataset]):
            domain, slot = s.split('-')
            if (domain in prev and slot in prev[domain]) and (domain not in curr or slot not in curr[domain]):
                diffs[i] = 1
            elif (domain not in prev or slot not in prev[domain]) and (domain in curr and slot in curr[domain]):
                diffs[i] = 2
            elif (domain in prev and slot in prev[domain]) and (domain in curr and slot in curr[domain]) and prev[domain][slot] != curr[domain][slot]:
                diffs[i] = 3
        
        return diffs
    
    def get_pred_act(self, act_str):
        
        act_vec = [0 for _ in range(len(definitions.ACT_TYPES[self.dataset]))]
        for w in act_str.split():
            if w[1:-1] in (definitions.ALL_DOMAINS[self.dataset] + ['general']):
                cur_domain = w[1:-1]
            elif w[0] == '[' and w[-1] == ']':
                cur_act = w[1:-1]
                if (cur_domain + '-' + cur_act) in definitions.ACT_TYPES[self.dataset]:
                    act_vec[definitions.ACT_TYPES[self.dataset].index(cur_domain + '-' + cur_act)] = 1
        
        return act_vec
    
    def get_pred_resp_token(self, redx):
        
        resp_vec = [0 for _ in range(len(definitions.RESP_SPEC_TOKENS[self.dataset]))]
        for w in redx.split():
            if w in definitions.RESP_SPEC_TOKENS[self.dataset]:
                resp_vec[definitions.RESP_SPEC_TOKENS[self.dataset].index(w)] = 1
                
        return resp_vec
    
    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, "{}_data.json".format(data_type)))

        encoded_data = []
        for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):
            encoded_dial = []

            accum_constraint_dict = {}
            for t in dial["log"]:
                turn_constrain_dict = self.bspn_to_constraint_dict(t["constraint"])
                for domain, sv_dict in turn_constrain_dict.items():
                    if domain not in accum_constraint_dict:
                        accum_constraint_dict[domain] = {}

                    for s, v in sv_dict.items():
                        if s not in accum_constraint_dict[domain]:
                            accum_constraint_dict[domain][s] = []

                        accum_constraint_dict[domain][s].append(v)

            prev_bspn = ""
            prev_constrain_dict = {}
            
            for idx, t in enumerate(dial["log"]):
                enc = {}
                enc["dial_id"] = fn
                enc["turn_num"] = t["turn_num"]
                enc["turn_domain"] = t["turn_domain"].split()
                
                if 'pointer' in t:
                    enc["pointer"] = [int(i) for i in t["pointer"].split(",")]

                target_domain = enc["turn_domain"][0] if len(enc["turn_domain"]) == 1 else enc["turn_domain"][1]

                target_domain = target_domain[1:-1]

                user_ids = self.encode_text(t["user"],
                                            bos_token=definitions.BOS_USER_TOKEN,
                                            eos_token=definitions.EOS_USER_TOKEN)

                enc["user"] = user_ids

                redx_ids = self.encode_text(t["resp"],
                                            bos_token=definitions.BOS_RESP_TOKEN,
                                            eos_token=definitions.EOS_RESP_TOKEN)

                enc["redx"] = redx_ids

                enc['slot_vec'] = [0 for _ in range(len(definitions.SLOT_TYPES[self.dataset]))]
                constraint_dict = self.bspn_to_constraint_dict(t["constraint"])
                ordered_constraint_dict = OrderedDict()
                for domain, slots in definitions.INFORMABLE_SLOTS[self.dataset].items():
                    if domain not in constraint_dict:
                        continue

                    ordered_constraint_dict[domain] = OrderedDict()
                    for slot in slots:
                        if slot not in constraint_dict[domain]:
                            continue

                        value = constraint_dict[domain][slot]

                        ordered_constraint_dict[domain][slot] = value
                        enc['slot_vec'][definitions.SLOT_TYPES[self.dataset].index(domain + '-' + slot)] = 1

                # enc['pred_slot'] = self.get_pred_slot(ordered_constraint_dict)
                enc['delta_vec'] = self.compare_constraint_dict(prev_constrain_dict, ordered_constraint_dict)
                enc['act_vec'] = self.get_pred_act(t["sys_act"])
                enc['resp_vec'] = self.get_pred_resp_token(t["resp"])
                
                ordered_bspn = self.constraint_dict_to_bspn(ordered_constraint_dict)

                bspn_ids = self.encode_text(ordered_bspn,
                                            bos_token=definitions.BOS_BELIEF_TOKEN,
                                            eos_token=definitions.EOS_BELIEF_TOKEN)

                enc["bspn"] = bspn_ids
                
                enc['act_str'] = t["sys_act"]

                aspn_ids = self.encode_text(t["sys_act"],
                                            bos_token=definitions.BOS_ACTION_TOKEN,
                                            eos_token=definitions.EOS_ACTION_TOKEN)

                enc["aspn"] = aspn_ids

                if 'pointer' in enc:
                    pointer = enc["pointer"][:-2]
                    if not any(pointer):
                        db_token = definitions.DB_NULL_TOKEN
                    else:
                        db_token = "[db_{}]".format(pointer.index(1))
                else:
                    # enc['db_match'] = t['match']
                    db_index = int(t['match'])
                    if db_index >= self.db.db_vec_size:
                        db_index = self.db.db_vec_size - 1
                    db_token = '[db_%s]' % db_index
                    assert db_token in definitions.DB_STATE_TOKENS

                dbpn_ids = self.encode_text(db_token,
                                            bos_token=definitions.BOS_DB_TOKEN,
                                            eos_token=definitions.EOS_DB_TOKEN)

                enc["dbpn"] = dbpn_ids

                if (len(enc["user"]) == 0 or
                        len(enc["redx"]) == 0 or len(enc["bspn"]) == 0 or
                        len(enc["aspn"]) == 0 or len(enc["dbpn"]) == 0):
                    raise ValueError(fn, idx)

                encoded_dial.append(enc)
                
                prev_constrain_dict = ordered_constraint_dict

            encoded_data.append(encoded_dial)

        return encoded_data

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        constraint_dict = OrderedDict()
        domain, slot = None, None
        for token in bspn:
            if token == definitions.EOS_BELIEF_TOKEN:
                break

            if token.startswith("["):
                token = token[1:-1]

                if token in definitions.ALL_DOMAINS[self.dataset] + ['general']:
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

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)


    def bspn_to_db_pointer(self, bspn, turn_domain):
        constraint_dict = self.bspn_to_constraint_dict(bspn)

        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith("[") else match_dom
        match = matnums[match_dom]

        vector = self.db.addDBIndicator(match_dom, match)

        return vector

    def canonicalize_span_value(self, domain, slot, value, cutoff=0.6):
        ontology = self.db.extractive_ontology

        if domain not in ontology or slot not in ontology[domain]:
            return value

        candidates = ontology[domain][slot]

        matches = get_close_matches(value, candidates, n=1, cutoff=cutoff)

        if len(matches) == 0:
            return value
        else:
            return matches[0]
