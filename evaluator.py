"""
   MTTOD: evaluator.py

   Evaluate MultiWoZ Performance.

   This code is referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/eval.py)

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
import math
import argparse
import logging

from types import SimpleNamespace
from collections import Counter, OrderedDict

from nltk.util import ngrams

from config import CONFIGURATION_FILE_NAME
from reader import MultiWOZReader

from utils import definitions
from utils.io_utils import get_or_create_logger, load_json
from utils.clean_dataset import clean_slot_values


logger = get_or_create_logger(__name__)


class BLEUScorer:
    """
    BLEU score calculator via GentScorer interface
    it calculates the BLEU-4 by taking the entire corpus in
    Calulate based multiple candidates against multiple references
    """
    def __init__(self):
        pass

    def score(self, parallel_corpus):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(
                                max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng]))
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0
                for i in range(4)]
        s = math.fsum(w * math.log(p_n)
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


class MultiWozEvaluator(object):
    def __init__(self, reader, eval_data_type="test"):
        self.reader = reader
        self.all_domains = definitions.ALL_DOMAINS[self.reader.dataset]

        self.gold_data = load_json(os.path.join(
            self.reader.data_dir, "{}_data.json".format(eval_data_type)))

        self.eval_data_type = eval_data_type

        self.bleu_scorer = BLEUScorer()
        
        ont = {
            "informable": { 
                "area" : ["centre","north","west","south","east"],
                "food" : ["afghan","african","afternoon tea","asian oriental","australasian","australian","austrian","barbeque","basque","belgian","bistro","brazilian","british","canapes","cantonese","caribbean","catalan","chinese","christmas","corsica","creative","crossover","cuban","danish","eastern european","english","eritrean","european","french","fusion","gastropub","german","greek","halal","hungarian","indian","indonesian","international","irish","italian","jamaican","japanese","korean","kosher","latin american","lebanese","light bites","malaysian","mediterranean","mexican","middle eastern","modern american","modern eclectic","modern european","modern global","molecular gastronomy","moroccan","new zealand","north african","north american","north indian","northern european","panasian","persian","polish","polynesian","portuguese","romanian","russian","scandinavian","scottish","seafood","singaporean","south african","south indian","spanish","sri lankan","steakhouse","swedish","swiss","thai","the americas","traditional","turkish","tuscan","unusual","vegetarian","venetian","vietnamese","welsh","world"],
                "pricerange" : ["cheap","moderate","expensive"]
            }
        }
        
        
        self.entities_flat = []
        self.entitiy_to_slot_dict = {}
        for s,v in ont['informable'].items():
            self.entities_flat.extend(v)
            for v1 in v:
                self.entitiy_to_slot_dict[v1] = s

        self.all_info_slot = []
        for d, s_list in definitions.INFORMABLE_SLOTS[self.reader.dataset].items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)

        # only evaluate these slots for dialog success
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']
        self.requestable_slots = ['address', 'name', 'phone', 'postcode', 'food', 'area', 'pricerange']

    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [], []
        for dial_id, dial in data.items():
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            for turn in dial:
                # excepoch <bos_resp>, <eos_resp>
                gen.append(" ".join(turn['resp_gen'].split()[1:-1]))
                truth.append(" ".join(turn['redx'].split()[1:-1]))

        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc

    def bleu_metric_cam(self, data, type='bleu'):
        # def clean(s):
        #     s = s.replace('<go_r> ', '')
        #     s = '<GO> ' + s
        #     return s

        gen, truth = [], []
        for row in data:
            gen.append(self.clean(row['resp_gen']))
            # gen.append(self.clean(row['resp']))
            truth.append(self.clean(row['redx']))
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        return sc
    
    def value_similar(self, a, b):
        return True if a == b else False

        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True
        return False

    def _bspn_to_dict(self, bspn, no_name=False, no_book=False):
        constraint_dict = self.reader.bspn_to_constraint_dict(bspn)

        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s, v in cons.items():
                key = domain+'-'+s
                if no_name and s == 'name':
                    continue
                if no_book:
                    if s in ['people', 'stay'] or \
                       key in ['hotel-day', 'restaurant-day', 'restaurant-time']:
                        continue
                constraint_dict_flat[key] = v

        return constraint_dict_flat

    def _constraint_compare(self, truth_cons, gen_cons,
                            slot_appear_num=None, slot_correct_num=None):
        tp, fp, fn = 0, 0, 0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            # v_truth = truth_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(
                        slot) else slot_correct_num.get(slot)+1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(
                    slot) else slot_appear_num.get(slot)+1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp, fp, fn, acc, list(set(false_slot))

    def dialog_state_tracking_eval(self, dials,
                                   eval_dial_list=None, no_name=False,
                                   no_book=False, add_auxiliary_task=False):
        total_turn, joint_match = 0, 0
        total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0
        slot_appear_num, slot_correct_num = {}, {}
        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            missed_jg_turn_id = []
            for turn_num, turn in enumerate(dial):
                bspn_gen = turn["bspn_gen_with_span"] if add_auxiliary_task else turn["bspn_gen"]

                gen_cons = self._bspn_to_dict(
                    turn['bspn_gen'], no_name=no_name, no_book=no_book)
                truth_cons = self._bspn_to_dict(
                    turn['bspn'], no_name=no_name, no_book=no_book)

                if truth_cons == gen_cons:
                    joint_match += 1
                else:
                    missed_jg_turn_id.append(str(turn['turn_num']))

                if eval_dial_list is None:
                    tp, fp, fn, acc, false_slots = self._constraint_compare(
                        truth_cons, gen_cons, slot_appear_num, slot_correct_num)
                else:
                    tp, fp, fn, acc, false_slots = self._constraint_compare(
                        truth_cons, gen_cons,)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_acc += acc
                total_turn += 1
                if not no_name and not no_book:
                    turn['wrong_inform'] = '; '.join(false_slots)   # turn inform metric record

            # dialog inform metric record
            if not no_name and not no_book:
                dial[0]['wrong_inform'] = ' '.join(missed_jg_turn_id)

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        accuracy = total_acc / \
            (total_turn * len(self.all_info_slot) + 1e-10) * 100
        joint_goal = joint_match / (total_turn+1e-10) * 100

        return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num

    def aspn_eval(self, dials, eval_dial_list=None):
        def _get_tp_fp_fn(label_list, pred_list):
            tp = len([t for t in pred_list if t in label_list])
            fp = max(0, len(pred_list) - tp)
            fn = max(0, len(label_list) - tp)
            return tp, fp, fn

        total_tp, total_fp, total_fn = 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_act = []
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                if cfg.same_eval_act_f1_as_hdsa:
                    pred_acts, true_acts = {}, {}
                    for t in turn['aspn_gen']:
                        pred_acts[t] = 1
                    for t in  turn['aspn']:
                        true_acts[t] = 1
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                else:
                    pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
                    true_acts = self.reader.aspan_to_act_list(turn['aspn'])
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                if fp + fn !=0:
                    wrong_act.append(str(turn['turn_num']))
                    turn['wrong_act'] = 'x'

                total_tp += tp
                total_fp += fp
                total_fn += fn

            dial[0]['wrong_act'] = ' '.join(wrong_act)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return f1 * 100

    def context_to_response_eval(self, dials, eval_dial_list=None, add_auxiliary_task=False):
        counts = {}
        for req in self.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}

            for domain in self.all_domains:
                if self.gold_data[dial_id]['goal'].get(domain):
                    true_goal = self.gold_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']

            # print('\n',dial_id)
            success, match, stats, counts = self._evaluateGeneratedDialogue(
                dial, goal, reqs, counts, add_auxiliary_task=add_auxiliary_task)
            '''
            if success == 0 or match == 0:
                print("success ", success, "; match ", match)
                print(goal)
                for turn in dial:
                    print("=" * 50 + " " + str(dial_id) + " " + "=" * 50)
                    print("user               | ", turn["user"])
                    print("-" * 50 + " " + str(turn["turn_num"]) + " " + "-" * 50)
                    print("bspn               | ", turn["bspn"])
                    print("bspn_gen           | ", turn["bspn_gen"])
                    if "bspn_gen_with_span" in turn:
                        print("bspn_gen_with_span | ", turn["bspn_gen_with_span"])
                    print("-" * 100)
                    print("resp               | ", turn["redx"])
                    print("resp_gen           | ", turn["resp_gen"])
                    print("=" * 100)

                input()
            '''
            successes += success
            matches += match
            dial_num += 1

            # for domain in gen_stats.keys():
            #     gen_stats[domain][0] += stats[domain][0]
            #     gen_stats[domain][1] += stats[domain][1]
            #     gen_stats[domain][2] += stats[domain][2]

            # if 'SNG' in filename:
            #     for domain in gen_stats.keys():
            #         sng_gen_stats[domain][0] += stats[domain][0]
            #         sng_gen_stats[domain][1] += stats[domain][1]
            #         sng_gen_stats[domain][2] += stats[domain][2]

        # self.logger.info(report)
        succ_rate = successes/(float(dial_num) + 1e-10) * 100
        match_rate = matches/(float(dial_num) + 1e-10) * 100

        return succ_rate, match_rate, counts, dial_num

    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts,
                                   soft_acc=False, add_auxiliary_task=False):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
        #'id'
        requestables = self.requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0:
                continue

            sent_t = turn['resp_gen']
            # sent_t = turn['resp']
            for domain in goal.keys():
                # for computing success
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if add_auxiliary_task:
                            bspn = turn['bspn_gen_with_span']
                        else:
                            bspn = turn['bspn_gen']

                        # bspn = turn['bspn']

                        constraint_dict = self.reader.bspn_to_constraint_dict(
                            bspn)
                        if constraint_dict.get(domain):
                            venues = self.reader.db.queryJsons(
                                domain, constraint_dict[domain], return_name=True)
                        else:
                            venues = []

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            # venue_offered[domain] = random.sample(venues, 1)
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            # flag = False
                            # for ven in venues:
                            #     if venue_offered[domain][0] == ven:
                            #         flag = True
                            #         break
                            # if not flag and venues:
                            flag = False
                            for ven in venues:
                                if ven not in venue_offered[domain]:
                                    # if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if flag and venues:
                            if flag and venues:
                                # sometimes there are no results so sample won't work
                                # print venues
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if '[value_reference]' in sent_t:
                            # if pointer was allowing for that?
                            if 'booked' in turn['pointer'] or 'ok' in turn['pointer']:
                                provided_requestables[domain].append(
                                    'reference')
                            # provided_requestables[domain].append('reference')
                    else:
                        if '[value_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0],
                'hotel': [0, 0, 0],
                'attraction': [0, 0, 0],
                'train': [0, 0, 0],
                'taxi': [0, 0, 0],
                'hospital': [0, 0, 0],
                'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.reader.db.queryJsons(
                    domain, goal[domain]['informable'], return_name=True)
                if type(venue_offered[domain]) is str and \
                   '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and \
                     len(set(venue_offered[domain]) & set(goal_venues))>0:
                    match += 1
                    match_stat = 1
            else:
                if '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request+'_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request+'_offer'] += 1

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                # for request in set(provided_requestables[domain]):
                #     if request in real_requestables[domain]:
                #         domain_success += 1
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1

                # if domain_success >= len(real_requestables[domain]):
                if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats, counts

    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in true_goal[domain]:
                    if 'id' in true_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in true_goal[domain]:
                    for reqs in true_goal[domain]['reqt']:  # addtional requests:
                        if reqs in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(reqs)
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append("reference")

            for s, v in true_goal[domain]['info'].items():
                s_, v_ = clean_slot_values(domain, s, v)
                if len(v_.split()) >1:
                    v_ = ' '.join(
                        [token.text for token in self.reader.nlp(v_)]).strip()
                goal[domain]["informable"][s_] = v_

            if 'book' in true_goal[domain]:
                goal[domain]["booking"] = true_goal[domain]['book']

        return goal

    def run_metrics(self, data, domain="all", file_list=None):
        metric_result = {'domain': domain}

        bleu = self.bleu_metric(data, file_list)

        jg, slot_f1, slot_acc, slot_cnt, slot_corr = self.dialog_state_tracking_eval(
            data, file_list)

        metric_result.update(
            {'joint_goal': jg, 'slot_acc': slot_acc, 'slot_f1': slot_f1})

        info_slots_acc = {}
        for slot in slot_cnt:
            correct = slot_corr.get(slot, 0)
            info_slots_acc[slot] = correct / slot_cnt[slot] * 100
        info_slots_acc = OrderedDict(sorted(info_slots_acc.items(), key=lambda x: x[1]))

        act_f1 = self.aspn_eval(data, file_list)

        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, file_list)

        req_slots_acc = {}
        for req in self.requestables:
            acc = req_offer_counts[req+'_offer']/(req_offer_counts[req+'_total'] + 1e-10)
            req_slots_acc[req] = acc * 100
        req_slots_acc = OrderedDict(sorted(req_slots_acc.items(), key = lambda x: x[1]))

        if dial_num:
            metric_result.update({'act_f1': act_f1,
                'success': success,
                'match': match,
                'bleu': bleu,
                'req_slots_acc': req_slots_acc,
                'info_slots_acc': info_slots_acc,
                'dial_num': dial_num})

            logging.info('[DST] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f',
                         jg, slot_acc, slot_f1, act_f1)
            logging.info('[CTR] match: %2.1f  success: %2.1f  bleu: %2.1f',
                         match, success, bleu)
            logging.info('[CTR] ' + '; '
                         .join(['%s: %2.1f' %(req, acc) for req, acc in req_slots_acc.items()]))

            return metric_result
        else:
            return None

    def e2e_eval(self, data, eval_dial_list=None, add_auxiliary_task=False):
        bleu = self.bleu_metric(data)
        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, eval_dial_list=eval_dial_list, add_auxiliary_task=add_auxiliary_task)

        return bleu, success, match

    def match_metric(self, data):
        dials = data
        match, total = 0, 1e-8
        for dial_id in dials:
            dial = dials[dial_id]
            truth_cons, gen_cons = {'1': '', '2': '', '3': ''}, None
            for turn_num, turn in enumerate(dial):
                # find the last turn which the system provide an entity
                if '[value' in turn['resp_gen']:
                    gen_cons = self._normalize_constraint(turn['bspn_gen'], ignore_dontcare=True)
                if '[value' in turn['redx']:
                    truth_cons = self._normalize_constraint(turn['bspn'], ignore_dontcare=True)
            if not gen_cons:
                # if no entity is provided, choose the state of the last dialog turn
                gen_cons = self._normalize_constraint(dial[-1]['bspn_gen'], ignore_dontcare=True)
            if list(truth_cons.values()) != ['', '', '']:
                if gen_cons == truth_cons:
                    match += 1
                total += 1

        return match / total
    
    def _normalize_constraint(self, constraint, ignore_dontcare=False, intersection=True):
        """
        Normalize belief span, e.g. delete repeated words
        :param constraint - {'food': 'asian oritental', 'pricerange': 'cheap'}
        :param intersection: if true, only keeps the words that appear in th ontology
                                        we set intersection=True as in previous works
        :returns: normalized constraint dict
                      e.g. - {'food': 'asian oritental', 'pricerange': 'cheap', 'area': ''}
        """
        results = self.reader.bspn_to_constraint_dict(constraint)['restaurant']
        
        constraint = {}
        for s in self.all_info_slot:
            if s.split('-')[-1] in results:
                constraint[s] = results[s.split('-')[-1]]
        # constraint = self.reader.bspn_to_constraint_dict(constraint)
        
        # print(constraint)
        normalized = {}
        for s in self.all_info_slot:
            normalized[s] = ''
        for s, v in constraint.items():
            if ignore_dontcare and v == 'dontcare':
                continue
            if intersection and v != 'dontcare' and v not in self.entities_flat:
                continue

            normalized[s] = v

        return normalized
    
    def tracker_metric(self, data, normalize=True):
        # turn level metric
        tp, fp, fn, db_correct = 0, 0, 0, 0
        goal_accr, slot_accr, total = 0, {}, 1e-8
        for s in self.all_info_slot:
            slot_accr[s] = 0

        for k,v in data.items():
            for row in v:
        # for row in data:
                if normalize:
                    gen = self._normalize_constraint(row['bspn_gen'])
                    truth = self._normalize_constraint(row['bspn'])
                else:
                    gen = self._normalize_constraint(row['bspn_gen'], intersection=False)
                    truth = self._normalize_constraint(row['bspn'], intersection=False)
                valid = 'thank' not in row['user'] and 'bye' not in row['user']
                if valid:
                    for slot, value in gen.items():
                        if value in truth[slot]:
                            tp += 1
                        else:
                            fp += 1
                    for slot, value in truth.items():
                        if value not in gen[slot]:
                            fn += 1

                if truth and valid:
                    total += 1
                    for s in self.all_info_slot:
                        if gen[s] == truth[s]:
                            slot_accr[s] += 1
                    if gen == truth:
                        goal_accr += 1
                    if row.get('dbpn_gen') and row.get('db_match'):
                        if row['dbpn_gen'] == row['db_match']:
                            db_correct += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        goal_accr /= total
        db_correct /= total
        for s in slot_accr:
            slot_accr[s] /= total
        return precision, recall, f1, goal_accr, slot_accr, db_correct
    
    def clean_replace(self, s, r, t, forward=True, backward=False):
        def clean_replace_single(s, r, t, forward, backward, sidx=0):
            # idx = s[sidx:].find(r)
            idx = s.find(r)
            if idx == -1:
                return s, -1
            idx_r = idx + len(r)
            if backward:
                while idx > 0 and s[idx - 1]:
                    idx -= 1
            elif idx > 0 and s[idx - 1] != ' ':
                return s, -1

            if forward:
                while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                    idx_r += 1
            elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                return s, -1
            return s[:idx] + t + s[idx_r:], idx_r

        # source, replace, target = s, r, t
        # count = 0
        sidx = 0
        while sidx != -1:
            s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
            # count += 1
            # print(s, sidx)
            # if count == 20:
            #     print(source, '\n', replace, '\n', target)
            #     quit()
        return s
    
    def clean(self, resp):
        # we  use the same clean process as in Sequicity, SEDST, FSDM
        # to ensure comparable results
        
        resp = ' '.join(resp.split()[1:-1])
        # resp = resp.replace(f' {self.reader.eos_r_token}', '')
        # resp = f'{self.reader.sos_r_token} {resp} {self.reader.eos_r_token}'
        for value, slot in self.entitiy_to_slot_dict.items():
            # s = s.replace(item, 'VALUE_{}'.format(self.entity_dict[item]))
            resp = self.clean_replace(resp, value, '[value_%s]' % slot)
        return resp
    
    def request_metric(self, data):
        # dialog level metric
        dials = data
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            truth_req, gen_req = set(), set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                resp_gen_token = self.clean(turn['resp_gen']).split()
                resp_token = self.clean(turn['redx']).split()
                for w in resp_gen_token:
                    if '[value_' in w and w.endswith(']') and w != '[value_name]':
                        gen_req.add(w[1:-1].split('_')[1])
                for w in resp_token:
                    if '[value_' in w and w.endswith(']') and w != '[value_name]':
                        truth_req.add(w[1:-1].split('_')[1])
            # print(dial_id)
            # print('gen_req:', gen_req)
            # print('truth_req:', truth_req)
            # print('')
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        # print('precision:', precision, 'recall:', recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall
    
    def e2e_eval_cam(self, data, eval_data_list=None):
        metrics = {}
        
        data_for_bleu = []
        for k,v in data.items():
            for t in v:
                data_for_bleu.append(t)
        
        bleu = self.bleu_metric_cam(data_for_bleu)
        p,r,f1,goal_acc,slot_acc, db_acc = self.tracker_metric(data)
        match = self.match_metric(data)
        req_f1, req_p, req_r = self.request_metric(data)

        metrics['bleu'] = bleu
        metrics['match'] = match
        metrics['req_f1'] = req_f1
        metrics['joint_goal'] = goal_acc
        metrics['slot_accu'] = slot_acc
        metrics['slot-p/r/f1'] = (p, r, f1)
        metrics['db_acc'] = db_acc

        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for evaluation")

    parser.add_argument("-data", type=str, required=True)
    parser.add_argument("-data_type", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("-excluded_domains", type=str, nargs="+")

    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(args.data)), CONFIGURATION_FILE_NAME)

    cfg = SimpleNamespace(**load_json(cfg_path))

    data = load_json(args.data)

    dial_by_domain = load_json("data/MultiWOZ_2.1/dial_by_domain.json")

    eval_dial_list = None
    if args.excluded_domains is not None:
        eval_dial_list = []
        for domains, dial_ids in dial_by_domain.items():
            domain_list = domains.split("-")

            if len(set(domain_list) & set(args.excluded_domains)) == 0:
                eval_dial_list.extend(dial_ids)

    reader = MultiWOZReader(cfg.backbone, cfg.version)

    evaluator = MultiWozEvaluator(reader, args.data_type)

    if cfg.task == "e2e":
        bleu, success, match = evaluator.e2e_eval(
            data, eval_dial_list=eval_dial_list, add_auxiliary_task=cfg.add_auxiliary_task)

        score = 0.5 * (success + match) + bleu

        logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f',
            match, success, bleu, score)
    else:
        joint_goal, f1, accuracy, _, _ = evaluator.dialog_state_tracking_eval(
            data, eval_dial_list=eval_dial_list, add_auxiliary_task=cfg.add_auxiliary_task)

        logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;',
            joint_goal, accuracy, f1)
