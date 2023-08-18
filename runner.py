"""
   MTTOD: runner.py

   implements train and predict function for MTTOD model.

   Copyright 2021 ETRI LIRS, Yohan Lee

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
import re
import copy
import math
import time
import glob
import shutil
from abc import *
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import BaseModelOutput
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
import numpy as np

from model import T5WithSpan, T5WithTokenSpan
from reader import MultiWOZIterator, MultiWOZReader
from evaluator import MultiWozEvaluator

from utils import definitions
from utils.io_utils import get_or_create_logger, load_json, save_json


logger = get_or_create_logger(__name__)


class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.steps_valid = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0
        self.steps_valid = 0
        self.belief_loss = 0.0
        self.span_loss = 0.0
        self.resp_loss = 0.0

        self.belief_correct = 0.0
        self.span_correct = 0.0
        self.resp_correct = 0.0
        
        self.f1_slot = 0.0
        self.f1_delta = 0.0
        self.f1_act = 0.0
        self.f1_resp = 0.0
        

        self.belief_count = 0.0
        self.span_count = 0.0
        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.belief_loss += step_outputs["belief"]["loss"]
        self.belief_correct += step_outputs["belief"]["correct"]
        self.belief_count += step_outputs["belief"]["count"]
        
        self.f1_slot += step_outputs['slot_vec']['f1_score']
        self.f1_delta += step_outputs['delta_vec']['f1_score']
        self.f1_act += step_outputs['act_vec']['f1_score']
        self.f1_resp += step_outputs['resp_vec']['f1_score']
        

        if "span" in step_outputs:
            self.span_loss += step_outputs["span"]["loss"]
            self.span_correct += step_outputs["span"]["correct"]
            self.span_count += step_outputs["span"]["count"]

            do_span_stats = True
        else:
            do_span_stats = False

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]

            do_resp_stats = True
        else:
            do_resp_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_span_stats, do_resp_stats)
        else:
            self.steps_valid += 1
            
    def info_stats(self, data_type, global_step, do_span_stats=False, do_resp_stats=False, bsz = 8):
        avg_step_time = self.step_time / self.log_frequency

        belief_ppl = math.exp(self.belief_loss / self.belief_count)
        belief_acc = (self.belief_correct / self.belief_count) * 100
        
        frequency = self.log_frequency if data_type == 'train' else self.steps_valid
        f1_slot = (self.f1_slot / frequency) * 100
        f1_delta = (self.f1_delta / frequency) * 100
        f1_act = (self.f1_act / frequency) * 100
        f1_resp = (self.f1_resp / frequency) * 100
        

        self.summary_writer.add_scalar(
            "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)
        
        self.summary_writer.add_scalar(
            "{}/f1_slot".format(data_type), f1_slot, global_step=global_step)
        
        self.summary_writer.add_scalar(
            "{}/f1_delta".format(data_type), f1_delta, global_step=global_step)
        
        self.summary_writer.add_scalar(
            "{}/f1_act".format(data_type), f1_act, global_step=global_step)
        
        self.summary_writer.add_scalar(
            "{}/f1_resp".format(data_type), f1_resp, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.belief_loss, belief_ppl, belief_acc)
        
        slot_info = "[slot_vec] f1 {:.2f}".format(f1_slot)
        delta_info = "[delta_vec] f1 {:.2f}".format(f1_delta)
        act_info = "[act_vec] f1 {:.2f}".format(f1_act)
        resp_vec_info = "[resp_vec] f1 {:.2f}".format(f1_resp)

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100

            self.summary_writer.add_scalar(
                "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

            resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
                self.resp_loss, resp_ppl, resp_acc)
        else:
            resp_info = ""

        if do_span_stats:
            if self.span_count == 0:
                span_acc = 0.0
            else:
                span_acc = (self.span_correct / self.span_count) * 100

            self.summary_writer.add_scalar(
                "{}/span_loss".format(data_type), self.span_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/span_acc".format(data_type), span_acc, global_step=global_step)

            span_info = "[span] loss {0:.2f}; acc {1:.2f};".format(
                self.span_loss, span_acc)

        else:
            span_info = ""

        logger.info(
            " ".join([common_info, belief_info, resp_info, span_info, slot_info, delta_info, act_info, resp_vec_info]))

        self.init_stats()


class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader

        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
            initialize_additional_decoder = False
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
            initialize_additional_decoder = False
        else:
            model_path = self.cfg.backbone
            initialize_additional_decoder = True

        logger.info("Load model from {}".format(model_path))

        if not self.cfg.add_auxiliary_task and not self.cfg.add_additional_loss:
            model_wrapper = T5WithSpan
        else:
            model_wrapper = T5WithTokenSpan

        num_span = len(definitions.EXTRACTIVE_SLOT)

        model = model_wrapper.from_pretrained(model_path, num_span=num_span, dataset = self.cfg.dataset)

        model.resize_token_embeddings(self.reader.vocab_size)

        if initialize_additional_decoder:
            model.initialize_additional_decoder()
        '''
        if self.cfg.num_gpus > 1:
            model = torch.nn.DataParallel(model)
        '''
        model.to(self.cfg.device)
        
        if self.cfg.num_gpus > 1:
            model = torch.nn.DataParallel(model)

        return model

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)
        '''
        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model
        '''
        model = self.model

        model.save_pretrained(save_path)

        # keep chekpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_traininig_steps_per_epoch *
            self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            #num_warmup_steps = int(num_train_steps * 0.2)
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        pred = pred.view(-1)
        label = label.view(-1)

        num_count = label.ne(pad_id).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        reader = MultiWOZReader(cfg.backbone, cfg.dataset, cfg.train_data_ratio)

        self.iterator = MultiWOZIterator(reader)

        super(MultiWOZRunner, self).__init__(cfg, reader)

    def cal_f1(self, pred, label, type):
        f1_scores = []
        label = label.view(-1, label.size(-1))
        pred[label == -100] = -100
        
        if type != 'delta':
            labels=[0,1]
        else:
            labels=[0,1,2,3]
        
        for i in range(pred.shape[0]):
            f1_scores.append(f1_score(label[i].cpu(), pred[i], labels=labels, average='macro',zero_division=1.0))
        
        return np.mean(f1_scores)
        

    def step_fn(self, inputs, slot_vec_labels, delta_vec_labels, act_vec_labels, resp_vec_labels, pos_vec_labels, aspn_pos, belief_labels, resp_labels):
        inputs = inputs.to(self.cfg.device)
        slot_vec_labels = slot_vec_labels.to(self.cfg.device)
        delta_vec_labels = delta_vec_labels.to(self.cfg.device)
        act_vec_labels = act_vec_labels.to(self.cfg.device)
        resp_vec_labels = resp_vec_labels.to(self.cfg.device)
        pos_vec_labels = pos_vec_labels.to(self.cfg.device)
        belief_labels = belief_labels.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)

        attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)

        belief_outputs = self.model(input_ids=inputs,
                                    attention_mask=attention_mask,
                                    lm_labels=belief_labels,
                                    slot_vec_labels = slot_vec_labels,
                                    delta_vec_labels = delta_vec_labels,
                                    act_vec_labels = act_vec_labels,
                                    resp_vec_labels = resp_vec_labels,
                                    pos_vec_labels = pos_vec_labels,
                                    return_dict=False,
                                    add_auxiliary_task=False,
                                    add_additional_loss=self.cfg.add_additional_loss,
                                    mix_p = self.cfg.mix_p,
                                    decoder_type="belief")

        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]

        span_loss = belief_outputs[2]
        span_pred = belief_outputs[3]
        
        slot_loss = belief_outputs[5]
        slot_pred = belief_outputs[6]
        
        delta_loss = belief_outputs[8]
        delta_pred = belief_outputs[9]
        
        act_loss = belief_outputs[11]
        act_pred = belief_outputs[12]
        
        resp_vec_loss = belief_outputs[14]
        resp_vec_pred = belief_outputs[15]

        if self.cfg.task == "e2e":
            last_hidden_state = belief_outputs[17]

            encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)

            resp_outputs = self.model(attention_mask=attention_mask,
                                      encoder_outputs=encoder_outputs,
                                      lm_labels=resp_labels,
                                      aspn_pos = aspn_pos,
                                      return_dict=False,
                                      decoder_type="resp")

            resp_loss = resp_outputs[0]
            resp_pred = resp_outputs[1]

            num_resp_correct, num_resp_count = self.count_tokens(
                resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)
        
        f1_slot = self.cal_f1(slot_pred, slot_vec_labels, 'slot')
        f1_delta = self.cal_f1(delta_pred, delta_vec_labels, 'delta')
        f1_act = self.cal_f1(act_pred, act_vec_labels, 'act')
        f1_resp = self.cal_f1(resp_vec_pred, resp_vec_labels, 'resp')
        

        # if self.cfg.add_auxiliary_task:
        #     num_span_correct, num_span_count = self.count_tokens(
        #         span_pred, span_labels, pad_id=0)

        loss = belief_loss

        # if self.cfg.add_auxiliary_task and self.cfg.aux_loss_coeff > 0:
        #     loss += (self.cfg.aux_loss_coeff * span_loss)

        coeff_1, coeff_2, coeff_3, coeff_4 = 1 - int(self.cfg.st), 1 - int(self.cfg.sc), 1 - int(self.cfg.act), 1- int(self.cfg.rk)
        
        if self.cfg.task == "e2e" and self.cfg.resp_loss_coeff > 0:
            loss += (self.cfg.resp_loss_coeff * resp_loss)
            loss += (coeff_1 * slot_loss + coeff_2 * delta_loss + coeff_3 * act_loss + coeff_4 * resp_vec_loss)

        '''
        if self.cfg.num_gpus > 1:
            loss = loss.sum()
            belief_loss = belief_loss.sum()
            num_belief_correct = num_belief_correct.sum()
            num_belief_count = num_belief_count.sum()

            if self.cfg.add_auxiliary_task:
                span_loss = span_loss.sum()
                num_span_correct = num_span_correct.sum()
                num_span_count = num_span_count.sum()

            if self.cfg.task == "e2e":
                resp_loss = resp_loss.sum()
                num_resp_correct = num_resp_correct.sum()
                num_resp_count = num_resp_count.sum()
        '''

        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        # if self.cfg.add_auxiliary_task:
        #     step_outputs["span"] = {"loss": span_loss.item(),
        #                             "correct": num_span_correct.item(),
        #                             "count": num_span_count.item()}

        if self.cfg.task == "e2e":
            step_outputs["resp"] = {"loss": resp_loss.item(),
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}
            step_outputs["slot_vec"] = {"loss": slot_loss.item(),
                                    "f1_score": f1_slot}
            step_outputs["delta_vec"] = {"loss": delta_loss.item(),
                                    "f1_score": f1_delta}
            step_outputs["act_vec"] = {"loss": act_loss.item(),
                                    "f1_score": f1_act}
            step_outputs["resp_vec"] = {"loss": resp_vec_loss.item(),
                                    "f1_score": f1_resp}

        return loss, step_outputs

    def train_epoch(self, train_iterator, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()

        for step, batch in enumerate(train_iterator):
            start_time = time.time()

            inputs, labels = batch

            loss, step_outputs = self.step_fn(inputs, *labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

    def train(self):
        train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
            "train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True,
            num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size)

        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        for epoch in range(1, self.cfg.epochs + 1):
            train_iterator = self.iterator.get_data_iterator(
                train_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size, epoch, self.cfg.mu, is_training=True, with_tree=self.cfg.with_tree)

            self.train_epoch(train_iterator, optimizer, scheduler, reporter)

            logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            self.save_model(epoch)

            if not self.cfg.no_validation:
                self.validation(reporter.global_step)

    def validation(self, global_step):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)

        dev_iterator = self.iterator.get_data_iterator(
            dev_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size, is_training=False)

        reporter = Reporter(1000000, self.cfg.model_dir)

        torch.set_grad_enabled(False)
        for batch in tqdm(dev_iterator, total=num_steps, desc="Validaction"):
            start_time = time.time()

            inputs, labels = batch

            _, step_outputs = self.step_fn(inputs, *labels)

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_span_stats = True if "span" in step_outputs else False
        do_resp_stats = True if "resp" in step_outputs else False

        reporter.info_stats("dev", global_step, do_span_stats, do_resp_stats, bsz = self.cfg.batch_size)

        torch.set_grad_enabled(True)

    def finalize_bspn(self, belief_outputs, domain_history, constraint_history, span_outputs=None, input_ids=None):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn

            batch_decoded.append(decoded)
            
            # print(self.reader.tokenizer.decode(belief_output))
            

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                # logger.warn("bos/eos action token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                logger.warn("bos/eos resp token not in : {}".format(
                    self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def predict(self):
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains)

        early_stopping = True if self.cfg.beam_size > 1 else False

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in self.iterator.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                    eval_dial_list.extend(dial_ids)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context, _ = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn["user"]), self.cfg.context_size, {})

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

                attention_mask = torch.where(
                    batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                # belief tracking
                with torch.no_grad():
                    encoder_outputs = self.model(input_ids=batch_encoder_input_ids,
                                                 attention_mask=attention_mask,
                                                 return_dict=False,
                                                 encoder_only=True,
                                                 mix_p = self.cfg.mix_p,
                                                 add_auxiliary_task=self.cfg.add_auxiliary_task)

                    span_outputs, encoder_hidden_states = encoder_outputs

                    if isinstance(encoder_hidden_states, tuple):
                        last_hidden_state = encoder_hidden_states[0]
                    else:
                        last_hidden_state = encoder_hidden_states

                    # wrap up encoder outputs
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=last_hidden_state)

                    belief_outputs = self.model.generate(encoder_outputs=encoder_outputs,
                                                         attention_mask=attention_mask,
                                                         eos_token_id=self.reader.eos_token_id,
                                                         max_length=200,
                                                         do_sample=self.cfg.do_sample,
                                                         num_beams=self.cfg.beam_size,
                                                         early_stopping=early_stopping,
                                                         temperature=self.cfg.temperature,
                                                         top_k=self.cfg.top_k,
                                                         top_p=self.cfg.top_p,
                                                         decoder_type="belief")

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                if self.cfg.add_auxiliary_task:
                    pred_spans = span_outputs[1].cpu().numpy().tolist()
                    input_ids = batch_encoder_input_ids.cpu().numpy().tolist()
                else:
                    pred_spans = None
                    input_ids = None

                decoded_belief_outputs = self.finalize_bspn(
                    belief_outputs, domain_history, constraint_dicts, pred_spans, input_ids)

                
                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])
                    '''
                    print(self.reader.tokenizer.decode(input_ids[t]))
                    print(self.reader.tokenizer.decode(turn["bspn_gen"]))
                    print(turn["span"])
                    print(self.reader.tokenizer.decode(turn["bspn_gen_with_span"]))
                    input()
                    '''

                if self.cfg.task == "e2e":
                    dbpn = []

                    if self.cfg.use_true_dbpn:
                        for turn in turn_batch:
                            dbpn.append(turn["dbpn"])
                    else:
                        for turn in turn_batch:
                            if self.cfg.add_auxiliary_task:
                                bspn_gen = turn["bspn_gen_with_span"]
                            else:
                                bspn_gen = turn["bspn_gen"]

                            bspn_gen = self.reader.tokenizer.decode(
                                bspn_gen, clean_up_tokenization_spaces=False)

                            db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                      turn["turn_domain"])

                            dbpn_gen = self.reader.encode_text(
                                db_token,
                                bos_token=definitions.BOS_DB_TOKEN,
                                eos_token=definitions.EOS_DB_TOKEN)

                            turn["dbpn_gen"] = dbpn_gen

                            dbpn.append(dbpn_gen)

                    for t, db in enumerate(dbpn):
                        if self.cfg.use_true_curr_aspn:
                            db += turn_batch[t]["aspn"]

                        # T5 use pad_token as start_decoder_token_id
                        dbpn[t] = [self.reader.pad_token_id] + db

                    #print(dbpn)

                    # aspn has different length
                    if self.cfg.use_true_curr_aspn:
                        for t, _dbpn in enumerate(dbpn):
                            resp_decoder_input_ids = self.iterator.tensorize([_dbpn])

                            resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                            encoder_outputs = BaseModelOutput(
                                last_hidden_state=last_hidden_state[t].unsqueeze(0))

                            with torch.no_grad():
                                resp_outputs = self.model.generate(
                                    encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask[t].unsqueeze(0),
                                    decoder_input_ids=resp_decoder_input_ids,
                                    eos_token_id=self.reader.eos_token_id,
                                    max_length=300,
                                    do_sample=self.cfg.do_sample,
                                    num_beams=self.cfg.beam_size,
                                    early_stopping=early_stopping,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p,
                                    decoder_type="resp")

                                resp_outputs = resp_outputs.cpu().numpy().tolist()

                                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                                turn_batch[t].update(**decoded_resp_outputs[0])

                    else:
                        resp_decoder_input_ids = self.iterator.tensorize(dbpn)

                        resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                        # response generation
                        with torch.no_grad():
                            resp_outputs = self.model.generate(
                                encoder_outputs=encoder_outputs,
                                attention_mask=attention_mask,
                                decoder_input_ids=resp_decoder_input_ids,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=300,
                                do_sample=self.cfg.do_sample,
                                num_beams=self.cfg.beam_size,
                                early_stopping=early_stopping,
                                temperature=self.cfg.temperature,
                                top_k=self.cfg.top_k,
                                top_p=self.cfg.top_p,
                                decoder_type="resp")

                        resp_outputs = resp_outputs.cpu().numpy().tolist()

                        decoded_resp_outputs = self.finalize_resp(resp_outputs)

                        for t, turn in enumerate(turn_batch):
                            turn.update(**decoded_resp_outputs[t])

                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        if self.cfg.add_auxiliary_task:
                            pv_bspn = turn["bspn_gen_with_span"]
                        else:
                            pv_bspn = turn["bspn_gen"]

                    if self.cfg.use_true_dbpn:
                        pv_dbpn = turn["dbpn"]
                    else:
                        pv_dbpn = turn["dbpn_gen"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    if 'woz' not in self.cfg.dataset:
                        pv_aspn = []
                    
                    if self.cfg.use_true_prev_resp:
                        if self.cfg.task == "e2e":
                            pv_resp = turn["redx"]
                        else:
                            pv_resp = turn["resp"]
                    else:
                        pv_resp = turn["resp_gen"]

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)

        if self.cfg.task == "e2e":
            if 'woz' in self.cfg.dataset:
                bleu, success, match = evaluator.e2e_eval(
                    results, eval_dial_list=eval_dial_list, add_auxiliary_task=self.cfg.add_auxiliary_task)

                score = 0.5 * (success + match) + bleu

                logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                    match, success, bleu, score))
            else:
                metrics = evaluator.e2e_eval_cam(results, eval_data_list=eval_dial_list)
                logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                    metrics['match'], metrics['req_f1'], metrics['bleu'], 0.5 * (metrics['match'] + metrics['req_f1']) + metrics['bleu']))
        else:
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results, add_auxiliary_task=self.cfg.add_auxiliary_task)

            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))
