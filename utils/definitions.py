"""
   MTTOD: utils/definitions.py

   Defines slot names and domain names for MTTOD

   This code is referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/ontology.py)

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

ALL_DOMAINS = {
    'multiwoz_2.0': ["attraction", "hotel", "restaurant", "taxi", "train", "hospital", "police"],
    'multiwoz_2.1': ["attraction", "hotel", "restaurant", "taxi", "train", "hospital", "police"],
    'multiwoz_2.2': ["attraction", "hotel", "restaurant", "taxi", "train", "hospital", "police"],
    'camrest': ["restaurant"]
}
    
DOMAIN_TOKENS = ['[hotel]', '[restaurant]', '[attraction]', '[train]', '[taxi]', '[general]', '[police]', '[hospital]']

NORMALIZE_SLOT_NAMES = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

REQUESTABLE_SLOTS = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking",
              "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}

ALL_REQSLOT = ["car", "address", "postcode", "phone", "internet", "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]

INFORMABLE_SLOTS = {
    'multiwoz_2.0': {
        "taxi": ["leave", "destination", "departure", "arrive"],
        "police": [],
        "hospital": ["department"],
        "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
        "attraction": ["area", "type", "name"],
        "train": ["destination", "day", "arrive", "departure", "people", "leave"],
        "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
    },
    'multiwoz_2.1': {
        "taxi": ["leave", "destination", "departure", "arrive"],
        "police": [],
        "hospital": ["department"],
        "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
        "attraction": ["area", "type", "name"],
        "train": ["destination", "day", "arrive", "departure", "people", "leave"],
        "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
    },
    'multiwoz_2.2': {
        "taxi": ["leave", "destination", "departure", "arrive"],
        "police": [],
        "hospital": ["department"],
        "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
        "attraction": ["area", "type", "name"],
        "train": ["destination", "day", "arrive", "departure", "people", "leave"],
        "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
    },
    'camrest':{
        "restaurant": ["food", "pricerange", "area"]
    }
}


{
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}

ALL_INFSLOT = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
               "leave", "destination", "departure", "arrive", "department", "food", "time"]

EXTRACTIVE_SLOT = ["leave", "arrive", "destination", "departure", "type", "name", "food"]

DA_ABBR_TO_SLOT_NAME = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

DIALOG_ACTS = \
    {
        'multiwoz_2.0': {
            'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
            'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
            'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
            'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
            'taxi': ['inform', 'request'],
            'police': ['inform', 'request'],
            'hospital': ['inform', 'request'],
            'general': ['bye', 'greet', 'reqmore', 'welcome'],
        },
        'multiwoz_2.1': {
            'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
            'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
            'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
            'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
            'taxi': ['inform', 'request'],
            'police': ['inform', 'request'],
            'hospital': ['inform', 'request'],
            'general': ['bye', 'greet', 'reqmore', 'welcome'],
        },
        'multiwoz_2.2': {
            'restaurant': ['inform', 'nobook', 'offerbook', 'nooffer', 'request', 'select', 'recommend', 'offerbooked'], 
            'hotel': ['inform', 'nobook', 'offerbook', 'nooffer', 'request', 'select', 'recommend', 'offerbooked'], 
            'general': ['greet', 'nobook', 'thank', 'offerbook', 'welcome', 'request', 'bye', 'reqmore', 'offerbooked'], 
            'taxi': ['inform', 'offerbook', 'request', 'offerbooked'], 
            'attraction': ['inform', 'nobook', 'offerbook', 'nooffer', 'request', 'select', 'recommend', 'offerbooked'], 
            'train': ['inform', 'nobook', 'offerbook', 'nooffer', 'request', 'select', 'offerbooked'], 
            'police': ['inform'], 
            'hospital': ['inform', 'offerbook', 'request', 'offerbooked']
        },
        'camrest':{
            'restaurant': ['inform','request']
        }
    }

BOS_USER_TOKEN = "<bos_user>"
EOS_USER_TOKEN = "<eos_user>"

USER_TOKENS = [BOS_USER_TOKEN, EOS_USER_TOKEN]

BOS_BELIEF_TOKEN = "<bos_belief>"
EOS_BELIEF_TOKEN = "<eos_belief>"

BELIEF_TOKENS = [BOS_BELIEF_TOKEN, EOS_BELIEF_TOKEN]

BOS_DB_TOKEN = "<bos_db>"
EOS_DB_TOKEN = "<eos_db>"

DB_TOKENS = [BOS_DB_TOKEN, EOS_DB_TOKEN]

BOS_ACTION_TOKEN = "<bos_act>"
EOS_ACTION_TOKEN = "<eos_act>"

ACTION_TOKENS = [BOS_ACTION_TOKEN, EOS_ACTION_TOKEN]

BOS_RESP_TOKEN = "<bos_resp>"
EOS_RESP_TOKEN = "<eos_resp>"

RESP_TOKENS = [BOS_RESP_TOKEN, EOS_RESP_TOKEN]

DB_NULL_TOKEN = "[db_null]"
DB_0_TOKEN = "[db_0]"
DB_1_TOKEN = "[db_1]"
DB_2_TOKEN = "[db_2]"
DB_3_TOKEN = "[db_3]"

DB_STATE_TOKENS = [DB_NULL_TOKEN, DB_0_TOKEN, DB_1_TOKEN, DB_2_TOKEN, DB_3_TOKEN]

SPECIAL_TOKENS = USER_TOKENS + BELIEF_TOKENS + DB_TOKENS + ACTION_TOKENS + RESP_TOKENS + DB_STATE_TOKENS

SLOT_TYPES = {}
for d,v in INFORMABLE_SLOTS.items():
    SLOT_TYPES[d] = []
    for k, v1 in v.items():
        for t in v1:
            SLOT_TYPES[d].append(k + '-' + t)

ACT_TYPES = {}

for d, v in DIALOG_ACTS.items():
    ACT_TYPES[d] = []
    for k,v1 in v.items():
        for t in v1:
            ACT_TYPES[d].append(k + '-' + t)

mapping_for_22 = {
    '[choice]': '[value_choice]',
    '[food]': '[value_food]',
    '[pricerange]': '[value_pricerange]',
    '[ref]': '[value_reference]',
    '[name]': '[value_name]',
    '[area]': '[value_area]',
    '[stars]': '[value_stars]',
    '[type]': '[value_type]',
    '[bookstay]': '[value_stay]',
    '[departure]': '[value_departure]',
    '[phone]': '[value_phone]',
    '[bookpeople]': '[value_people]',
    '[destination]': '[value_destination]',
    '[address]': '[value_address]',
    '[booktime]': '[value_time]',
    '[bookday]': '[value_day]',
    '[postcode]': '[value_postcode]',
    '[leaveat]': '[value_leave]',
    '[entrancefee]': '[value_price]',
    '[price]': '[value_price]',
    '[trainid]': '[value_id]',
    '[arriveby]': '[value_arrive]',
    '[day]': '[value_day]',
    '[duration]': '[value_time]',
    '[department]': '[value_department]'
}


RESP_SPEC_TOKENS = \
{
    'multiwoz_2.0': 
        ['[value_choice]', '[value_pricerange]', '[value_day]', '[value_reference]', '[value_name]', '[value_address]', '[value_phone]', '[value_postcode]', '[value_area]', '[value_departure]', '[value_leave]', '[value_arrive]', '[value_type]', '[value_price]', '[value_destination]', '[value_time]', '[value_food]', '[value_id]', '[value_people]', '[value_department]', '[value_car]', '[value_stars]', '[value_stay]'],
    'multiwoz_2.1': 
        ['[value_choice]', '[value_pricerange]', '[value_day]', '[value_reference]', '[value_name]', '[value_address]', '[value_phone]', '[value_postcode]', '[value_area]', '[value_departure]', '[value_leave]', '[value_arrive]', '[value_type]', '[value_price]', '[value_destination]', '[value_time]', '[value_food]', '[value_id]', '[value_people]', '[value_department]', '[value_car]', '[value_stars]', '[value_stay]'],
    'multiwoz_2.2': 
        ['[value_choice]', '[value_pricerange]', '[value_day]', '[value_reference]', '[value_name]', '[value_address]', '[value_phone]', '[value_postcode]', '[value_area]', '[value_departure]', '[value_leave]', '[value_arrive]', '[value_type]', '[value_price]', '[value_destination]', '[value_time]', '[value_food]', '[value_id]', '[value_people]', '[value_department]', '[value_car]', '[value_stars]', '[value_stay]'],
        # ['[value_choice]', '[value_food]', '[value_pricerange]', '[value_name]', '[value_area]', '[value_reference]', '[value_stars]', '[value_type]', '[value_bookstay]', '[value_phone]', '[value_address]', '[value_postcode]', '[value_departure]', '[value_destination]', '[value_bookpeople]', '[value_booktime]', '[value_bookday]', '[value_arriveby]', '[value_leaveat]', '[value_day]', '[value_entrancefee]', '[value_duration]', '[value_trainid]', '[value_price]', '[value_id]', '[value_leave]', '[value_arrive]', '[value_time]', '[value_openhours]', '[value_car]', '[value_department]'],
    'camrest': 
        ['[value_phone]', '[value_area]', '[value_food]', '[value_postcode]', '[value_name]', '[value_address]', '[value_pricerange]']
}


