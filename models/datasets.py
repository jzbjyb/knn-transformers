from typing import Dict, List, Set, Callable, Tuple, Union, Callable
import os
import json
import random
import logging
from operator import itemgetter
from collections import Counter, defaultdict
import re
import string
import numpy as np
import spacy
from datasets import Dataset, load_dataset
from beir.datasets.data_loader import GenericDataLoader
from .templates import CtxPrompt
logging.basicConfig(level=logging.INFO)


class BaseDataset:
    nlp = spacy.load('en_core_web_sm')

    @classmethod
    def entity_f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: str = None
    ):
        if type(ground_truth) is str:
            ground_truth = [ground_truth]
        p = r = f1 = num_ent = 0
        for gold in ground_truth:
            pred_ents: List[str] = [cls.normalize_answer(ent.text) for ent in cls.nlp(prediction).ents]
            gold_ents: List[str] = [cls.normalize_answer(ent.text) for ent in cls.nlp(gold).ents]
            num_common_ents: int = sum((Counter(pred_ents) & Counter(gold_ents)).values())
            _p = (num_common_ents / len(pred_ents)) if len(pred_ents) else 1
            _r = (num_common_ents / len(gold_ents)) if len(gold_ents) else 1
            assert _p <= 1 and _r <= 1
            _f1 = (2 * _p * _r) / ((_p + _r) or 1)
            p, r, f1 = max(p, _p), max(r, _r), max(f1, _f1)
            num_ent += len(gold_ents)
        num_ent /= len(ground_truth)
        return {'ent_f1': f1, 'ent_precision': p, 'ent_recall': r, 'num_ent': num_ent}

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(cls.get_all_alias(ground_truth_id))
        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)

            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def get_gold_ctxs(cls, _id: str, num_golds: int = None, num_distractors: int = 1):
        assert num_golds is None
        if not hasattr(cls, 'rawdata'):
            rawdata = json.load(open(cls.raw_train_data_file, 'r'))
            cls.rawdata: Dict[str, Dict] = {e['_id']: e for e in rawdata}
        example = cls.rawdata[_id]
        title2paras: Dict[str, List[str]] = {title: sents for title, sents in example['context']}
        golds = [(f'{title}__{para_ind}', title2paras[title][para_ind].strip()) for title, para_ind in example['supporting_facts']
            if title in title2paras and para_ind < len(title2paras[title])]
        if num_distractors:
            all_tp: Set[Tuple[str, int]] = set((title, para_ind) for title in title2paras for para_ind in range(len(title2paras[title])))
            gold_tp: Set[Tuple[str, int]] = set((title, para_ind) for title, para_ind in example['supporting_facts'])
            neg_tp: List[Tuple[str, int]] = list(all_tp - gold_tp)
            random.shuffle(neg_tp)
            neg_tp = neg_tp[:num_distractors]
            negs = [(f'{title}__{para_ind}', title2paras[title][para_ind].strip()) for title, para_ind in neg_tp]
            ctxs = golds + negs
            random.shuffle(ctxs)  # shuffle golds and negs
            return ctxs
        else:
            return golds

    def format(
        self,
        fewshot: int = 0,
    ):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            # ctx
            example['demo'] = demo
            example['case'] = case
            return example
        self.dataset = self.dataset.map(_format_for_dataset)

    def retrieval_augment_examplars(
        self,
        qagent: "QueryAgent",
        add_index: bool = False,
        use_gold: bool = False,
    ):
        if qagent.append_retrieval:
            return self.retrieval_augment_examplars_append(qagent, add_index=add_index, use_gold=use_gold)
        return self.retrieval_augment_examplars_prepend(qagent, add_index=add_index, use_gold=use_gold)

    def retrieval_augment_examplars_prepend(
        self,
        qagent: "QueryAgent",
        add_index: bool = False,
        use_gold: Union[bool, Callable] = False,
    ):
        for examplar in self.examplars:
            question = examplar['question']

            if use_gold:
                _id = examplar['id']
                ctxs: List[Tuple[str, str]] = use_gold(_id)
                examplar['ctxs'] = ctxs
            else:  # search question
                ctx_ids, ctx_texts = qagent.retrieve([question], is_question=True)
                ctx_ids, ctx_texts = ctx_ids[0], ctx_texts[0]  # (ret_topk) * 2
                examplar['ctxs'] = list(zip(ctx_ids, ctx_texts))

    def retrieval_augment_examplars_append(
        self,
        qagent: "QueryAgent",
        add_index: bool = False,
        use_gold: bool = False,
    ):
        retrieval_at_beginning = qagent.retrieval_at_beginning

        for examplar in self.examplars:
            question = examplar['question']
            cot = examplar['cot']
            new_cot: List[str] = []
            assert type(cot) is not str

            # search question
            ctx_ids, ctx_texts = qagent.retrieve([question])
            ctx_ids, ctx_texts = ctx_ids[0], ctx_texts[0]  # (ret_topk) * 2
            new_cot.append(CtxPrompt.get_append_retrieval(' '.join(ctx_texts), index=0 if add_index else None))

            # search cot
            ind = 1
            ctx_ind = 0
            for t in cot:
                query = None
                if not retrieval_at_beginning:
                    if qagent.retrieval_trigers:
                        for rts, rte in qagent.retrieval_trigers:
                            if re.search(rts, t) and t.endswith(rte):
                                query = re.sub(rts, '', t).strip()
                                break
                    else:
                        query = t.strip()
                if query is not None:
                    if qagent.retrieval_trigers:
                        if add_index:
                            prefix = f'Follow up {ind}: '
                            new_cot.append(prefix + query + '\n')
                            assert 'Follow up' in qagent.retrieval_trigers[0][0] and qagent.retrieval_trigers[0][1].endswith('\n')
                        else:
                            new_cot.append(t)
                    else:
                        new_cot.append(t)
                    if use_gold:
                        assert qagent.ret_topk == 1
                        ctx_texts = [examplar['ctxs'][ctx_ind][1]]
                        ctx_ind += 1
                    else:
                        # (1, ret_topk) * 2
                        ctx_ids, ctx_texts = qagent.retrieve([query])
                        # (ret_topk) * 2
                        ctx_ids, ctx_texts = ctx_ids[0], ctx_texts[0]
                    new_cot.append(CtxPrompt.get_append_retrieval(' '.join(ctx_texts), index=ind if add_index else None))
                else:
                    prefix = f'Thought {ind}: ' if add_index else ''
                    new_cot.append(prefix + t)
                    ind += 1
            examplar['cot'] = new_cot
            examplar['ctxs'] = []


class StrategyQA(BaseDataset):
    cot_examplars: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctxs': [(None, "Hamsters are prey animals."),
                (None, "Prey animals provide food for predators.")],
            'cot': ('Hamsters are prey animals. ',
                'Prey are food for predators. ',
                'Thus, hamsters provide food for some animals.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctxs': [(None, "Brooke Shields graduated from Princeton University."),
                (None, "Princeton is ranked as the number 1 national college by US news."),
                (None, "University of Pennsylvania is ranked as number 6 national college by US news."),
                (None, "Princeton only admits around 6 percent of applicants as of 2018."),
                (None, "University of Pennsylvania accepts around 9% of applicants as of 2018.")],
            'cot': ('Brooke Shields went to Princeton University. ',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. ',
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctxs': [(None, "Hydrogen is the first element and has an atomic number of one."),
                (None, "The Spice Girls has five members."),
                (None, "To square a number, you multiply it by itself.")],
            'cot': ("Hydrogen has an atomic number of 1. ",
                "1 squared is 1. ",
                "There are 5 Spice Girls. ",
                "Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctxs': [(None, "Frost isn't uncommon to see during the month of December, as it is the winter."),
                (None, "College commencement ceremonies often happen during the months of December, May, and sometimes June.")],
            'cot': ("College commencement ceremonies can happen in December, May, and June. ",
                "December is in the winter, so there can be frost. ",
                "Thus, there could be frost at some commencements."),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctxs': [(None, "The War in Vietnam (1945-46) lasted around 6 months."),
                (None, "The gestation period for a llama is 11 months.")],
            'cot': ("The War in Vietnam was 6 months. ",
                "The gestation period for a llama is 11 months, which is more than 6 months. ",
                "Thus, a llama could not give birth twice during the War in Vietnam."),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctxs': [(None, "The density of a raw pear is about 0.59 g/cm^3."),
                (None, "The density of water is about 1 g/cm^3."),
                (None, "Objects only sink if they are denser than the surrounding fluid.")],
            'cot': ("The density of a pear is about 0.6g/cm^3, which is less than water. ",
                "Objects less dense than water float. ",
                "Thus, a pear would float."),
            'answer': 'no',
        }
    ]
    cot_demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step): '
    cot_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search): '
    cot_output_template = lambda self, cot, ans: f'{cot} So the final answer is {ans}.'

    sa_ctx_examplars: List[Dict] = [
        {
            'cot': ('Follow up: What types of animal are hamsters?\n',
                'Hamsters are prey animals.\n',
                'Follow up: Do prey provide food for any other animals?\n',
                'Prey are food for predators.'),
        },
        {
            'cot': ('Follow up: What college did Brooke Shields go to?\n',
                'Brooke Shields went to Princeton University.\n',
                'Follow up: How is Princeton University ranked?\n',
                'Princeton is ranked as the number 1 national college by US news.\n',
                'Follow up: How is University of Pennsylvania ranked?\n',
                'University of Pennsylvania is ranked as number 6 national college by US news.\n',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
        },
        {
            'cot': ('Follow up: What is the atomic number of hydrogen?\n',
                'Hydrogen has an atomic number of 1.\n',
                'Follow up: How many people are in the Spice Girls band?\n',
                'There are 5 Spice Girls.\n',
                "1 squared is 1. Thus, Hydrogen's atomic number squared is less than 5."),
        },
        {
            'cot': ('Follow up: What seasons can you expect see frost?\n',
                'Frost usually can be seen in the winter.\n',
                'Follow up: What months do college commencements occur?\n',
                'College commencement ceremonies can happen in December, May, and June.\n',
                'December is in the winter, so there can be frost. Thus, there could be frost at some commencements.'),
        },
        {
            'cot': ('Follow up: How long did the Vietnam war last?\n',
                'The War in Vietnam was 6 months.\n',
                'Follow up: How long is llama gestational period?\n',
                'The gestation period for a llama is 11 months.\n',
                '2 times 11 months is 22 months. 6 months is not longer than 22 months.'),
        },
        {
            'cot': ('Follow up: What is the density of a pear?\n',
                'The density of a pear is about 0.59 g/cm^3.\n',
                'Follow up: What is the density of water?\n',
                'The density of water is about 1 g/cm^3.\n',
                '0.59 g/cm^3 is not greater than 1 g/cm^3? Thus, a pear would float.'),
        }
    ]
    sa_ctx_demo_input_template = sa_ctx_test_input_template = lambda self, ques: f'Question: {ques}\n'
    sa_ctx_output_template = lambda self, cot, ans: f'{cot}\nSo the final answer is: {ans}.'

    tool_examplars: List[Dict] = [
        {
            'cot': ('[Search("Hamsters")] ',
                'Hamsters are prey animals. ',
                '[Search("prey animals")] ',
                'Prey are food for predators. ',
                'Thus, hamsters provide food for some animals.'),
        },
        {
            'cot': ('[Search("University Brooke Shields went to")] ',
                'Brooke Shields went to Princeton University. ',
                '[Search("rank of Princeton University and University of Pennsylvania")] ',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. ',
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
        },
        {
            'cot': ('[Search("Hydrogen atomic number")] ',
                "Hydrogen has an atomic number of 1. ",
                "1 squared is 1. ",
                '[Search("number of Spice Girls")] ',
                "There are 5 Spice Girls. ",
                "Thus, Hydrogen's atomic number squared is less than 5."),
        },
        {
            'cot': ('[Search("College commencement time")] ',
                "College commencement ceremonies can happen in December, May, and June. ",
                '[Search("forst time")] ',
                "December is in the winter, so there can be frost. ",
                "Thus, there could be frost at some commencements."),
        },
        {
            'cot': ('[Search("War in Vietnam duration)] ',
                "The War in Vietnam was 6 months. ",
                '[Search("llama gestational period")] ',
                "The gestation period for a llama is 11 months, which is more than 6 months. ",
                "Thus, a llama could not give birth twice during the War in Vietnam."),
        },
        {
            'cot': ('[Search("pear and water density")] ',
                "The density of a pear is about 0.6g/cm^3, which is less than water. ",
                "Objects less dense than water float. ",
                "Thus, a pear would float."),
        }
    ]
    tool_demo_input_template = tool_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search): '
    tool_output_template = lambda self, cot, ans: f'{cot} So the final answer is {ans}.'

    def __init__(self, beir_dir: str, prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'sa_ctx', 'tool'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        for e, ref_e in zip(self.examplars, self.cot_examplars):  # copy missing keys from cot_examplars
            for k in ref_e:
                if k not in e:
                    e[k] = ref_e[k]
        self.dataset = self.load_data(beir_dir)

    def load_data(self, beir_dir: str):
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='dev')
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                question = example['text']
                cot = example['metadata']['cot']
                ans = example['metadata']['answer']
                rel_dids = [did for did, rel in qrels[qid].items() if rel]
                ctxs = [(did, corpus[did]['text']) for did in rel_dids]
                output = self.output_template(cot, ans)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'cot': cot,
                    'answer': ans,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        return Dataset.from_list(dataset)


class HotpotQA(BaseDataset):
    raw_train_data_file: str = 'data/hotpotqa/hotpot_train_v1.1.json'
    cot_examplars: List[Dict] = [
        {
            'question': "Which magazine was started first Arthur's Magazine or First for Women?",
            'ctxs': [(None, "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century."),
                (None, "First for Women is a woman's magazine published by Bauer Media Group in the USA.")],
            'cot': ("Arthur's Magazine started in 1844. First for Women started in 1989. So Arthur's Magazine was started first."),
            'answer': "Arthur's Magazine",
        },
        {
            'question': 'The Oberoi family is part of a hotel company that has a head office in what city?',
            'ctxs': [(None, 'The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.'),
                (None, 'The Oberoi Group is a hotel company with its head office in Delhi.')],
            'cot': ("The Oberoi family is part of the hotel company called The Oberoi Group. The Oberoi Group has its head office in Delhi."),
            'answer': 'Delhi',
        },
        {
            'question': "What nationality was James Henry Miller's wife?",
            'ctxs': [(None, 'Margaret "Peggy" Seeger (born June 17, 1935) is an American folksinger.'),
                (None, 'She is also well known in Britain, where she has lived for more than 30 years, and was married to the singer and songwriter Ewan MacColl until his death in 1989.'),
                (None, 'James Henry Miller (25 January 1915 - 22 October 1989), better known by his stage name Ewan MacColl, was an English folk singer, songwriter, communist, labour activist, actor, poet, playwright and record producer.')],
            'cot': ("James Henry Miller's wife is June Miller. June Miller is an American."),
            'answer': 'American',
        },
        {
            'question': 'The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?',
            'ctxs': [(None, 'House of Anubis is a mystery television series developed for Nickelodeon based on the Dutch-Belgian television series "Het Huis Anubis".'),
                (None, 'It first aired in September 2006 and the last episode was broadcast on December 4, 2009.')],
            'cot': ('"House of Anubis" is based on the Dutch-Belgian television series Het Huis Anubis. Het Huis Anubis is firstaired in September 2006.'),
            'answer': '2006',
        },
    ]
    cot_demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step): '
    cot_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search): '
    cot_output_template = lambda self, cot, ans: f'{cot} The answer is {ans}.'

    tool_examplars: List[Dict] = [
        {
            'question': "Which magazine was started first Arthur's Magazine or First for Women?",
            'ctxs': [(None, "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century."),
                (None, "First for Women is a woman's magazine published by Bauer Media Group in the USA.")],
            'cot': ("[Search(\"Arthur's Magazine\")] Arthur's Magazine started in 1844. [Search(\"First for Women\")] First for Women started in 1989. So Arthur's Magazine was started first."),
            'answer': "Arthur's Magazine",
        },
        {
            'question': 'The Oberoi family is part of a hotel company that has a head office in what city?',
            'ctxs': [(None, 'The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.'),
                (None, 'The Oberoi Group is a hotel company with its head office in Delhi.')],
            'cot': ("[Search(\"The Oberoi family's company\")] The Oberoi family is part of the hotel company called The Oberoi Group. [Search(\"The Oberoi Group's head office\")] The Oberoi Group has its head office in Delhi."),
            'answer': 'Delhi',
        },
        {
            'question': "What nationality was James Henry Miller's wife?",
            'ctxs': [(None, 'Margaret "Peggy" Seeger (born June 17, 1935) is an American folksinger.'),
                (None, 'She is also well known in Britain, where she has lived for more than 30 years, and was married to the singer and songwriter Ewan MacColl until his death in 1989.'),
                (None, 'James Henry Miller (25 January 1915 - 22 October 1989), better known by his stage name Ewan MacColl, was an English folk singer, songwriter, communist, labour activist, actor, poet, playwright and record producer.')],
            'cot': ("[Search(\"James Henry Miller's wife\")] James Henry Miller's wife is June Miller. [Search(\"June Miller's nationality\")] June Miller is an American."),
            'answer': 'American',
        },
        {
            'question': 'The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?',
            'ctxs': [(None, 'House of Anubis is a mystery television series developed for Nickelodeon based on the Dutch-Belgian television series "Het Huis Anubis".'),
                (None, 'It first aired in September 2006 and the last episode was broadcast on December 4, 2009.')],
            'cot': ('[Search("House of Anubis was based on")] "House of Anubis" is based on the Dutch-Belgian television series Het Huis Anubis. [Search("Het Huis Anubis air date")] Het Huis Anubis is first aired in September 2006.'),
            'answer': '2006',
        },
    ]
    tool_demo_input_template = tool_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search): '
    tool_output_template = lambda self, cot, ans: f'{cot} The answer is {ans}.'

    cot_interleave_examplars: List[Dict] = [
        {
            "id": "5ae0185b55429942ec259c1b",
            "question": "What was the 2014 population of the city where Lake Wales Medical Center is located?",
            "cot": "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140.",
            "answer": "15,140"
        },
        {
            "id": "5a758ea55542992db9473680",
            "question": "who is older Jeremy Horn or Renato Sobral?",
            "cot": "Jeremy Horn was born on August 25, 1975. Renato Sobral was born on September 7, 1975. Thus, Jeremy Horn is older.",
            "answer": "Jeremy Horn"
        },
        {
            "id": "5a89d58755429946c8d6e9d9",
            "question": "Does The Border Surrender or Unsane have more members?",
            "cot": "The Border Surrender band has following members: Keith Austin, Simon Shields, Johnny Manning and Mark Austin. That is, it has 4 members. Unsane is a trio of 3 members. Thus, The Border Surrender has more members.",
            "answer": "The Border Surrender"
        },
        {
            "id": "5adfad0c554299603e41835a",
            "question": "Were Lonny and Allure both founded in the 1990s?",
            "cot": "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s.",
            "answer": "no"
        },
        {
            "id": "5abb14bd5542992ccd8e7f07",
            "question": "In which country did this Australian who was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\" receive para-military training?",
            "cot": "The Australian who was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\" is David Hicks. David Hicks received his para-military training in Afghanistan.",
            "answer": "Afghanistan"
        },
        {
            "id": "5a89c14f5542993b751ca98a",
            "question": "Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?",
            "cot": "The debut album of the band \"Hot Hot Heat\" was \"Make Up the Breakdown\". The debut album of the band \"The Operation M.D.\" was \"We Have an Emergency\".",
            "answer": "The Operation M.D."
        },
        {
            "id": "5a790e7855429970f5fffe3d",
            "question": "Who was born first? Jan de Bont or Raoul Walsh?",
            "cot": "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first.",
            "answer": "Raoul Walsh"
        },
        {
            "id": "5a88f9d55542995153361218",
            "question": "Which band formed first, Sponge Cola or Hurricane No. 1?",
            "cot": "Sponge Cola band was formed in 1998. Hurricane No. 1 was formed in 1996. Thus, Hurricane No. 1 band formed the first.",
            "answer": "Hurricane No. 1."
        },
        {
            "id": "5a7bbc50554299042af8f7d0",
            "question": "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
            "cot": "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau.",
            "answer": "The Phantom Hour"
        },
        {
            "id": "5a77acab5542992a6e59df76",
            "question": "Who was born first, James D Grant, who uses the pen name of Lee Child, or Bernhard Schlink?",
            "cot": "James D Grant, who uses the pen name of Lee Child, was born in 1954. Bernhard Schlink was born in 1944. Thus, Bernhard Schlink was born first.",
            "answer": "Bernhard Schlink"
        },
        {
            "id": "5a7fc53555429969796c1b55",
            "question": "The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?",
            "cot": "The actor that stars as Joe Proctor on the series \"Power\" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Thus, Turtle Assante's last name is Assante.",
            "answer": "Assante"
        },
        {
            "id": "5a8ed9f355429917b4a5bddd",
            "question": "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?",
            "cot": "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album.",
            "answer": "Walls and Bridges"
        },
        {
            "id": "5a754ab35542993748c89819",
            "question": "In what country was Lost Gravity manufactured?",
            "cot": "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company.",
            "answer": "Germany"
        },
        {
            "id": "5ac2ada5554299657fa2900d",
            "question": "How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?",
            "cot": "The singer of \"A Girl Like Me\" singer is Rihanna. In the American Music Awards of 2012, Rihana won one award.",
            "answer": "one"
        },
        {
            "id": "5ab92dba554299131ca422a2",
            "question": "Jeremy Theobald and Christopher Nolan share what profession?",
            "cot": "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer.",
            "answer": "producer"
        },
        {
            "id": "5add363c5542990dbb2f7dc8",
            "question": "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
            "cot": "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20.",
            "answer": "20"
        },
        {
            "id": "5abfb3435542990832d3a1c1",
            "question": "Which American neo-noir science fiction has Pierce Gagnon starred?",
            "cot": "Pierce Gagnon has starred in One Tree Hill, Looper, Wish I Was Here and Extant. Of these, Looper is an American neo-noir science fiction.",
            "answer": "Looper"
        },
        {
            "id": "5a835abe5542996488c2e426",
            "question": "Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
            "cot": "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn.",
            "answer": "Scott Glenn"
        },
        {
            "id": "5a8f44ab5542992414482a25",
            "question": "What year did Edburga of Minster-in-Thanet's father die?",
            "cot": "The father of Edburga of Minster-in-Thanet is King Centwine. Centwine died after 685.",
            "answer": "after 685"
        },
        {
            "id": "5a90620755429933b8a20508",
            "question": "James Paris Lee is best known for investing the Lee-Metford rifle and another rifle often referred to by what acronymn?",
            "cot": "James Paris Lee is best known for investing the Lee-Metford rifle and Lee-Enfield series of rifles. Lee-Enfield is often referred to by the acronym of SMLE.",
            "answer": "SMLE"
        }
    ]  # shuffled
    cot_interleave_demo_input_template = cot_interleave_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer: '
    cot_interleave_output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    cot_interleave_ret_examplars = cot_interleave_examplars
    cot_interleave_ret_demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step): '
    cot_interleave_ret_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search): '
    cot_interleave_ret_output_template = cot_interleave_output_template

    def __init__(self, split: str, prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'tool', 'cot_interleave', 'cot_interleave_ret'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        if len(self.examplars) == len(self.cot_examplars):
            for e, ref_e in zip(self.examplars, self.cot_examplars):  # copy missing keys from cot_examplars
                for k in ref_e:
                    if k not in e:
                        e[k] = ref_e[k]
        self.dataset = self.load_data(split)

    def load_data(self, split):
        # follow "Rationale-Augmented Ensembles in Language Models"
        dataset = load_dataset('hotpot_qa', 'distractor')[split]
        def _map(example: Dict):
            qid = example['id']
            question = example['question']
            qtype = example['type']
            level = example['level']
            ans = example['answer']
            title2paras: Dict[str, List[str]] = dict(zip(example['context']['title'], example['context']['sentences']))
            ctxs: List[Tuple[str, str]] = []
            for title, para_ind in zip(example['supporting_facts']['title'], example['supporting_facts']['sent_id']):
                if title in title2paras and para_ind < len(title2paras[title]):
                    ctxs.append((None, title2paras[title][para_ind]))
            output = self.output_template(cot=None, ans=ans)
            return {
                'qid': qid,
                'question': question,
                'answer': ans,
                'gold_output': output,
                'ctxs': ctxs,
                'type': qtype,
                'level': level,
            }
        return dataset.map(_map)


class WikiMultiHopQA(BaseDataset):
    raw_train_data_file: str = 'data/2wikimultihopqa/data_ids_april7/train.json'
    wid2alias_file: str = 'data/2wikimultihopqa/data_ids_april7/id_aliases.json'
    cot_examplars: List[Dict] = [
        {
            'question': "Who lived longer, Theodor Haecker or Harry Vaughan Watkins?",
            'cot': ("Theodor Haecker was 65 years old when he died. Harry Vaughan Watkins was 69 years old when he died."),
            'answer': "Harry Vaughan Watkins",
        },
        {
            'question': 'Why did the founder of Versus die?',
            'cot': ("The founder of Versus was Gianni Versace. "
                "Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997."),
            'answer': 'Shot',
        },
        {
            'question': "Who is the grandchild of Dambar Shah?",
            'cot': ("Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. "
                "He was the father of Krishna Shah. "
                "Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. "
                "He was the father of Rudra Shah."),
            'answer': 'Rudra Shah',
        },
        {
            'question': "Are both director of film FAQ: Frequently Asked Questions and director of film The Big Money from the same country?",
            'cot': ("The director of the film FAQ: Frequently Asked Questions is Carlos Atanes. "
                "The director of the film The Big Money is John Paddy Carstairs. "
                "The nationality of Carlos Atanes is Spanish. "
                "The nationality of John Paddy Carstairs is British."),
            'answer': 'No',
        },
    ]
    cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer: '
    cot_output_template = lambda self, cot, ans: f'{cot} So the final answer is {ans}.'

    cot_ret_examplars = cot_examplars
    cot_ret_demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step): '
    cot_ret_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search): '
    cot_ret_output_template = cot_output_template

    sa_examplars: List[Dict] = [
        {
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: How old was Theodor Haecker when he died?\n"
                "Intermediate answer: Theodor Haecker was 65 years old when he died.\n"
                "Follow up: How old was Harry Vaughan Watkins when he died?\n"
                "Intermediate answer: Harry Vaughan Watkins was 69 years old when he died."),
        },
        {
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who founded Versus?\n"
                "Intermediate answer: Gianni Versace.\n"
                "Follow up: Why did Gianni Versace die?\n"
                "Intermediate answer: Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997."),
        },
        {
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who is the child of Dambar Shah?\n"
                "Intermediate answer: Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. He was the father of Krishna Shah.\n"
                "Follow up: Who is the child of Krishna Shah?\n"
                "Intermediate answer: Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. He was the father of Rudra Shah."),
        },
        {
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who directed the film FAQ: Frequently Asked Questions?\n"
                "Intermediate answer: Carlos Atanes.\n"
                "Follow up: Who directed the film The Big Money?\n"
                "Intermediate answer: John Paddy Carstairs.\n"
                "Follow up: What is the nationality of Carlos Atanes?\n"
                "Intermediate answer: Carlos Atanes is Spanish.\n"
                "Follow up: What is the nationality of John Paddy Carstairs?\n"
                "Intermediate answer: John Paddy Carstairs is British."),
        },
    ]
    sa_demo_input_template = sa_test_input_template = lambda self, ques: f'Question: {ques}\n'
    sa_output_template = lambda self, cot, ans: f'{cot}\nSo the final answer is {ans}.'

    cot_interleave_examplars: List[Dict] = [
        {
            'id': '5811079c0bdc11eba7f7acde48001122',
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
            'answer': "19 June 2013",
        },
        {
            'id': '97954d9408b011ebbd84ac1f6bf848b6',
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
            'answer': "no",
        },
        {
            'id': '35bf3490096d11ebbdafac1f6bf848b6',
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
            'answer': "no",
        },
        {
            'id': 'c6805b2908a911ebbd80ac1f6bf848b6',
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
            'answer': "Martin Hodge",
        },
        {
            'id': '5897ec7a086c11ebbd61ac1f6bf848b6',
            'question': "Which film came out first, The Night Of Tricks or The Genealogy?",
            'cot': "The Night of Tricks was published in the year 1939. The Genealogy was published in the year 1979. Thus, The Night of Tricks came out first.",
            'answer': "The Night Of Tricks",
        },
        {
            'id': 'e5150a5a0bda11eba7f7acde48001122',
            'question': "When did the director of film Laughter In Hell die?",
            'cot': "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
            'answer': "August 25, 1963",
        },
        {
            'id': 'a5995da508ab11ebbd82ac1f6bf848b6',
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
            'answer': "Twenty Plus Two",
        },
        {
            'id': 'cdbb82ec0baf11ebab90acde48001122',
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
            'answer': "Genghis Khan",
        },
        {
            'id': 'f44939100bda11eba7f7acde48001122',
            'question': "What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?",
            'cot': "The mother of Grand Duke Alexei Alexandrovich of Russia is Maria Alexandrovna. Maria Alexandrovna died from tuberculosis.",
            'answer': "tuberculosis",
        },
        {
            'id': '4724c54e08e011ebbda1ac1f6bf848b6',
            'question': "Which film has the director died earlier, When The Mad Aunts Arrive or The Miracle Worker (1962 Film)?",
            'cot': "When The Mad Aunts Arrive was directed by Franz Josef Gottlieb. The Miracle Worker (1962 film) was directed by Arthur Penn. Franz Josef Gottlieb died on 23 July 2006. Arthur Penn died on September 28, 2010. Thus, of the two, the director to die earlier is Franz Josef Gottlieb, who directed When The Mad Aunts Arrive.",
            'answer': "When The Mad Aunts Arrive",
        },
        {
            'id': 'f86b4a28091711ebbdaeac1f6bf848b6',
            'question': "Which album was released earlier, What'S Inside or Cassandra'S Dream (Album)?",
            'cot': "What's Inside was released in the year 1995. Cassandra's Dream (album) was released in the year 2008. Thus, of the two, the album to release earlier is What's Inside.",
            'answer': "What's Inside",
        },
        {
            'id': '13cda43c09b311ebbdb0ac1f6bf848b6',
            'question': "Are both mountains, Serre Mourene and Monte Galbiga, located in the same country?",
            'cot': "Serre Mourene is located in Spain. Monte Galbiga is located in Italy. Thus, the two countries are not located in the same country.",
            'answer': "no",
        },
        {
            'id': '228546780bdd11eba7f7acde48001122',
            'question': "What is the date of birth of the director of film Best Friends (1982 Film)?",
            'cot': "The film Best Friends was directed by Norman Jewison. Norman Jewison was born on July 21, 1926.",
            'answer': "July 21, 1926",
        },
        {
            'id': 'c6f63bfb089e11ebbd78ac1f6bf848b6',
            'question': "Which film has the director born first, Two Weeks With Pay or Chhailla Babu?",
            'cot': "Two Weeks with Pay was directed by Maurice Campbell. Chhailla Babu was directed by Joy Mukherjee. Maurice Campbell was born on November 28, 1919. Joy Mukherjee was born on 24 February 1939. Thus, from the two directors, Chhailla Babu was born first, who directed Two Weeks With Pay.",
            'answer': "Two Weeks With Pay",
        },
        {
            'id': '1ceeab380baf11ebab90acde48001122',
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
            'answer': "Prithvipati Shah",
        },
        {
            'id': '8727d1280bdc11eba7f7acde48001122',
            'question': "When was the director of film P.S. Jerusalem born?",
            'cot': "P.S. Jerusalem was directed by Danae Elon. Danae Elon was born on December 23, 1970.",
            'answer': "December 23, 1970",
        },
        {
            'id': 'f1ccdfee094011ebbdaeac1f6bf848b6',
            'question': "Which album was released more recently, If I Have to Stand Alone or Answering Machine Music?",
            'cot': "If I Have to Stand Alone was published in the year 1991. Answering Machine Music was released in the year 1999. Thus, of the two, the album to release more recently is Answering Machine Music.",
            'answer': "Answering Machine Music",
        },
        {
            'id': '79a863dc0bdc11eba7f7acde48001122',
            'question': "Where did the director of film Maddalena (1954 Film) die?",
            'cot': "The film Maddalena is directed by Augusto Genina. Augusto Genina died in Rome.",
            'answer': "Rome",
        },
        {
            'id': '028eaef60bdb11eba7f7acde48001122',
            'question': "When did the director of film The Boy And The Fog die?",
            'cot': "The director of The Boy and the Fog is Roberto Gavaldón. Roberto Gavaldón died on September 4, 1986.",
            'answer': "September 4, 1986",
        },
        {
            'id': 'af8c6722088b11ebbd6fac1f6bf848b6',
            'question': "Are the directors of films The Sun of the Sleepless and Nevada (1927 film) both from the same country?",
            'cot': "The director of Sun of the Sleepless is Temur Babluani. The director of Nevada (1927 film) is John Waters. John Waters is from the country of America. Temur Babluani is from the country of Georgia. Thus, John Walters and Temur Babluani are not from the same country.",
            'answer': "no",
        }
    ]
    cot_interleave_demo_input_template = cot_interleave_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer: '
    cot_interleave_output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    cot_interleave_ret_examplars = cot_interleave_examplars
    cot_interleave_ret_demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step): '
    cot_interleave_ret_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer (with step-by-step & Search): '
    cot_interleave_ret_output_template = cot_interleave_output_template

    def __init__(self, beir_dir: str, prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'cot_ret', 'sa', 'cot_interleave', 'cot_interleave_ret'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        if len(self.examplars) == len(self.cot_examplars):
            for e, ref_e in zip(self.examplars, self.cot_examplars):  # copy missing keys from cot_examplars
                for k in ref_e:
                    if k not in e:
                        e[k] = ref_e[k]
        self.dataset = self.load_data(beir_dir)

    @classmethod
    def load_wid2alias(cls):
        if hasattr(cls, 'wid2alias'):
            return
        cls.wid2alias: Dict[str, List[str]] = {}
        with open(cls.wid2alias_file, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                cls.wid2alias[l['Q_id']] = l['aliases']

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        cls.load_wid2alias()
        if ground_truth_id and ground_truth_id in cls.wid2alias:
            return cls.wid2alias[ground_truth_id]
        return []

    def load_data(self, beir_dir: str):
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                question = example['text']
                ans = example['metadata']['answer']
                ans_id = example['metadata']['answer_id']
                ctxs = example['metadata']['ctxs']
                output = self.output_template(cot=None, ans=ans)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'answer_id': ans_id,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        return Dataset.from_list(dataset)

class WikiSum(BaseDataset):
    raw_train_data_file: str = ''
    summary_examplars: List[Dict] = [
        {
            "id": "24951400",
            "question": "marilyn artus",
            "answer_raw": "Marilyn Artus -LRB- Marilyn McBrier Artus -RRB- is a visual artist whose work explores the female experience . </t> <t> Marilyn has also been a burlesque promoter , curator and female artist mentor . </t> <t> She has created shows that explore the suffragette era in the US , paid tribute to founding burlesque performers , and continues to expose the many different stereotypes that women navigate on a daily basis . </t> <t> Marilyn grew up in Norman and Tulsa , Oklahoma . </t> <t> She spent two years of college at University of the Incarnate Word in San Antonio , Texas . </t> <t> She then returned to Oklahoma and finished her Bachelor of Fine Arts in printmaking at the University of Oklahoma . </t> <t> She worked for 13 years in the gift industry designing products and packaging for United Design Corporation and Relevant Products for manufacturing worldwide . </t> <t> In 2008 , Marilyn became a full-time visual artist . </t> <t> Some of the highlights of Marilyn 's art career have been solo and group shows in Oklahoma and Washington , being the first to receive the annual Brady Craft Alliance Award for Innovation in Fiber Arts in 2011 , and in 2010 leading an art making workshop at the Brooklyn Museum in New York City in association with the retrospective exhibit Seductive Subversion : Women Pop Artists , 1958-1968 .",
            "answer": "Marilyn Artus (Marilyn McBrier Artus) is a visual artist whose work explores the female experience. Marilyn has also been a burlesque promoter, curator and female artist mentor. She has created shows that explore the suffragette era in the US, paid tribute to founding burlesque performers, and continues to expose the many different stereotypes that women navigate on a daily basis. Marilyn grew up in Norman and Tulsa, Oklahoma. She spent two years of college at University of the Incarnate Word in San Antonio, Texas. She then returned to Oklahoma and finished her Bachelor of Fine Arts in printmaking at the University of Oklahoma. She worked for 13 years in the gift industry designing products and packaging for United Design Corporation and Relevant Products for manufacturing worldwide. In 2008, Marilyn became a full-time visual artist. Some of the highlights of Marilyn's art career have been solo and group shows in Oklahoma and Washington, being the first to receive the annual Brady Craft Alliance Award for Innovation in Fiber Arts in 2011, and in 2010 leading an art making workshop at the Brooklyn Museum in New York City in association with the retrospective exhibit Seductive Subversion: Women Pop Artists, 1958-1968."
        },
        {
            "id": "45683026",
            "question": "screening information dataset",
            "answer_raw": "A screening information dataset -LRB- SIDS -RRB- is a study of the hazards associated with a particular chemical substance or group of related substances , prepared under the auspices of the Organisation for Economic Co-operation and Development -LRB- OECD -RRB- . </t> <t> The substances studied are high production volume -LRB- HPV -RRB- chemicals , which are manufactured or imported in quantities of more than 1000 tonnes per year for any single OECD market . </t> <t> The list of HPV chemicals is prepared by the OECD Secretariat and updated regularly . </t> <t> As of 2004 , 4,843 chemicals were on the list . </t> <t> Of these , roughly 1000 have been prioritised for special attention , and SIDS are prepared for these chemicals , usually by an official agency in one of the OECD member countries with the collaboration of the UN International Programme on Chemical Safety -LRB- IPCS -RRB- . </t> <t> The procedures for investigating the risks of an HPV chemical are described in the OECD Manual for Investigation of HPV Chemicals . </t> <t> The initial stage is the collection of existing information -LRB- either published or supplied by manufacturers -RRB- on the chemical . </t> <t> If the existing information is insufficient to make an assessment of the risks , the chemical may be tested at this stage to collect more data . </t> <t> The initial report of the investigation is discussed at a SIDS initial assessment meeting -LRB- SIAM -RRB- , which includes : representatives of OECD member countries experts nominated by the IPCS , the OECD Business and Industry Advisory Committee , Trade Union Advisory Committee , and environmental organizations representatives of companies which produce the chemical secretariat staff from OECD , IPCS , and UNEP chemicals The SIAM can either accept the draft report or call for revisions -LRB- including further testing -RRB- . </t> <t> Once the comments and discussion of the SIAM have been taken into account , the report is published by the United Nations Environment Programme -LRB- UNEP -RRB- . </t> <t> The possibility of new testing to complete the study is what distinguishes SIDS reports from similar studies such as Concise International Chemical Assessment Documents -LRB- CICADs -RRB- . </t> <t> In this sense , SIDS are similar to European Union Risk Assessment Reports -LRB- RARs -RRB- . </t> <t> The distinction is that the SIDS programme is specifically aimed at HPV chemicals , while the chemicals selected for EU RARs are chosen more on the basis of a hazard profile , so include chemicals with much lower production volumes .",
            "answer": "A screening information dataset (SIDS) is a study of the hazards associated with a particular chemical substance or group of related substances, prepared under the auspices of the Organisation for Economic Co-operation and Development (OECD). The substances studied are high production volume (HPV) chemicals, which are manufactured or imported in quantities of more than 1000 tonnes per year for any single OECD market. The list of HPV chemicals is prepared by the OECD Secretariat and updated regularly. As of 2004, 4,843 chemicals were on the list. Of these, roughly 1000 have been prioritised for special attention, and SIDS are prepared for these chemicals, usually by an official agency in one of the OECD member countries with the collaboration of the UN International Programme on Chemical Safety (IPCS). The procedures for investigating the risks of an HPV chemical are described in the OECD Manual for Investigation of HPV Chemicals. The initial stage is the collection of existing information (either published or supplied by manufacturers) on the chemical. If the existing information is insufficient to make an assessment of the risks, the chemical may be tested at this stage to collect more data. The initial report of the investigation is discussed at a SIDS initial assessment meeting (SIAM), which includes: representatives of OECD member countries experts nominated by the IPCS, the OECD Business and Industry Advisory Committee, Trade Union Advisory Committee, and environmental organizations representatives of companies which produce the chemical secretariat staff from OECD, IPCS, and UNEP chemicals The SIAM can either accept the draft report or call for revisions (including further testing). Once the comments and discussion of the SIAM have been taken into account, the report is published by the United Nations Environment Programme (UNEP). The possibility of new testing to complete the study is what distinguishes SIDS reports from similar studies such as Concise International Chemical Assessment Documents (CICADs). In this sense, SIDS are similar to European Union Risk Assessment Reports (RARs). The distinction is that the SIDS programme is specifically aimed at HPV chemicals, while the chemicals selected for EU RARs are chosen more on the basis of a hazard profile, so include chemicals with much lower production volumes.",
        },
        {
            "id": "17326014",
            "question": "elliott smith",
            "answer_raw": "Steven Paul `` Elliott '' Smith -LRB- August 6 , 1969 -- October 21 , 2003 -RRB- was an American singer , songwriter , and musician . </t> <t> Smith was born in Omaha , Nebraska , raised primarily in Texas , and lived for much of his life in Portland , Oregon , where he first gained popularity . </t> <t> Smith 's primary instrument was the guitar , though he was also proficient with piano , clarinet , bass guitar , drums , and harmonica . </t> <t> Smith had a distinctive vocal style , characterized by his `` whispery , spiderweb-thin delivery '' , and used multi-tracking to create vocal layers , textures , and harmonies . </t> <t> After playing in the rock band Heatmiser for several years , Smith began his solo career in 1994 , with releases on the independent record labels Cavity Search and Kill Rock Stars -LRB- KRS -RRB- . </t> <t> In 1997 , he signed a contract with DreamWorks Records , for which he recorded two albums . </t> <t> Smith rose to mainstream prominence when his song `` Miss Misery '' -- included in the soundtrack for the film Good Will Hunting -LRB- 1997 -RRB- -- was nominated for an Oscar in the Best Original Song category in 1998 . </t> <t> Smith had trouble with alcohol and other drugs throughout his life , while suffering from depression , and these topics often appear in his lyrics . </t> <t> In 2003 , aged 34 , he died in Los Angeles , California , from two stab wounds to the chest . </t> <t> The autopsy evidence was inconclusive as to whether the wounds were self-inflicted . </t> <t> At the time of his death , Smith was working on his sixth studio album , From a Basement on the Hill , which was posthumously completed and released in 2004 .",
            "answer": "Steven Paul \"Elliott\" Smith (August 6, 1969 -- October 21, 2003) was an American singer, songwriter, and musician. Smith was born in Omaha, Nebraska, raised primarily in Texas, and lived for much of his life in Portland, Oregon, where he first gained popularity. Smith's primary instrument was the guitar, though he was also proficient with piano, clarinet, bass guitar, drums, and harmonica. Smith had a distinctive vocal style, characterized by his \"whispery, spiderweb-thin delivery\", and used multi-tracking to create vocal layers, textures, and harmonies. After playing in the rock band Heatmiser for several years, Smith began his solo career in 1994, with releases on the independent record labels Cavity Search and Kill Rock Stars (KRS). In 1997, he signed a contract with DreamWorks Records, for which he recorded two albums. Smith rose to mainstream prominence when his song \"Miss Misery\" -- included in the soundtrack for the film Good Will Hunting (1997) -- was nominated for an Oscar in the Best Original Song category in 1998. Smith had trouble with alcohol and other drugs throughout his life, while suffering from depression, and these topics often appear in his lyrics. In 2003, aged 34, he died in Los Angeles, California, from two stab wounds to the chest. The autopsy evidence was inconclusive as to whether the wounds were self-inflicted. At the time of his death, Smith was working on his sixth studio album, From a Basement on the Hill, which was posthumously completed and released in 2004.",
        },
        {
            "id": "53050741",
            "question": "susan wood -lrb- science fiction -rrb-",
            "answer_raw": "Susan Joan Wood -LRB- August 22 , 1948 -- November 12 , 1980 -RRB- was a Canadian literary critic , professor , author and science fiction fan and editor , born in Ottawa , Ontario . </t> <t> Wood discovered science fiction fandom while she was studying at Carleton University in the 1960s . </t> <t> Wood met fellow fan Mike Glicksohn of Toronto at Boskone VI in 1969 . </t> <t> Wood and Glicksohn married in 1970 -LRB- she subsequently sometimes published as Susan Wood Glicksohn -RRB- , and they published the fanzine Energumen together until 1973 . </t> <t> Energumen won the 1973 Hugo for Best Fanzine . </t> <t> Wood and Glicksohn were co-guests of honor at the 1975 World Science Fiction Convention . </t> <t> Wood published a great deal of trenchant criticism of the field , both in fanzines and in more formal venues . </t> <t> She received three Hugo Awards for Best Fan Writer , in 1974 , 1977 , and 1981 . </t> <t> In 1976 she was instrumental in organizing the first feminist panel at a science fiction convention , at MidAmericon -LRB- that year 's WorldCon -RRB- . </t> <t> The reaction to this helped lead to the founding of A Women 's APA and of WisCon . </t> <t> Wood earned a B.A. -LRB- 1969 -RRB- and an M.A. -LRB- 1970 -RRB- from Carleton University and a Ph.D. -LRB- 1975 -RRB- from the University of Toronto . </t> <t> She joined the English Department at the University of British Columbia in 1975 and taught Canadian literature , science fiction and children 's literature . </t> <t> She was the Vancouver editor of the Pacific Northwest Review of Books -LRB- Jan.-Oct . </t> <t> 1978 -RRB- and also edited the special science fiction/fantasy issue of Room of One 's Own . </t> <t> She wrote numerous articles and book reviews that were published in books and academic journals , while continuing to write for fanzines . </t> <t> While teaching courses in science fiction at UBC , one of her students was William Gibson ; his first published story , `` Fragments of a Hologram Rose '' , was originally written as an assignment in the class . </t> <t> A memorial scholarship fund at Carleton University was established after her death , funded in part by donations from science fiction fandom -LRB- and from the sale of parts of her collection of science fiction art -RRB- .",
            "answer": "Susan Joan Wood (August 22, 1948 -- November 12, 1980) was a Canadian literary critic, professor, author and science fiction fan and editor, born in Ottawa, Ontario. Wood discovered science fiction fandom while she was studying at Carleton University in the 1960s. Wood met fellow fan Mike Glicksohn of Toronto at Boskone VI in 1969. Wood and Glicksohn married in 1970 (she subsequently sometimes published as Susan Wood Glicksohn), and they published the fanzine Energumen together until 1973. Energumen won the 1973 Hugo for Best Fanzine. Wood and Glicksohn were co-guests of honor at the 1975 World Science Fiction Convention. Wood published a great deal of trenchant criticism of the field, both in fanzines and in more formal venues. She received three Hugo Awards for Best Fan Writer, in 1974, 1977, and 1981. In 1976 she was instrumental in organizing the first feminist panel at a science fiction convention, at MidAmericon (that year's WorldCon). The reaction to this helped lead to the founding of A Women's APA and of WisCon. Wood earned a B.A. (1969) and an M.A. (1970) from Carleton University and a Ph.D. (1975) from the University of Toronto. She joined the English Department at the University of British Columbia in 1975 and taught Canadian literature, science fiction and children's literature. She was the Vancouver editor of the Pacific Northwest Review of Books (Jan.-Oct. 1978) and also edited the special science fiction/fantasy issue of Room of One's Own. She wrote numerous articles and book reviews that were published in books and academic journals, while continuing to write for fanzines. While teaching courses in science fiction at UBC, one of her students was William Gibson; his first published story, \"Fragments of a Hologram Rose\", was originally written as an assignment in the class. A memorial scholarship fund at Carleton University was established after her death, funded in part by donations from science fiction fandom (and from the sale of parts of her collection of science fiction art).",
        },
        {
            "id": "30068119",
            "question": "al jafariyah district",
            "answer_raw": "Al Jafariyah District is a district of the Raymah Governorate , Yemen . </t> <t> As of 2003 , the district had a population of 69,705 inhabitants .",
            "answer": "Al Jafariyah District is a district of the Raymah Governorate, Yemen. As of 2003, the district had a population of 69,705 inhabitants.",
        },
        {
            "id": "35625125",
            "question": "md&di",
            "answer_raw": "MD&DI is a trade magazine for the medical device and diagnostic industry published by UBM Canon -LRB- Los Angeles -RRB- . </t> <t> It includes peer-reviewed articles on specific technology issues and overviews of key business , industry , and regulatory topics . </t> <t> It was established in 1979 . </t> <t> In 2009 it had a monthly print circulation of 48,040 but is now an online publication with a claimed circulation of 89,000 . </t> <t> UBM Canon and the magazine has also sponsored the Medical Design and Manufacturing -LRB- MD&D -RRB- West Conference & Exposition -LRB- formerly the MD&DI West Conference & Expo -RRB- , a medical device trade show , since 1978 . </t> <t> The magazine sponsored the Medical Design Excellence Awards and produces a list of 100 Notable People in the Medical Device Industry . </t> <t> The term `` use error '' was first used in May 1995 in an MD&DI guest editorial , The Issue Is ` Use , ' Not ` User , ' Error , by William Hyman .",
            "answer": "MD&DI is a trade magazine for the medical device and diagnostic industry published by UBM Canon (Los Angeles). It includes peer-reviewed articles on specific technology issues and overviews of key business, industry, and regulatory topics. It was established in 1979. In 2009 it had a monthly print circulation of 48,040 but is now an online publication with a claimed circulation of 89,000. UBM Canon and the magazine has also sponsored the Medical Design and Manufacturing (MD&D) West Conference & Exposition (formerly the MD&DI West Conference & Expo), a medical device trade show, since 1978. The magazine sponsored the Medical Design Excellence Awards and produces a list of 100 Notable People in the Medical Device Industry. The term \"use error\" was first used in May 1995 in an MD&DI guest editorial, The Issue Is 'Use,' Not 'User,' Error, by William Hyman.",
        },
        {
            "id": "26697219",
            "question": "lineage eternal",
            "answer_raw": "Lineage Eternal is an upcoming massively multiplayer online role-playing game -LRB- MMORPG -RRB- by NCSOFT . </t> <t> It is part of the Lineage series and a sequel to the first Lineage game . </t> <t> Lineage Eternal was first announced in November 2011 but has suffered numerous delays in its release schedule , with the earliest beta testing planned for 2016 . </t> <t> NCSoft announced the first South Korea Closed Beta would begin on November 30 , 2016 and end on December 04 , 2016 .",
            "answer": "Lineage Eternal is an upcoming massively multiplayer online role-playing game (MMORPG) by NCSOFT. It is part of the Lineage series and a sequel to the first Lineage game. Lineage Eternal was first announced in November 2011 but has suffered numerous delays in its release schedule, with the earliest beta testing planned for 2016. NCSoft announced the first South Korea Closed Beta would begin on November 30, 2016 and end on December 04, 2016.",
        },
        {
            "id": "28034908",
            "question": "clarksville-montgomery county school system",
            "answer_raw": "Clarksville-Montgomery County School System -LRB- CMCSS -RRB- is a system of schools in Montgomery County , Tennessee serving a population of over 147,000 people . </t> <t> It is the seventh largest district in Tennessee and has earned whole district accreditation . </t> <t> CMCSS is also ISO 9001 certified . </t> <t> Dr. B. J. Worthington is the Director of Schools . </t> <t> There are 39 schools in the district serving approximately 32,000 children from pre-kindergarten through twelfth grade : one K-5 magnet school , 23 elementary , seven middle , seven high , and one middle college -LRB- on the campus of Austin Peay State University -RRB- . </t> <t> The middle college allows high school juniors to take one university course for credit , and high school seniors to take two courses for credit . </t> <t> The school system employs about 3,900 teachers , administrators and support staff .",
            "answer": "Clarksville-Montgomery County School System (CMCSS) is a system of schools in Montgomery County, Tennessee serving a population of over 147,000 people. It is the seventh largest district in Tennessee and has earned whole district accreditation. CMCSS is also ISO 9001 certified. Dr. B. J. Worthington is the Director of Schools. There are 39 schools in the district serving approximately 32,000 children from pre-kindergarten through twelfth grade: one K-5 magnet school, 23 elementary, seven middle, seven high, and one middle college (on the campus of Austin Peay State University). The middle college allows high school juniors to take one university course for credit, and high school seniors to take two courses for credit. The school system employs about 3,900 teachers, administrators and support staff.",
        },
        {
            "id": "4692135",
            "question": "nancy wilson/cannonball adderley",
            "answer_raw": "Nancy Wilson/Cannonball Adderley is a 1961 studio album by Nancy Wilson with Cannonball Adderley and his quintet . </t> <t> Wilson considered her vocals on the album `` as a sort of easy-going third horn '' -LRB- Wilson quoted in the liner notes -RRB- . </t> <t> All tracks were recorded in New York City , those with Wilson on June 27 and 29 , 1961 , and the instrumental tracks on August 23 and 24 , 1961 .",
            "answer": "Nancy Wilson/Cannonball Adderley is a 1961 studio album by Nancy Wilson with Cannonball Adderley and his quintet. Wilson considered her vocals on the album \"as a sort of easy-going third horn\" (Wilson quoted in the liner notes). All tracks were recorded in New York City, those with Wilson on June 27 and 29, 1961, and the instrumental tracks on August 23 and 24, 1961.",
        },
        {
            "id": "39199753",
            "question": "kim bok-joo",
            "answer_raw": "Kim Bok-joo -LRB- born 17 October 1960 -RRB- is a South Korean former middle distance runner who competed in the 1984 Summer Olympics .",
            "answer": "Kim Bok-joo (born 17 October 1960) is a South Korean former middle distance runner who competed in the 1984 Summer Olympics.",
        }
    ]  # shuffled
    summary_demo_input_template = summary_test_input_template = lambda self, ques: f'Generate a summary about {ques}\nSummary: '
    summary_output_template = lambda self, cot, ans: ans

    summary_ret_examplars = summary_examplars
    summary_ret_demo_input_template = lambda self, ques: f'Generate a summary about {ques}\nSummary: '
    summary_ret_test_input_template = lambda self, ques: f'Generate a summary about {ques}\nSummary (with search): '
    summary_ret_output_template = summary_output_template

    def __init__(self, beir_dir: str, split: str = 'test', prompt_type: str = 'summary'):
        assert prompt_type in {'summary', 'summary_ret'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        if len(self.examplars) == len(self.summary_examplars):
            for e, ref_e in zip(self.examplars, self.summary_examplars):  # copy missing keys from cot_examplars
                for k in ref_e:
                    if k not in e:
                        e[k] = ref_e[k]
        self.dataset = self.load_data(beir_dir, split=split)

    @classmethod
    def clean_summary(cls, summary: str):
        pass

    @classmethod
    def get_gold_ctxs(cls, _id: str, num_golds: int = 3, num_distractors: int = 0):
        assert num_distractors == 0
        if not hasattr(cls, 'rawdata'):
            corpus, queries, qrels = GenericDataLoader('data/wikisum/wikisum_all_beir').load(split='train')
            cls.rawdata: Tuple = (corpus, qrels)
        corpus, qrels = cls.rawdata
        rel_dids = [did for did, rel in qrels[_id].items() if rel]
        golds = [(did, corpus[did].get('text')) for did in rel_dids]
        if num_golds and num_golds < len(golds):
            random.shuffle(golds)
            golds = golds[:num_golds]
        return golds

    def load_data(self, beir_dir: str, split: str = 'test'):
        qrel_file = os.path.join(beir_dir, 'qrels', f'{split}.tsv')
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        qids: Set[str] = set()
        with open(qrel_file, 'r') as fin:
            fin.readline()  # skip header
            for l in fin:
                qid, did, rel = l.strip().split()
                qids.add(qid)
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                if qid not in qids:
                    continue
                question = example['text']
                ans = example['metadata']['summary']
                output = self.output_template(cot=None, ans=ans)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'gold_output': output,
                })
        return Dataset.from_list(dataset)


class ELI5(BaseDataset):
    raw_train_data_file: str = 'data/eli5/val_with_ref.jsonl'
    cot_examplars: List[Dict] = [
        {
            "id": "1zl9do",
            "question": "why is using water to scrub dried stains on your counter so much more affective than using a dry towel?",
            "answer_raw": "Water is [really good at dissolving things](_URL_0_). The dried stuff will absorb some water and soften, making it easier to rub/scrape off.",
            "answer": "Water is really good at dissolving things. The dried stuff will absorb some water and soften, making it easier to rub/scrape off."
        },
        {
            "id": "1n40nj",
            "question": "How do short films make a profit?",
            "answer_raw": "There are tons of \"shorts\", after a fashion. They're just on television -- most TV shows are basically in the tradition of the serial shorts (once a staple of movie theaters), adapted from cinema to the small screen.\n\nModern short *films* mostly don't make a profit. Many get made by student filmmakers and artists, either as exercises, vanity projects, or to make an artistic point. They often don't cost very much to make: maybe $25,000. More expensive Hollywood/studio shorts get made essentially as prestige projects, to draw attention and praise to the studio, director, actors, etc. \n\nDisney/Pixar and their peers still make animated shorts for a few reasons. The short acts as a \"bonus\" attached to a feature film, which audiences like. It's a side project the crew can work on while taking breaks from a main project (often there's no voice talent required, so it's just animators). The short also sets a mood, without being a part of the feature film itself, which is useful to the director's storytelling. And the short is a chance to experiment with technical and storytelling techniques, which are still emerging in animation.",
            "answer": "There are tons of \"shorts\", after a fashion. They're just on television -- most TV shows are basically in the tradition of the serial shorts (once a staple of movie theaters), adapted from cinema to the small screen. Modern short *films* mostly don't make a profit. Many get made by student filmmakers and artists, either as exercises, vanity projects, or to make an artistic point. They often don't cost very much to make: maybe $25,000. More expensive Hollywood/studio shorts get made essentially as prestige projects, to draw attention and praise to the studio, director, actors, etc. Disney/Pixar and their peers still make animated shorts for a few reasons. The short acts as a \"bonus\" attached to a feature film, which audiences like. It's a side project the crew can work on while taking breaks from a main project (often there's no voice talent required, so it's just animators). The short also sets a mood, without being a part of the feature film itself, which is useful to the director's storytelling. And the short is a chance to experiment with technical and storytelling techniques, which are still emerging in animation."
        },
        {
            "id": "96bq3x",
            "question": "Why aren't carrots tastier than they are? (high sugar but not a sweet \"treat\")",
            "answer_raw": "Carrots have a lot of sugar compared to most other vegetables. Not compared to a candy bar. According to [this page](_URL_0_), it takes about two and a half pounds of carrots to get as much sugar as a single snickers bar.\n\nCorrespondingly, carrots are noticeably sweeter than most other vegetables, but not anywhere near as sweet as candy.",
            "answer": "Carrots have a lot of sugar compared to most other vegetables. Not compared to a candy bar. According to this page, it takes about two and a half pounds of carrots to get as much sugar as a single snickers bar. Correspondingly, carrots are noticeably sweeter than most other vegetables, but not anywhere near as sweet as candy.",
        },
        {
            "id": "6m5zsx",
            "question": "Why do things seem funnier when your with a friend but then stop being funny when you're alone?",
            "answer_raw": "Laughter seems to be a shared social behavior in many mammals. There are also different kinds of laughter , voluntary and involuntary. laughter mostly seems to happen in social settings. So far it appears as though its some sort of acceptance behavior like \" look we find the same thing comical\".There also huge differences in the sounds we make between involuntary and voluntary laughter. i cant find the link but i recently listened to the podcast \"undiscovered \" i think that was who did a special on laughter that was fascinating.",
            "answer": "Laughter seems to be a shared social behavior in many mammals. There are also different kinds of laughter, voluntary and involuntary. laughter mostly seems to happen in social settings. So far it appears as though its some sort of acceptance behavior like \"look we find the same thing comical\". There also huge differences in the sounds we make between involuntary and voluntary laughter. i cant find the link but i recently listened to the podcast \"undiscovered\" i think that was who did a special on laughter that was fascinating."
        },
        {
            "id": "5jr2g1",
            "question": "Why are the legislative and executive branches of government seen as corrupt and bad, while the judicial branch (especially the Supreme Court) is so revered and respected?",
            "answer_raw": "They have the highest barrier of entry.  A successful local high school basketball coach can run for Congress, and just about anyone with enough cash and/or popularity can make a serious run for President.  \n\nTechnically, the President can nominate literally anyone as a Justice, and the Senate can approve it.  \n\nHowever, historically speaking, to become a Supreme Court Judge, you first have to be a respected Federal District Judge, and before that a respected Circuit Judge, and before that a Lawyer a decade or so, and before that a Clerk in a high-powered legal office, and before that law student at a top-15 college, and before that a stellar Bacheleors and High School student.  \n\nAll along the way you need to join the right clubs and write legal opinions as a hobby so when the President's staff are making shortlists, everyone generally knows where you stand and can decide if there's a decent chance you both match the administration's views, America's needs, AND straddle enough lines to get Senate Approval. \n\nOh, and no scandals.  You need to get through the first ~50-60 years of your life in the most boring way possible.  \n\nRun the population of America through that filter and you get some pretty respectable people who are legal professionals first, politicians second.  They're not looking for fame or fortune, but are REALLY good at making a case.  \n\nGo to the SCOTUS blog and read some of their dissenting opinions on the most controversial cases and you'll really be able to get a better understanding of the 'other' side.  Because they're appointed for life* they aren't beholden to public opinion.  Since cameras aren't allowed in the proceedings, and their lives and personalities are pretty boring, it's really hard to make them a celebrity.  \n\nBasically, they're the only politicians in America that can just do their jobs as designed.",
            "answer": "They have the highest barrier of entry. A successful local high school basketball coach can run for Congress, and just about anyone with enough cash and/or popularity can make a serious run for President. Technically, the President can nominate literally anyone as a Justice, and the Senate can approve it. However, historically speaking, to become a Supreme Court Judge, you first have to be a respected Federal District Judge, and before that a respected Circuit Judge, and before that a Lawyer a decade or so, and before that a Clerk in a high-powered legal office, and before that law student at a top-15 college, and before that a stellar Bacheleors and High School student. All along the way you need to join the right clubs and write legal opinions as a hobby so when the President's staff are making shortlists, everyone generally knows where you stand and can decide if there's a decent chance you both match the administration's views, America's needs, AND straddle enough lines to get Senate Approval. Oh, and no scandals. You need to get through the first ~50-60 years of your life in the most boring way possible. Run the population of America through that filter and you get some pretty respectable people who are legal professionals first, politicians second. They're not looking for fame or fortune, but are REALLY good at making a case. Go to the SCOTUS blog and read some of their dissenting opinions on the most controversial cases and you'll really be able to get a better understanding of the 'other' side. Because they're appointed for life* they aren't beholden to public opinion. Since cameras aren't allowed in the proceedings, and their lives and personalities are pretty boring, it's really hard to make them a celebrity. Basically, they're the only politicians in America that can just do their jobs as designed."
        },
        {
            "id": "2qv5t2",
            "question": "Going from the outside to the inside of your body, at what point does skin become membrane/part of an internal organ?",
            "answer_raw": "At those points. You described it exactly. Skin is by definition on the outside of the body. As soon as it \"goes\" inside (mouth, anus, whatever), it is the gastrointestinal tract.",
            "answer": "At those points. You described it exactly. Skin is by definition on the outside of the body. As soon as it \"goes\" inside (mouth, anus, whatever), it is the gastrointestinal tract."
        },
        {
            "id": "84vt7j",
            "question": "How exactly do press release distribution services work?",
            "answer_raw": "First off, what is your goal by paying to have a press release distributed and what is the press about?\n\nThe idea of the press release is to get info in front of press to see if they are interested in covering whatever it is that you're publicizing... but you can't force people to cover it. Higher price services will target specific media in your industry, but others simply spam out to any and all media email addresses and websites.\n\nThere are those sites that will post everything, and as you mentioned, nobody reads them... because people don't read press releases in general. And a site that just posts everything and anything has too much garbage for people to weed through to find anything of use. Now having your press release on those sites will help a little bit with SEO, and help improve search results that drive people to your actual website.",
            "answer": "First off, what is your goal by paying to have a press release distributed and what is the press about? The idea of the press release is to get info in front of press to see if they are interested in covering whatever it is that you're publicizing... but you can't force people to cover it. Higher price services will target specific media in your industry, but others simply spam out to any and all media email addresses and websites. There are those sites that will post everything, and as you mentioned, nobody reads them... because people don't read press releases in general. And a site that just posts everything and anything has too much garbage for people to weed through to find anything of use. Now having your press release on those sites will help a little bit with SEO, and help improve search results that drive people to your actual website."
        },
        {
            "id": "2ubcsb",
            "question": "Why do we lose our ability to hear as we age?",
            "answer_raw": "Here is my very basic explanation:\n\nYour hearing sensors are very small hairs that line the inside of your ear canal.  Different hairs are calibrated for different frequencies (or 'pitches')  High frequencies, like sirens vibrate the hairs very quickly, while lower frequencies vibrate their hairs more slowly.  A\n\nPart of aging is our high-frequency hairs wear out and stop responding.  High frequency=more energy so they wear out faster than low-frequency hairs. The result of this is things sound 'muffled'. Eventually the hairs will be stuck 'off' permanently and gradually this takes over your ears.\n\nFun fact: when you come out of a loud concert, the 'cotton in your ears' effect is from your sensors largely being 'stunned' or 'stuck off' and not responding normally when you leave the concert for a while.  Also worth mentioning: whenever this happens some of the hairs never un-stun and you have slightly lost some of your hearing.\n\nAlso: the medical condition tinnitus or \"ringing in your ears\" is from a specific hair for a specific frequency being 'stuck on'.  Its caused (at least in the modern world) by listening to music too loud for too long.",
            "answer": "Here is my very basic explanation: Your hearing sensors are very small hairs that line the inside of your ear canal. Different hairs are calibrated for different frequencies (or 'pitches') High frequencies, like sirens vibrate the hairs very quickly, while lower frequencies vibrate their hairs more slowly. A part of aging is our high-frequency hairs wear out and stop responding. High frequency=more energy so they wear out faster than low-frequency hairs. The result of this is things sound 'muffled'. Eventually the hairs will be stuck 'off' permanently and gradually this takes over your ears. Fun fact: when you come out of a loud concert, the 'cotton in your ears' effect is from your sensors largely being 'stunned' or 'stuck off' and not responding normally when you leave the concert for a while. Also worth mentioning: whenever this happens some of the hairs never un-stun and you have slightly lost some of your hearing. Also: the medical condition tinnitus or \"ringing in your ears\" is from a specific hair for a specific frequency being 'stuck on'. Its caused (at least in the modern world) by listening to music too loud for too long."
        }
    ]  # shuffled
    cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Generate a long descriptive answer to the following question: {ques}\nAnswer: '
    cot_output_template = lambda self, cot, ans: ans

    cot_ret_examplars = cot_examplars
    cot_ret_demo_input_template = lambda self, ques: f'Generate a long descriptive answer to the following question: {ques}\nAnswer: '
    cot_ret_test_input_template = lambda self, ques: f'Generate a long descriptive answer to the following question: {ques}\nAnswer (with search): '
    cot_ret_output_template = cot_output_template

    def __init__(self, prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'cot_ret'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        if len(self.examplars) == len(self.cot_examplars):
            for e, ref_e in zip(self.examplars, self.cot_examplars):  # copy missing keys from cot_examplars
                for k in ref_e:
                    if k not in e:
                        e[k] = ref_e[k]
        self.dataset = self.load_data()

    def load_data(self, split: str = 'validation'):
        id2ctxs: Dict[str, List[str]] = defaultdict(list)
        with open(self.raw_train_data_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                _id = example['id']
                for out in example['output']:
                    for prov in out['provenance']:
                        ctx = prov['wikipedia_evidence'].strip()
                        id2ctxs[_id].append(ctx)

        rawdata = load_dataset('kilt_tasks', name='eli5')
        dataset = []
        for i, example in enumerate(rawdata[split]):
            qid = example['id']
            question = example['input']
            answers: List[str] = []
            for candidate in example['output']:
                ans = candidate['answer'].strip()
                if ans:
                    answers.append(ans)
            assert len(answers) >= 1
            ctxs = id2ctxs[qid]
            assert len(ctxs), 'no gold ctxs'
            output = self.output_template(cot=None, ans=answers[0])
            dataset.append({
                'qid': qid,
                'question': question,
                'answer': answers[0],
                'answers': answers,
                'gold_output': output,
                'ctxs': ctxs,
            })
        return Dataset.from_list(dataset)


class WoW(BaseDataset):
    raw_train_data_file: str = 'data/wow/val_with_ref.jsonl'
    cot_examplars: List[Dict] = [
        {
            "id": "540dc478-99d6-11ea-8a20-773209e30a7b_3",
            "question": "I love Nachos. It is like a yummy chippy cheesy salad.\nI also love nachos. They are a Mexican dish from northern Mexico and I'm glad we have adopted them here! lol\nI would love to give a nice plate of authentic made Nachos with some peppers.\nA man named Ignacio \"Nacho\" Anaya is the creator of the nacho dish and I would like to personally thank him! lol Do you prefer beer or chicken or just cheese?\nI like black beans and cheese with Jalapenos. Yum!\nI so love black beans on my nachos! Also, there can never be enough cheese! lol When he was asked what he wanted to call them, he said \"Nacho's Especiales\" which I find quite perfect! \nTotally fun how he came up with the name. I bet he had everyone wanting to visit and have Nachos with him.",
            "answer_raw": "I agree! As word spread about his nachos, the name just became \"special nachos\" which also works for me. Now i'm really hungry! lol",
            "answer": "I agree! As word spread about his nachos, the name just became \"special nachos\" which also works for me. Now i'm really hungry! lol.",
        },
        {
            "id": "52fddf1e-99d6-11ea-8a20-773209e30a7b_1",
            "question": "I want to be an engineer, I would love to work at NASA, such a prestigious organization.\nYes, Nasa is ndependent agency of the executive branch of the United States federal government responsible for the civilian space program.\nDo you when it was created ? It must have an interesting history.",
            "answer_raw": "President Dwight D. Eisenhower created NASA in 1958, with an orientation of encouraging peaceful applications in space science",
            "answer": "President Dwight D. Eisenhower created NASA in 1958, with an orientation of encouraging peaceful applications in space science.",
        },
        {
            "id": "5b621d64-99d6-11ea-8a20-773209e30a7b_2",
            "question": "Have you ever read the book Oliver Twist? It is about an orphan trying to survive.\nYes i love that book, its really sad though. \"Oliver Twist\" is notable for its unromantic portrayal by Dickens of criminals and their sordid lives.\nYeah, I remember how he depicted criminals. Charles Dickens wrote some great novels. My favorite was \"David Copperfield\".\nReally good one i read it too although, I really wish their could be a solution to the amount of orphan kids their is in the world.\nI agree we need to find a better solution. Some countries don't have the money to make sure they are being taken care of.",
            "answer_raw": "While the exact definition of orphan varies, one legal definition is a child bereft through \"death or disappearance etc, of their parents.",
            "answer": "While the exact definition of orphan varies, one legal definition is a child bereft through \"death or disappearance etc, of their parents.",
        },
        {
            "id": "6446ef40-99d6-11ea-8a20-773209e30a7b_0",
            "question": "I've only done yoga a handful of times, do you know much about it?",
            "answer_raw": "I have never really done it, I have tried to meditate. I know it originated in ancient india though, and I know sort of what they do",
            "answer": "I have never really done it, I have tried to meditate. I know it originated in ancient india though, and I know sort of what they do.",
        },
        {
            "id": "6058be0e-99d6-11ea-8a20-773209e30a7b_4",
            "question": "I have a Boxer named Millie, she's a medium sized short haired Boxer from Germany\nA boxer! My brother was thinking about getting one of those. What do you like about them, or what do you know about them so far?\nWell they were bred from the Old English Bulldog and the now extinct Bullenbeisser.\nI had never heard of a Bullenbeiser... Hm, alright. What's something unique about them?\nThey're brachycephalic meaning they have broad short skulls with a square muzzle and very strong jaws.\nHm.. I see. Are there any positives of the breed?\nThey're very lovable despite their powerful biting ability ideal for hanging on to large prey\nI see. ",
            "answer_raw": "They come in lots of colors also, fawn or brindled with or without white markings and some are even solid white.",
            "answer": "They come in lots of colors also, fawn or brindled with or without white markings and some are even solid white.",
        },
        {
            "id": "52dfd53c-99d6-11ea-8a20-773209e30a7b_0",
            "question": "hello i do enjoy horseback riding",
            "answer_raw": "Me too! I am a professional equestrian, I ride horses for practical working purposes in police work and I also help herd animals on a ranch.",
            "answer": "Me too! I am a professional equestrian, I ride horses for practical working purposes in police work and I also help herd animals on a ranch.",
        },
        {
            "id": "79ad45f0-99d6-11ea-8a20-773209e30a7b_1",
            "question": "I enjoy playing cue sports, a wide variety of games generally played with a cue stick.  What about you?\nYears ago I used to play pool, but wasn't very good at it. My Grandfather was a very good billiards player.",
            "answer_raw": "Did he not give you a few tips? I love to play either billiards, pool and snooker similar games but different meanings in various parts of the world.",
            "answer": "Did he not give you a few tips? I love to play either billiards, pool and snooker similar games but different meanings in various parts of the world.",
        },
        {
            "id": "5d16130e-99d6-11ea-8a20-773209e30a7b_2",
            "question": "What is ovo vegetarian ?\nIt's basically vegetarianism but you can eat eggs and not dairy.\nReally?\nYeah, in contrast with lacto vegetarianism which is the opposite.\nWhy are people ovo vegans ?",
            "answer_raw": "They see more morality to eating eggs than they do eating dairy products.",
            "answer": "They see more morality to eating eggs than they do eating dairy products.",
        }
    ]  # shuffled
    cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Given the context, generate the next response. Context: {ques}\nResponse: '
    cot_output_template = lambda self, cot, ans: ans

    cot_ret_examplars = cot_examplars
    cot_ret_demo_input_template = lambda self, ques: f'Given the context, generate the next response. Context: {ques}\nResponse: '
    cot_ret_test_input_template = lambda self, ques: f'Given the context, generate the next response. Context: {ques}\nResponse (with search): '
    cot_ret_output_template = cot_output_template

    def __init__(self, jsonl_file: str = None, prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'cot_ret'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        if len(self.examplars) == len(self.cot_examplars):
            for e, ref_e in zip(self.examplars, self.cot_examplars):  # copy missing keys from cot_examplars
                for k in ref_e:
                    if k not in e:
                        e[k] = ref_e[k]
        self.dataset = self.load_data(jsonl_file)

    def load_data(self, jsonl_file: str = None, split: str = 'validation'):
        id2ctxs_file = jsonl_file or self.raw_train_data_file
        id2ctxs: Dict[str, List[str]] = defaultdict(list)
        with open(id2ctxs_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                _id = example['id']
                for out in example['output']:
                    for prov in out['provenance']:
                        ctx = prov['wikipedia_evidence'].strip()
                        id2ctxs[_id].append(ctx)
        dataset = []
        ids_in_demo = set(e['id'] for e in self.examplars)
        if jsonl_file:
            with open(jsonl_file, 'r') as fin:
                for l in fin:
                    example = json.loads(l)
                    qid = example['id']
                    if qid in ids_in_demo:
                        continue
                    question = example['input']
                    answers: List[str] = []
                    for candidate in example['output']:
                        ans = candidate['answer'].strip()
                        if ans:
                            answers.append(ans)
                    assert len(answers) >= 1
                    ctxs = id2ctxs[qid]
                    assert len(ctxs), 'no gold ctxs'
                    output = self.output_template(cot=None, ans=answers[0])
                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answers[0],
                        'answers': answers,
                        'gold_output': output,
                        'ctxs': ctxs,
                    })
        else:
            rawdata = load_dataset('kilt_tasks', name='wow')
            for i, example in enumerate(rawdata[split]):
                qid = example['id']
                question = example['input']
                answers: List[str] = []
                for candidate in example['output']:
                    ans = candidate['answer'].strip()
                    if ans:
                        answers.append(ans)
                assert len(answers) >= 1
                ctxs = id2ctxs[qid]
                assert len(ctxs), 'no gold ctxs'
                output = self.output_template(cot=None, ans=answers[0])
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answers[0],
                    'answers': answers,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        return Dataset.from_list(dataset)


class WoWLong(WoW):
    raw_train_data_file: str = 'data/wow/val_with_ref.jsonl'
    cot_examplars: List[Dict] = [
        {
            "id": "60f4c402-99d6-11ea-8a20-773209e30a7b_2",
            "question": "I've recently tried to become an expert on hoarding, since my mom apparently has become one. It was really interesting to find out that natural disasters can lead to people becoming hoarders, and I think that's what happened to her with the last hurricane.\nAren't you concerned about her health?\nAbsolutely, especially because hoarders can make their homes into fire hazards what with all the blocked exists. This is not to mention the vermin infestations, or the waste from the animals she's collecting.. I'm worried about the common problem of stacks of things possibly falling onto her, as well.\nGood lord. What are some other factors that can lead to hoarding? It sounds like a form of OCD.",
            "answer_raw": "You're absolutely right, it is. Excessive acquisition of items, and the inability/unwillingness to get rid of the large amassed amount of things they collect, is a pattern of behavior that comes about because it apparently causes them distress/impairment to lose these things. It's not just health risks... She's dealt with the common economic burden and loss of family/friends wanting to deal with her, which often happens to hoarders",
            "answer": "You're absolutely right, it is. Excessive acquisition of items, and the inability/unwillingness to get rid of the large amassed amount of things they collect, is a pattern of behavior that comes about because it apparently causes them distress/impairment to lose these things. It's not just health risks... She's dealt with the common economic burden and loss of family/friends wanting to deal with her, which often happens to hoarders."
        },
        {
            "id": "64b2d5e8-99d6-11ea-8a20-773209e30a7b_3",
            "question": "I play guitar and I own two Fender Guitars in my collection. They are the iconic Stratocaster and Telecaster. Both were designed by Leo Fender in California in the 50s. \nI bet they sound amazing. I love the sound of guitars.\nThe telecaster is the 1st commercial successful guitar. Leo copied Ford's assembly line approach. Many players use tele in diverse music styles, Johhny Greenwood, Radiohead, Bruce Springsteen and Luther Perkins from Johnny Cash all played telecasters.\nWow, that must have been a long time ago. Johnny Cash has passed away already.\nYes, the telecaster came out in the early 1950s. Perkins used it until his un-timely death in 1968.\nIt is amazing how music changes over time and yet some beautiful elements remain the same.",
            "answer_raw": "Agreed, George Harrison also used a telecaster. His was made of Rosewood which is rare and expensive. He played on the roof of Abbey Roads studios in the Beatles last live performance. George was also a really into Indian Classical music and played sitar.",
            "answer": "Agreed, George Harrison also used a telecaster. His was made of Rosewood which is rare and expensive. He played on the roof of Abbey Roads studios in the Beatles last live performance. George was also a really into Indian Classical music and played sitar."
        },
        {
            "id": "533b8602-99d6-11ea-8a20-773209e30a7b_2",
            "question": "I really like dogs and have two myself. I know they come in many shape, sizes and colors and are great companions. Do you like dogs?\nI love dogs! I have a Lab myself, what kind of dog do you have?\nI have a Chihuahua and a Shi-tzu. After getting mine, I can see why they are termed, \"man's best friend\", but woman's best friend applies too!\nI had a Shi-tzu as a kid, he was naughy but fun.  Do tou know how long have dogs been living around humans?",
            "answer_raw": "I don't know that, but I do know they were the first species to be bred domestically. Of course, they have been continued to be bred throughout the years for various reasons, including behavior, physical attributes and sensory capabilities.",
            "answer": "I don't know that, but I do know they were the first species to be bred domestically. Of course, they have been continued to be bred throughout the years for various reasons, including behavior, physical attributes and sensory capabilities."
        },
        {
            "id": "61e138f0-99d6-11ea-8a20-773209e30a7b_3",
            "question": "I drink  hydroxyl alcohol from time to time, clear liquors are allowed on the Keto diet which is nice\nI have never heard of that, what exactly is that?\nHydroxyl alcohol or Ketogenic dieting?\nHydroxyl Alcohol, but really both!\nHydroxyl Alcohol is any organic compound that is bound to a saturated carbon atom. The Ketogenic diet is a high fat high protein diet that functions with very limited carbs for glucogenisi\nHuh, that seems like an interesting diet, I wonder what type of food you usually eat as part of a keto diet",
            "answer_raw": "Honestly, everything you would eat just made differently. I had pizza for lunch, except instead of crust, you throw mozzarella on a pan and it becomes the crust, tastes identical to thin crust. It's a diet that can actually regulate epilepsy in children",
            "answer": "Honestly, everything you would eat just made differently. I had pizza for lunch, except instead of crust, you throw mozzarella on a pan and it becomes the crust, tastes identical to thin crust. It's a diet that can actually regulate epilepsy in children."
        },
        {
            "id": "6f3ce152-99d6-11ea-8a20-773209e30a7b_3",
            "question": "I really like Granny Smith apples. Can you tell me anything about them?\nI love cooking with granny smith apples that have such a great flavor for a cooking apple. From my memory the origins are a bit fuzzy, but they were an accidental discovery by a lady in Australia in 1868.\nOh wow was she a Grandma by any chance lol.\nI imagine she was, the apple variety was discovered shortly before her passing. She gave birth to 8 children in her life, 5 survived infancy so I would say there was a good chance she had grandkids. \nWell that is quite a progeny. I am sure she has many ancestors. So what color are the apples, red?\nGranny Smith apples are typically green and turn yellow when they are over-ripened (aka rotten). They are a bit tart. \nOh ok. I seem to remember they are often dipped in caramel and sold at fairs. They are quite delicious.",
            "answer_raw": "They are also good for eating raw or dipped in caramel. From what I understand she found the tree growing out of a pile of her rubbish and it actually had really good apples. It is theorized that one of her apples trees cross pollinated with a crab apple.",
            "answer": "They are also good for eating raw or dipped in caramel. From what I understand she found the tree growing out of a pile of her rubbish and it actually had really good apples. It is theorized that one of her apples trees cross pollinated with a crab apple."
        },
        {
            "id": "717286e8-99d6-11ea-8a20-773209e30a7b_2",
            "question": "I love watching Hockey.\nOh awesome, but consider this there are actually different variations of hockey like bandy, field hockey, and ice hockey! Many people dont know that about it.\nI've never heard of bandy.\nYes the origin of the word hockey is unknown but it is thought of as a derivative of hoquet which is a middle french word for shepherd's stave.\nWhen did hockey first start?",
            "answer_raw": "The first recorded use of the word hockey was in 1773 in the book of Juvenile sports nd pastimes to which are prefixed, memoirs of the author: including  anew mode to infant education. All of that is the title so it could have been played beforehand but that is the first recorded use of the title hockey!",
            "answer": "The first recorded use of the word hockey was in 1773 in the book of Juvenile sports nd pastimes to which are prefixed, memoirs of the author: including  anew mode to infant education. All of that is the title so it could have been played beforehand but that is the first recorded use of the title hockey!"
        },
        {
            "id": "530735f0-99d6-11ea-8a20-773209e30a7b_2",
            "question": "Reading is my hubby ,and you?\nI like to read. It's a very complex process. It's a way to decode symbols and get meaning from them.\nThat's nice to know! How many hours do you spend studying?\nWell, reading takes a lot of practice and refinement. I probably spend 30 hours a week reading. There are a lot of tips and tricks I know for reading better.\nThat's good! What do you know about reading speed?",
            "answer_raw": "Well, there are a lot of strategies. To improve your reading speed I recommend Looking at the center of the column you are reading and moving just your eyes. Try to use your peripheral vision to see the edges of the columns. What do you like to read?",
            "answer": "Well, there are a lot of strategies. To improve your reading speed I recommend Looking at the center of the column you are reading and moving just your eyes. Try to use your peripheral vision to see the edges of the columns. What do you like to read?"
        },
        {
            "id": "5c5660fe-99d6-11ea-8a20-773209e30a7b_3",
            "question": "I love cats but i have terrible allergies to them.  It is an alergic reaction to something the cat produces.  Are you allergic?\nOh no! I wonder if it is to the cat's dander? No I am not alllergic to cats. What symptoms do you experience?\nI have a coughing fit, wheezing, chest tightening, itching, rash, watering eyes - you name it.  I get it!\nThat's awful. Have you seen a doctor about it?\nYes the doctor just explained that an allergen is a type of antigen that produces an abnornally vigorous immune response. and he basically recommended that I stay away from cats.  Not much else can be done aparently.\nWow, that's no fun at all.",
            "answer_raw": "Nope.  My immune system just doesnt cope with cats.  It works well detecting a wide variety of agents known as pathogens that it does not like, then it reacts! its ok though I just have dogs instead.  They are lovely pets to have too.",
            "answer": "Nope.  My immune system just doesnt cope with cats.  It works well detecting a wide variety of agents known as pathogens that it does not like, then it reacts! its ok though I just have dogs instead.  They are lovely pets to have too."
        }
    ]  # shuffled


class ASQA(BaseDataset):
    cot_examplars: List[Dict] = [
        {
            "id": "-6497998034447212269",
            "question": "When did bat out of hell come out?",
            "answer": "Bat Out of Hell is a debut album, that came out on October 21, 1977, by American rock singer Meat Loaf and composer Jim Steinman. It was developed from a musical, Neverland, a futuristic rock version of Peter Pan, which Steinman wrote for a workshop in 1974. The British television show with the same name, released on 26 November 1966, a thriller that followed two lovers, Diana Stewart, and Mark Paxton, who are haunted by the voice of Diana's husband over the telephone after he is murdered by the couple."
        },
        {
            "id": "4504214239697119124",
            "question": "Who is the chairman of the federal reserve?",
            "answer": "The current and 16th Chair of the Federal Reserve is Jerome Powell, who has held the position since 2018 after a nomination from President Trump. Powell replaced Janet Yellen, who was appointed by President Obama in 2014. Previously, the office was held by Ben Bernanke, who was first appointed by President Bush in 2006 and Alan Greenspan, who was first appointed by President Reagan in 1987."
        },
        {
            "id": "-6171603303439929107",
            "question": "What kind of car is in national lampoon's vacation?",
            "answer": "National Lampoon's Vacation, sometimes referred to as Vacation, is a 1983 American road comedy film directed by Harold Ramis, starring Chevy Chase, Beverly D'Angelo, Imogene Coca, Randy Quaid, John Candy, and Christie Brinkley in her acting debut. The Wagon Queen Family Truckster station wagon was created specifically for the film. It is based on a 1979 Ford LTD Country Squire station wagon."
        },
        {
            "id": "7189427191376660295",
            "question": "Who sang the song god's not dead?",
            "answer": "Like a Lion is song written by Daniel Bashta that was originally performed by Passion with David Crowder on the 2010 album Passion: Awakening. In 2011, this song was covered by Newsboys as God's Not Dead (Like a Lion) and released as a single from their album God's Not Dead. In the Newsboys' version, the lead vocals are performed by Michael Tait and Kevin Max is featured. The Newsboys' version charted in 2014 after the release of the film God's Not Dead. The band performs the song in a concert sequence at the end of the film."
        },
        {
            "id": "-5409444124551037323",
            "question": "Who won last triple crown of horse racing?",
            "answer": "In horse racing, a horse is said to have won the Triple Crown if they win the Kentucky Derby, Preakness Stakes, and Belmont Stakes all in the same year. The last triple crown of horse racing occurred in 2018 with the horse Justify. Justify's jockey was Mike Smith, his trainer was Bob Baffert, and his breeder was John D Gunther."
        },
        {
            "id": "-2639660647813019469",
            "question": "When did the broncos last win the superbowl?",
            "answer": "The Super Bowl is the annual American football game that determines the champion of the National Football League (NFL). Since January 1971, the winner of the American Football Conference (AFC) Championship Game has faced the winner of the National Football Conference (NFC) Championship Game in the culmination of the NFL playoffs. The Denver Broncos of the AFC have won the Super Bowl on January 25,1998; January 31, 1999; and February 7, 2016."
        },
        {
            "id": "-717926424137536243",
            "question": "How many cvs stores are there in the usa?",
            "answer": "CVS Pharmacy, Inc., previously CVS/pharmacy, is an American retail corporation headquartered in Woonsocket, Rhode Island and was owned by its original holding company Melville Corporation from its inception until its current parent company was spun off into its own company in 1996. In 1997, CVS nearly tripled its 1,400 stores after purchasing the 2,500-store Revco chain. After January 2006, CVS operated over 6,200 stores in 43 states and the District of Columbia and in some locations, CVS has two stores less than two blocks apart. CVS Pharmacy is currently the largest pharmacy chain in the United States by number of locations, with over 9,600 as of 2016, and total prescription revenue and its parent company ranks as the fifth largest U.S. corporation by FY2020 revenues in the Fortune 500."
        },
        {
            "id": "-6525373399334681447",
            "question": "Who plays max branning's wife in eastenders?",
            "answer": "Max Branning had 4 wives in EastEnders. The first wife was Sukie Smith, followed by Jo Joyner, then Kierston Wareing, and finally, Tanya Franks, as fourth."
        }
    ]  # shuffled
    #cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Generate a long descriptive answer to the following ambiguous question: {ques}\nAnswer: '
    cot_output_template = lambda self, cot, ans: ans
    cot_demo_input_template = cot_test_input_template = lambda self, ques: f'Generate a comprehensive and informative answer for a given question based on the provided search results above. You must only use information from the provided search results. Combine search results together into a coherent answer. Do not repeat text.\nQuestion: {ques}\nAnswer: '

    def __init__(self, json_file: str = None, split: str = 'dev', prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'cot_ret'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        if len(self.examplars) == len(self.cot_examplars):
            for e, ref_e in zip(self.examplars, self.cot_examplars):  # copy missing keys from cot_examplars
                for k in ref_e:
                    if k not in e:
                        e[k] = ref_e[k]
        self.dataset = self.load_data(json_file, split)

    def load_data(self, json_file: str = None, split: str = 'dev'):
        dataset = []
        num_hasctx = 0
        with open(json_file, 'r') as fin:
            data = json.load(fin)[split]
            for key, example in data.items():
                qid = key
                question = example['ambiguous_question']
                answers: List[str] = []
                title2content: Dict[str, str] = {}
                for ann in example['annotations']:
                    ans = ann['long_answer'].strip()
                    answers.append(ans)
                    for know in ann['knowledge']:
                        title2content[know['wikipage']] = know['content']
                for qa in example['qa_pairs']:
                    if qa['wikipage'] is None:
                        continue
                    title2content[qa['wikipage']] = qa['context']
                assert len(answers) >= 1
                answers = sorted(answers, key=lambda x: -len(x))  # sort based on length
                output = self.output_template(cot=None, ans=answers[0])
                ctxs: List[Tuple[str, str]] = list(title2content.items())  # could be empty
                num_hasctx += int(len(ctxs) > 0)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answers[0],
                    'answers': answers,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        logging.info(f'{num_hasctx} / {len(dataset)} have gold ctxs')
        return Dataset.from_list(dataset)


class LMData(BaseDataset):
    none_examplars = []
    none_demo_input_template = none_test_input_template = lambda self, ques: f'\nGenerate follow up of the documets: {ques}'
    none_output_template = lambda self, cot, ans: ans

    def __init__(self, jsonl_file: str = None, prompt_type: str = 'none'):
        assert prompt_type in {'none'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        self.dataset = self.load_data(jsonl_file)

    def load_data(self, jsonl_file: str = None):
        dataset = []
        with open(jsonl_file, 'r') as fin:
            for i, l in enumerate(fin):
                example = json.loads(l)
                qid = example['metadata']['line'] + '_' + str(i)
                question = example['source']
                if len(question.strip()) <= 0:  # skip empty source
                    continue
                answer = example['target']
                output = self.output_template(cot=None, ans=answer)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                    'gold_output': output,
                })
        return Dataset.from_list(dataset)
