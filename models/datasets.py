from typing import Dict, List, Callable, Tuple
import os
import json
from operator import itemgetter
import re
import string
import numpy as np
from datasets import Dataset, load_dataset
from beir.datasets.data_loader import GenericDataLoader
from .templates import CtxPrompt


class BaseDataset:
    @classmethod
    def exact_match_score(cls, prediction, ground_truth):
        return cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth)

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
            cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else [],
            'ctx': ' '.join(map(itemgetter(1), self.examplars[i]['ctxs'])) if 'ctxs' in self.examplars[i] and self.examplars[i]['ctxs'] else None,
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
        retrieval_at_beginning: bool = False,
        add_index: bool = False,
        use_gold: bool = False,
    ):
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


    def __init__(self, split: str, prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'tool'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        for e, ref_e in zip(self.examplars, self.cot_examplars):  # copy missing keys from cot_examplars
            for k in ref_e:
                if k not in e:
                    e[k] = ref_e[k]
        self.dataset = self.load_data(split)

    def load_data(self, split):
        # follow "Rationale-Augmented Ensembles in Language Models"
        dataset = load_dataset('hotpot_qa', 'distractor')[split].select(range(0, 1000))
        def _map(example: Dict):
            qid = example['id']
            question = example['question']
            qtype = example['type']
            level = example['level']
            ans = example['answer']
            cot = ''
            title2paras: Dict[str, List[str]] = dict(zip(example['context']['title'], example['context']['sentences']))
            ctxs: List[Tuple[str, str]] = []
            for title, para_ind in zip(example['supporting_facts']['title'], example['supporting_facts']['sent_id']):
                ctxs.append((None, title2paras[title][para_ind]))
            output = self.output_template(cot, ans)
            return {
                'qid': qid,
                'question': question,
                'cot': cot,
                'answer': ans,
                'gold_output': output,
                'ctxs': ctxs,
                'type': qtype,
                'level': level,
            }
        return dataset.map(_map)


class WikiMultiHopQA(BaseDataset):
    sa_examplars: List[Dict] = [
        {
            'question': "Who lived longer, Theodor Haecker or Harry Vaughan Watkins?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: How old was Theodor Haecker when he died?\n"
                "Intermediate answer: Theodor Haecker was 65 years old when he died.\n"
                "Follow up: How old was Harry Vaughan Watkins when he died?\n"
                "Intermediate answer: Harry Vaughan Watkins was 69 years old when he died."),
            'answer': "Harry Vaughan Watkins",
        },
        {
            'question': 'Why did the founder of Versus die?',
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who founded Versus?\n"
                "Intermediate answer: Gianni Versace.\n"
                "Follow up: Why did Gianni Versace die?\n"
                "Intermediate answer: Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997."),
            'answer': 'Shot',
        },
        {
            'question': "Who is the grandchild of Dambar Shah?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who is the child of Dambar Shah?\n"
                "Intermediate answer: Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. He was the father of Krishna Shah.\n"
                "Follow up: Who is the child of Krishna Shah?\n"
                "Intermediate answer: Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. He was the father of Rudra Shah."),
            'answer': 'Rudra Shah',
        },
        {
            'question': "Are both director of film FAQ: Frequently Asked Questions and director of film The Big Money from the same country?",
            'cot': ("Are follow up questions needed here: Yes.\n"
                "Follow up: Who directed the film FAQ: Frequently Asked Questions?\n"
                "Intermediate answer: Carlos Atanes.\n"
                "Follow up: Who directed the film The Big Money?\n"
                "Intermediate answer: John Paddy Carstairs.\n"
                "Follow up: What is the nationality of Carlos Atanes?\n"
                "Intermediate answer: Carlos Atanes is Spanish.\n"
                "Follow up: What is the nationality of John Paddy Carstairs?\n"
                "Intermediate answer: John Paddy Carstairs is British."),
            'answer': 'No',
        },
    ]
    sa_demo_input_template = sa_test_input_template = lambda self, ques: f'Question: {ques}\n'
    sa_output_template = lambda self, cot, ans: f'{cot}\nSo the final answer is {ans}.'

    def __init__(self, beir_dir: str, prompt_type: str = 'cot'):
        assert prompt_type in {'sa'}
        self.demo_input_template = getattr(self, f'{prompt_type}_demo_input_template')
        self.test_input_template = getattr(self, f'{prompt_type}_test_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplars = getattr(self, f'{prompt_type}_examplars')
        self.dataset = self.load_data(beir_dir)

    @classmethod
    def load_wid2alias(cls, wid2alias_file: str):
        if hasattr(cls, 'wid2alias'):
            return
        cls.wid2alias: Dict[str, List[str]] = {}
        with open(wid2alias_file, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                cls.wid2alias[l['Q_id']] = l['aliases']

    @classmethod
    def exact_match_score(
        cls,
        prediction,
        ground_truth: str,
        ground_truth_id: str = None,
        wid2alias_file: str = 'data/2wikimultihopqa/data_ids_april7/id_aliases.json'):
        cls.load_wid2alias(wid2alias_file)
        ground_truths = {ground_truth}
        if ground_truth_id and ground_truth_id in cls.wid2alias:
            ground_truths.update(cls.wid2alias[ground_truth_id])
        print(len(ground_truths), ground_truth_id, ground_truth_id in cls.wid2alias)
        return np.max([cls.normalize_answer(prediction) == cls.normalize_answer(gt) for gt in ground_truths])

    def load_data(self, beir_dir: str):
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                question = example['text']
                cot = ''
                ans = example['metadata']['answer']
                ans_id = example['metadata']['answer_id']
                ctxs = example['metadata']['ctxs']
                output = self.output_template(cot, ans)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'cot': cot,
                    'answer': ans,
                    'answer_id': ans_id,
                    'gold_output': output,
                    'ctxs': ctxs,
                })
        return Dataset.from_list(dataset)
