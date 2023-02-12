from typing import Dict, List, Callable
import os
import json
from operator import itemgetter
import re
from datasets import Dataset
from beir.datasets.data_loader import GenericDataLoader
from .templates import CtxPrompt


class BaseDataset:
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
