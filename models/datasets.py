from typing import Dict, List, Set, Callable, Tuple, Union, Callable
import os
import json
import random
from operator import itemgetter
from collections import Counter
import re
import string
import numpy as np
from datasets import Dataset, load_dataset
from beir.datasets.data_loader import GenericDataLoader
from .templates import CtxPrompt


class BaseDataset:
    @classmethod
    def exact_match_score(cls, prediction, ground_truth):
        correct = int(cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth))
        return {'correct': correct, 'incorrect': 1 - correct}

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
            'id': '5ab92dba554299131ca422a2',
            'question': "Jeremy Theobald and Christopher Nolan share what profession?",
            'cot': "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer.",
            'answer': "producer",
        },
        {
            'id': '5a7bbc50554299042af8f7d0',
            'question': "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
            'cot': "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau.",
            'answer': "The Phantom Hour",
        },
        {
            'id': '5add363c5542990dbb2f7dc8',
            'question': "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
            'cot': "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20.",
            'answer': "20",
        },
        {
            'id': '5a835abe5542996488c2e426',
            'question': "Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
            'cot': "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn.",
            'answer': "Scott Glenn",
        },
        {
            'id': '5ae0185b55429942ec259c1b',
            'question': "What was the 2014 population of the city where Lake Wales Medical Center is located?",
            'cot': "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140.",
            'answer': "15,140",
        },
        {
            'id': '5a790e7855429970f5fffe3d',
            'question': "Who was born first? Jan de Bont or Raoul Walsh?",
            'cot': "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first.",
            'answer': "Raoul Walsh",
        },
        {
            'id': '5a754ab35542993748c89819',
            'question': "In what country was Lost Gravity manufactured?",
            'cot': "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company.",
            'answer': "Germany",
        },
        {
            'id': '5a89c14f5542993b751ca98a',
            'question': "Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?",
            'cot': "The debut album of the band \"Hot Hot Heat\" was \"Make Up the Breakdown\". The debut album of the band \"The Operation M.D.\" was \"We Have an Emergency\".",
            'answer': "The Operation M.D.",
        },
        {
            'id': '5abb14bd5542992ccd8e7f07',
            'question': "In which country did this Australian who was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\" receive para-military training?",
            'cot': "The Australian who was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\" is David Hicks. David Hicks received his para-military training in Afghanistan.",
            'answer': "Afghanistan",
        },
        {
            'id': '5a89d58755429946c8d6e9d9',
            'question': "Does The Border Surrender or Unsane have more members?",
            'cot': "The Border Surrender band has following members: Keith Austin, Simon Shields, Johnny Manning and Mark Austin. That is, it has 4 members. Unsane is a trio of 3 members. Thus, The Border Surrender has more members.",
            'answer': "The Border Surrender",
        },
        {
            'id': '5a88f9d55542995153361218',
            'question': "Which band formed first, Sponge Cola or Hurricane No. 1?",
            'cot': "Sponge Cola band was formed in 1998. Hurricane No. 1 was formed in 1996. Thus, Hurricane No. 1 band formed the first.",
            'answer': "Hurricane No. 1.",
        },
        {
            'id': '5a90620755429933b8a20508',
            'question': "James Paris Lee is best known for investing the Lee-Metford rifle and another rifle often referred to by what acronymn?",
            'cot': "James Paris Lee is best known for investing the Lee-Metford rifle and Lee-Enfield series of rifles. Lee-Enfield is often referred to by the acronym of SMLE.",
            'answer': "SMLE",
        },
        {
            'id': '5a77acab5542992a6e59df76',
            'question': "Who was born first, James D Grant, who uses the pen name of Lee Child, or Bernhard Schlink?",
            'cot': "James D Grant, who uses the pen name of Lee Child, was born in 1954. Bernhard Schlink was born in 1944. Thus, Bernhard Schlink was born first.",
            'answer': "Bernhard Schlink",
        },
        {

            'id': '5abfb3435542990832d3a1c1',
            'question': "Which American neo-noir science fiction has Pierce Gagnon starred?",
            'cot': "Pierce Gagnon has starred in One Tree Hill, Looper, Wish I Was Here and Extant. Of these, Looper is an American neo-noir science fiction.",
            'answer': "Looper",
        },
        {
            'id': '5a8f44ab5542992414482a25',
            'question': "What year did Edburga of Minster-in-Thanet's father die?",
            'cot': "The father of Edburga of Minster-in-Thanet is King Centwine. Centwine died after 685.",
            'answer': "after 685",
        },
        {
            'id': '5adfad0c554299603e41835a',
            'question': "Were Lonny and Allure both founded in the 1990s?",
            'cot': "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s.",
            'answer': "no",
        },
        {
            'id': '5a7fc53555429969796c1b55',
            'question': "The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?",
            'cot': "The actor that stars as Joe Proctor on the series \"Power\" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Thus, Turtle Assante's last name is Assante.",
            'answer': "Assante",
        },
        {
            'id': '5a8ed9f355429917b4a5bddd',
            'question': "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?",
            'cot': "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album.",
            'answer': "Walls and Bridges",
        },
        {
            'id': '5ac2ada5554299657fa2900d',
            'question': "How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?",
            'cot': "The singer of \"A Girl Like Me\" singer is Rihanna. In the American Music Awards of 2012, Rihana won one award.",
            'answer': "one",
        },
        {
            'id': '5a758ea55542992db9473680',
            'question': "who is older Jeremy Horn or Renato Sobral?",
            'cot': "Jeremy Horn was born on August 25, 1975. Renato Sobral was born on September 7, 1975. Thus, Jeremy Horn is older.",
            'answer': "Jeremy Horn",
        }
    ]
    cot_interleave_demo_input_template = cot_interleave_test_input_template = lambda self, ques: f'Question: {ques}\nAnswer: '
    cot_interleave_output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

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

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
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
    def get_gold_ctxs(cls, _id: str, num_distractors: int = 1):
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
