from typing import List, Tuple, Any, Union
import argparse
import random
import json
from collections import defaultdict
import csv
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from kilt.knowledge_source import KnowledgeSource

class Wikipedia(object):
    def __init__(self):
        self.ks = KnowledgeSource()

    def get_provenance(self, wiki_id: str, ps: int, pe: int, cs: int, ce: int, whole_paragraph: bool = False) -> str:
        page = self.ks.get_page_by_id(wiki_id)
        prov: List[str] = []
        if ps == pe:  # only one paragraph
            if whole_paragraph:
                prov.append(page['text'][ps])
            else:
                prov.append(page['text'][ps][cs:ce])
        else:
            for pi in range(ps, pe + 1):
                if pi == ps:
                    if whole_paragraph:
                        prov.append(page['text'][pi])
                    else:
                        prov.append(page['text'][pi][cs:])
                elif pi == pe:
                    if whole_paragraph:
                        prov.append(page['text'][pi])
                    else:
                        prov.append(page['text'][pi][:ce])
                else:
                    prov.append(page['text'][pi])
        return ' '.join(prov)

def prep_eli5(
    args, 
    split: str = 'validation', 
    evidence_method: str = 'provenance', 
    skip_answer_as_evidence: bool = True):
    assert evidence_method in {'provenance', 'self_provenance', 'answer', 'self_answer'}

    qa_file = f'{args.out_file}_qa.json'
    prov_file = f'{args.out_file}_evidence.json'
    eli5 = load_dataset('kilt_tasks', name='eli5')
    wikipedia = Wikipedia()

    with open(qa_file, 'w') as qfin, open(prov_file, 'w') as pfin:
        for i, example in enumerate(eli5[split]):
            inp: str = example['input']
            answer: str = None
            evidences: List[str] = []

            # collect
            for ans_or_provs in example['output']:
                ans = ans_or_provs['answer'].strip()
                provs = ans_or_provs['provenance']

                this_is_ans = False
                if ans and answer is None:  # use the first answer as the qa pair
                    this_is_ans = True
                    answer = ans
                if 'self' in evidence_method or not skip_answer_as_evidence or not this_is_ans:  # whether use the real answer
                    if 'provenance' in evidence_method and len(provs):  # collect all provenance
                        for prov in provs:
                            wiki_id = prov['wikipedia_id']
                            ps, pe, cs, ce = prov['start_paragraph_id'], prov['end_paragraph_id'], prov['start_character'], prov['end_character']
                            #prov = prov['meta']['evidence_span'][-1].split('\r')[0]
                            prov = wikipedia.get_provenance(wiki_id, ps, pe, cs, ce, whole_paragraph=True)  # always use the whole paragraph
                            evidences.append(prov)
                    if 'answer' in evidence_method and ans:
                        evidences.append(ans)
            
            # write qa pairs
            qfin.write(json.dumps({'translation': {'en': inp, 'zh': answer}}) + '\n')

            # write evidences
            if 'self' in evidence_method:
                for evi in evidences:
                    pfin.write(json.dumps({'translation': {'en': inp, 'zh': answer, 'decoder_prefix': evi}}) + '\n')
            else:
                for evi in evidences:
                    pfin.write(json.dumps({'translation': {'en': inp, 'zh': evi}}) + '\n')

class PredictionWithRetrieval(object):
    def __init__(self, n_heads: int, topk: int, tokenizer: AutoTokenizer, use_tokenizer: bool = False):
        self.n_heads = n_heads
        self.topk = topk
        self.tokenizer = tokenizer
        self.use_tokenizer = use_tokenizer
        self.tokens: List[List] = []

    def convert_token_id(self, token_id: int) -> Union[str, int]:
        if self.use_tokenizer:
            return self.tokenizer.convert_ids_to_tokens([token_id])[0]
        else:
            return token_id
    
    def add_one_word(self, line: str):
        seq = line.strip().split(' ')
        token: List = []  # [pred_token, head1, head2, ..., headn]
        self.tokens.append(token)
        token.append(self.convert_token_id(int(seq[0])))
        for i in range(1, len(seq)):
            t = int(seq[i])
            i -= 1
            if i % (self.topk * 2) == 0:  # a new head
                token.append([])
            # convert token id to token
            if i % 2 == 0:  # token
                token[-1].append(self.convert_token_id(t))
            else:  # id
                token[-1][-1] = (token[-1][-1], t)
        assert len(token) == 1 + self.n_heads
    
    def get_ids(self, head_idx: int):
        return np.array([[ret_id for (_, ret_id) in tok[head_idx + 1]] for tok in self.tokens])
    
    def get_ids_portion(self, id: int, head_idx: int):
        ids = self.get_ids(head_idx=head_idx)  # (n_tokens, topk)
        return (ids == id).any(axis=1).sum() / ids.shape[0]  # percentage of tokens with retrieval from id

def retrieval_track(args, n_heads: int = 32, topk: int = 4) -> List[PredictionWithRetrieval]:
    tokenizer = AutoTokenizer.from_pretrained('google/t5-xl-lm-adapt')
    pwrs: List[PredictionWithRetrieval] = []
    pwr = PredictionWithRetrieval(n_heads=n_heads, topk=topk, tokenizer=tokenizer, use_tokenizer=False)
    with open(args.inp_file, 'r') as fin, open(args.inp_file.replace('.txt', '.tsv'), 'w') as fout:
        tsv_writer = csv.writer(fout, delimiter='\t')
        for l in tqdm(fin):
            if l.strip() == '':
                pwrs.append(pwr)
                pwr = PredictionWithRetrieval(n_heads=n_heads, topk=topk, tokenizer=tokenizer, use_tokenizer=False)
                #tsv_writer.writerow([])
            else:
                pwr.add_one_word(l)
                #tsv_writer.writerow(l)
    return pwrs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=['eli5', 'retrieval_track'])
    parser.add_argument('--inp_file', type=str, default=None, help='input file')
    parser.add_argument('--out_file', type=str, default=None, help='output file')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    if args.task == 'eli5':
        prep_eli5(args, evidence_method='answer', skip_answer_as_evidence=False)
    
    elif args.task == 'retrieval_track':
        n_heads = 32
        topk = 4
        pwrs = retrieval_track(args, n_heads=n_heads, topk=topk)
        print(f'total number of examples {len(pwrs)}')
        for head_idx in range(n_heads):
            print(head_idx, np.mean([pwr.get_ids_portion(i, head_idx) for i, pwr in enumerate(pwrs)]))
