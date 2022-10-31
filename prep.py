from typing import List, Tuple, Any, Union, Dict
import argparse
import random
import os
import json
from collections import defaultdict
import csv
from tqdm import tqdm
import numpy as np
import torch
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

def prep_kilt(
    output_file: str, 
    dataset_name: str,
    split: str = 'validation', 
    evidence_method: str = 'provenance', 
    whole_paragraph_as_evidence: bool = False,
    skip_answer_as_evidence: bool = True,
    num_negative_evidence: int = 0,
    subsample: int = 0,
    output_format: str = 'translation'):
    assert dataset_name in {'eli5', 'wow'}
    assert evidence_method in {'provenance', 'self_provenance', 'answer', 'self_answer'}
    assert output_format in {'translation', 'dpr'}

    data = load_dataset('kilt_tasks', name=dataset_name)
    wikipedia = Wikipedia()
        
    formatteds: List[Tuple] = []
    for i, example in tqdm(enumerate(data[split]), desc='format data'):
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
                        prov = wikipedia.get_provenance(wiki_id, ps, pe, cs, ce, whole_paragraph=whole_paragraph_as_evidence)  # always use the whole paragraph
                        evidences.append(prov)
                if 'answer' in evidence_method and ans:
                    evidences.append(ans)
        
        if len(evidences) <= 0 and num_negative_evidence:  # remove examples without ctx
            continue
        formatteds.append((inp, evidences, answer))
    
    print(f'#examples {len(formatteds)}')

    if num_negative_evidence:
        for i, example in tqdm(enumerate(formatteds), desc='generate negatives'):
            inds = np.random.choice(len(formatteds), num_negative_evidence + 1, replace=False)
            negs = [evi for ind in inds if ind != i for evi in formatteds[ind][1]]
            random.shuffle(negs)
            negs = negs[:num_negative_evidence]
            assert len(negs) == num_negative_evidence
            formatteds[i] = formatteds[i] + (negs,)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_format == 'translation':
        qa_file = f'{output_file}_qa.json'
        prov_file = f'{output_file}_evidence.json'
        with open(qa_file, 'w') as qfin, open(prov_file, 'w') as pfin:
            for inp, evidences, answer in formatted:
                # write qa pairs
                qfin.write(json.dumps({'translation': {'en': inp, 'zh': answer}}) + '\n')
                # write evidences
                if 'self' in evidence_method:
                    for evi in evidences:
                        pfin.write(json.dumps({'translation': {'en': inp, 'zh': answer, 'decoder_prefix': evi}}) + '\n')
                else:
                    for evi in evidences:
                        pfin.write(json.dumps({'translation': {'en': inp, 'zh': evi}}) + '\n')
    elif output_format == 'dpr':
        assert num_negative_evidence, 'dpr format requires negative evidence'
        if subsample:
            subsample = min(subsample, len(formatteds))
            dpr_file = f'{output_file}_dpr.{subsample}.json'
            inds = np.random.choice(len(formatteds), subsample, replace=False)
        else:
            dpr_file = f'{output_file}_dpr.json'
            inds = range(len(formatteds))
        dpr_data = [{
            'question': formatteds[ind][0],
            'answers': [formatteds[ind][2]],
            'ctxs': [{'title': '', 'text': ctx, 'score': 1} for ctx in formatteds[ind][1]] + [{'title': '', 'text': ctx, 'score': 0} for ctx in formatteds[ind][3]]
        } for ind in inds]
        with open(dpr_file, 'w') as fout:
            json.dump(dpr_data, fout, indent=True)
    else:
        raise NotImplementedError

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
    with open(args.inp, 'r') as fin, open(args.inp.replace('.txt', '.tsv'), 'w') as fout:
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

def shuffle_evidence(inp_file: str, out_file: str):
    data: List[Dict] = []
    evis: List[str] = []
    with open(inp_file, 'r') as fin, open(out_file, 'w') as fout:
        for l in fin:
            data.append(json.loads(l))
            evis.append(data[-1]['translation']['decoder_prefix'])
        random.shuffle(evis)
        for example, evi in zip(data, evis):
            example['translation']['decoder_prefix'] = evi
            fout.write(json.dumps(example) + '\n')

def head_analysis(attn_file: str, rank: bool = True, show_n_heads: int = 5):
    attensions: torch.FloatTensor = torch.load(attn_file)  # (n_heads, n_examples, n_docs)
    sorted, indices = torch.sort(attensions, dim=-1, descending=True)  # (n_heads, n_examples, n_docs)
    top1_acc = indices[:, :, 0].eq(0).float().mean(-1).numpy()  # (n_heads)
    rank = np.argsort(-top1_acc)[:show_n_heads]
    print('\t'.join(map(str, rank)))
    print('\t'.join(map(str, top1_acc[rank])))

def retrieval_acc(filename: str):
    ret_ids: List[int] = []
    with open(filename, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l.startswith('||'):
                ret_ids.append(int(l[2:]))
    ret_ids = np.array(ret_ids)
    print(f'len {len(ret_ids)}')
    acc = np.mean(ret_ids == np.arange(len(ret_ids)))
    print(acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=['kilt', 'retrieval_track', 'head_analysis', 'shuffle_evidence', 'retrieval_acc'])
    parser.add_argument('--inp', type=str, default=None, help='input file')
    parser.add_argument('--out', type=str, default=None, help='output file')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    if args.task == 'kilt':
        prep_kilt(
            output_file=args.out, 
            dataset_name='wow', 
            split='train', 
            evidence_method='self_provenance', 
            whole_paragraph_as_evidence=False, 
            skip_answer_as_evidence=True,
            num_negative_evidence=9,
            subsample=10000,
            output_format='dpr')

    elif args.task == 'retrieval_track':
        n_heads = 32
        topk = 4
        pwrs = retrieval_track(args, n_heads=n_heads, topk=topk)
        print(f'total number of examples {len(pwrs)}')
        for head_idx in range(n_heads):
            print(head_idx, np.mean([pwr.get_ids_portion(i, head_idx) for i, pwr in enumerate(pwrs)]))
    
    elif args.task == 'head_analysis':
        head_analysis(args.inp)
    
    elif args.task == 'shuffle_evidence':
        shuffle_evidence(args.inp, args.out)
    
    elif args.task == 'retrieval_acc':
        retrieval_acc(args.inp)
