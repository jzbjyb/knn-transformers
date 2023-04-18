from typing import List, Tuple, Any, Union, Dict, Set, Callable
import argparse
import random
import os
import functools
import json
import time
import glob
from collections import defaultdict
import csv
import copy
import evaluate
import re
import logging
from urllib.parse import unquote
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from models.datasets import HotpotQA, WikiMultiHopQA, WikiSum, WikiAsp, ELI5, WoW, ASQA, LMData, MMLU
from models.templates import ApiReturn


class Wikipedia(object):
    def __init__(self):
        from kilt.knowledge_source import KnowledgeSource
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
        return ' '.join(prov).strip(), page['text']

def prep_kilt(
    output_file: str,
    dataset_name: str,
    split: str = 'validation',
    evidence_method: str = 'provenance',
    whole_paragraph_as_evidence: bool = False,
    skip_answer_as_evidence: bool = True,
    remove_wo_ctx: bool = True,
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
                        evidences.append(prov.strip())
                if 'answer' in evidence_method and ans:
                    evidences.append(ans)

        if len(evidences) <= 0 and remove_wo_ctx:  # remove examples without ctx
            continue
        formatteds.append((inp, evidences, answer))

    print(f'#examples {len(formatteds)}')

    if num_negative_evidence:
        # collect unique evidences
        evidence2id = {}
        for example in formatteds:
            for evi in example[1]:
                if evi not in evidence2id:
                    evidence2id[evi] = len(evidence2id)
        evidences = list(evidence2id.keys())  # TODO: make sure the dict is ordered

        for i, example in tqdm(enumerate(formatteds), desc='generate negatives'):
            gold_inds: Set[int] = set(evidence2id[evi] for evi in example[1])
            to_sample_count = min(num_negative_evidence + len(gold_inds), len(evidences))
            to_keep_count = min(num_negative_evidence, len(evidences) - len(gold_inds))
            sample_inds: Set[int] = set(np.random.choice(len(evidences), to_sample_count, replace=False).tolist())
            keep_inds: List[int] = list(sample_inds - gold_inds)
            random.shuffle(keep_inds)
            keep_inds = keep_inds[:to_keep_count]
            assert len(keep_inds) == to_keep_count
            formatteds[i] = formatteds[i] + ([evidences[ind] for ind in keep_inds],)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_format == 'translation':
        assert num_negative_evidence == 0

        if subsample:
            subsample = min(subsample, len(formatteds))
            qa_file = f'{output_file}_qa.{subsample}.json'
            prov_file = f'{output_file}_evidence.{subsample}.json'
            inds = np.random.choice(len(formatteds), subsample, replace=False)
        else:
            qa_file = f'{output_file}_qa.json'
            prov_file = f'{output_file}_evidence.json'
            inds = range(len(formatteds))

        with open(qa_file, 'w') as qfin, open(prov_file, 'w') as pfin:
            for ind in inds:
                inp, evidences, answer = formatteds[ind]
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


def add_ref_to_kilt(
    dataset_name: str,
    split: str = 'validation',
    output_file: str = None,
    skip_empty_ctx: bool = False):
    assert dataset_name in {'eli5', 'wow'}

    data = load_dataset('kilt_tasks', name=dataset_name)[split]
    wikipedia = Wikipedia()

    provnum2count: Dict[int, int] = defaultdict(lambda: 0)
    examples: List[Tuple[int, Dict]] = []
    with open(output_file, 'w') as fout:
        for i, example in tqdm(enumerate(data), desc='format data'):
            # collect refs
            provnum = 0
            ans = None
            for out in example['output']:
                ans = out['answer'].strip() if ans is None else ans
                for prov in out['provenance']:
                    wiki_id = prov['wikipedia_id']
                    ps, pe, cs, ce = prov['start_paragraph_id'], prov['end_paragraph_id'], prov['start_character'], prov['end_character']
                    evi, paras = wikipedia.get_provenance(wiki_id, ps, pe, cs, ce, whole_paragraph=True)
                    prov['wikipedia_paragraphs'] = paras
                    prov['wikipedia_evidence'] = evi
                    provnum += 1
            provnum2count[provnum] += 1
            if skip_empty_ctx and provnum <= 0:
                continue
            examples.append((len(ans), example))

        examples = sorted(examples, key=lambda x: -x[0])
        for _, example in examples:
            fout.write(json.dumps(example) + '\n')
    print('provnum2count', [(k, v / len(data)) for k, v in provnum2count.items()])


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

def retrieval_acc(filename: str, format: str = 'out'):
    assert format in {'out', 'pt'}
    if format == 'pt':
        return retrieval_acc_pt(filename)
    ret_ids: List[int] = []
    with open(filename, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l.startswith('||'):
                ret_ids.append(int(l.split('||')[-1]))
    ret_ids = np.array(ret_ids)
    print(f'len {len(ret_ids)}')
    acc = np.mean(ret_ids == np.arange(len(ret_ids)))
    print(acc)

def retrieval_acc_pt(filename: str, method: str = 'str'):
    examples = [json.loads(l) for l in open('data/wow/val_astarget_selfprov_evidence.json', 'r')]
    examples_dedup = [json.loads(l) for l in open('data/wow/val_astarget_selfprov_evidence.dedup.json', 'r')]
    assert method in {'index', 'str'}
    head2ids = torch.load(filename, map_location='cpu')
    head_accs: List[Tuple[int, float]] = []
    for head, ids in head2ids.items():
        if method == 'index':
            acc = ids.eq(torch.arange(ids.size(0)).unsqueeze(-1)).any(-1).float().mean(0).item()
        elif method == 'str':
            acc = np.mean([examples_dedup[did]['translation']['decoder_prefix'].strip() == examples[i]['translation']['decoder_prefix'].strip()
                for i, did in enumerate(ids[:, 0])])
        head_accs.append((head, acc))
        print(f'{head}: {ids.size(0)} {acc}')

    head_accs = sorted(head_accs, key=lambda x: (-x[1], x[0]))
    print('ranked')
    for i in range(min(len(head_accs), 5)):
        print(f'{head_accs[i][0]}: {head_accs[i][1]}')

def save_beir_format(
    beir_dir: str,
    qid2dict: Dict[str, Dict] = None,
    did2dict: Dict[str, Dict] = None,
    split2qiddid: Dict[str, List[Tuple[str, str]]] = None):
    # save
    os.makedirs(beir_dir, exist_ok=True)
    if qid2dict is not None:
        with open(os.path.join(beir_dir, 'queries.jsonl'), 'w') as fout:
            for qid in qid2dict:
                fout.write(json.dumps(qid2dict[qid]) + '\n')
    if did2dict is not None:
        with open(os.path.join(beir_dir, 'corpus.jsonl'), 'w') as fout:
            for did in tqdm(did2dict, desc='save corpus'):
                fout.write(json.dumps(did2dict[did]) + '\n')
    if split2qiddid is not None:
        os.makedirs(os.path.join(beir_dir, 'qrels'), exist_ok=True)
        for split in split2qiddid:
            with open(os.path.join(beir_dir, 'qrels', f'{split}.tsv'), 'w') as fout:
                fout.write('query-id\tcorpus-id\tscore\n')
                for qid, did in split2qiddid[split]:
                    fout.write(f'{qid}\t{did}\t1\n')

def translation_to_beir(
    translation_file: str,
    beir_dir: str,
    split: str = 'dev',
    dedup_question: bool = False,
    dedup_doc: bool = False):
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    question2qid: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with open(translation_file, 'r') as fin:
        for l in fin:
            example = json.loads(l)['translation']
            question = example['en'].strip()
            answer = example['zh'].strip()
            evidence = example['decoder_prefix'].strip()
            if dedup_doc and evidence in doc2did:
                did = doc2did[evidence]
            else:
                did = f'{str(len(qid2dict) + len(did2dict))}'
                did2dict[did] = {'_id': did, 'title': '', 'text': evidence}
                doc2did[evidence] = did
            if dedup_question and question in question2qid:
                qid = question2qid[question]
            else:
                qid = f'{str(len(qid2dict) + len(did2dict))}'
                qid2dict[qid] = {'_id': qid, 'text': question, 'metadata': {'answer': answer, 'docid': did}}
                question2qid[question] = qid
            split2qiddid[split].append((qid, did))
    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)

def convert_fid_to_beir(
    fid_file: str,
    beir_dir: str,
    split='dev'):
    with open(fid_file, 'r') as fin:
        data = json.load(fin)
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for example in data:
        qid: str = example['id']
        question: str = example['question']
        answer: List[List[str]] = example['answers']
        qid2dict[qid] = {'_id': qid, 'text': question, 'metadata': {'answer': answer}}
        for i, ctx in enumerate(example['ctxs']):
            did = ctx['id']
            did2dict[did] = {'_id': did, 'title': ctx['title'], 'text': ctx['text']}
            if i == 0:
                split2qiddid[split].append((qid, did))
    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)

def use_answer_as_query_in_beir(
    beir_dir: str,
    out_dir: str,
    truncate_to: int = None,
    tokenizer: AutoTokenizer = None):
    from_query_file = os.path.join(beir_dir, 'queries.jsonl')
    to_query_file = os.path.join(out_dir, 'queries.jsonl')

    # query file
    os.makedirs(out_dir, exist_ok=True)
    with open(from_query_file, 'r') as fin, open(to_query_file, 'w') as fout:
        for l in fin:
            example = json.loads(l)
            ans = example['metadata']['answer']
            if truncate_to:
                ans = tokenizer.decode(tokenizer.encode(ans, add_special_tokens=False)[:truncate_to])
            example['metadata']['original_text'] = example['text']
            example['text'] = ans
            fout.write(json.dumps(example) + '\n')

    # corpus and qrel
    rel_dir = os.path.relpath(beir_dir, out_dir)
    os.symlink(os.path.join(rel_dir, 'corpus.jsonl'), os.path.join(out_dir, 'corpus.jsonl'))
    os.symlink(os.path.join(rel_dir, 'qrels'), os.path.join(out_dir, 'qrels'))

class BEIRDataset:
    def __init__(self, root_dir: str, name: str):
        self.name = name
        self.qid2answer, self.qid2meta = self.load_query(os.path.join(root_dir, 'queries.jsonl'))

    @classmethod
    def get_answer_wow(cls, metadata: Dict) -> List[str]:
        return [metadata['answer']]

    @classmethod
    def get_answer_eli5(cls, metadata: Dict) -> List[str]:
        return [metadata['answer']]

    @classmethod
    def get_answer_wikisum(cls, metadata: Dict) -> List[str]:
        return [metadata['summary']]

    @classmethod
    def get_answer_strategyqa(cls, metadata: Dict) -> List[str]:
        return [metadata['answer']]

    @classmethod
    def get_answer_strategyqa_cot(cls, metadata: Dict) -> List[str]:
        a = metadata['answer']
        cot = metadata['cot']
        return [f'{cot} Therefore, the final answer is {a}.']

    @classmethod
    def get_answer_wiki103(cls, metadata: Dict) -> List[str]:
        return [metadata['continue']]

    def load_query(self, filename: str):
        qid2meta: Dict[str, Dict] = {}
        qid2answer: Dict[str, Any] = {}
        with open(filename, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                id, text, metadata = l['_id'], l['text'], l['metadata']
                qid2meta[id] = metadata
                ans = getattr(self, f'get_answer_{self.name}')(metadata)
                if ans is None:
                    continue
                qid2answer[id] = ans
        return qid2answer, qid2meta

def convert_beir_to_fid_format(
    beir_dir: str,
    out_dir: str,
    dataset_name: str,
    splits: List[str],
    topk: int = 100,
    add_self: bool = False,
    add_self_to_the_first: bool = False,
    add_qrel_as_answer: str = None):

    assert add_qrel_as_answer in {None, 'title', 'text'}
    clean_text_for_tsv = lambda x: '' if x is None else x.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    beir_data = BEIRDataset(beir_dir, name=dataset_name)

    # build index
    hostname = 'localhost'
    number_of_shards = 1  # TODO
    corpus, _, _ = GenericDataLoader(data_folder=beir_dir).load(split=splits[0])
    model = BM25Search(index_name=dataset_name, hostname=hostname, initialize=True, number_of_shards=number_of_shards)
    model.index(corpus)
    time.sleep(5)

    for split_ind, split in enumerate(splits):
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split=split)

        # retrieve
        model = BM25Search(index_name=dataset_name, hostname=hostname, initialize=False, number_of_shards=number_of_shards)
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus, queries)
        print(f'retriever evaluation for k in: {retriever.k_values}')
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        print(ndcg, _map, recall, precision)

        # output
        os.makedirs(out_dir, exist_ok=True)
        if split_ind == 0:
            with open(os.path.join(out_dir, 'psgs.tsv'), 'w') as fout, \
                open(os.path.join(out_dir, 'line2docid.tsv'), 'w') as l2dfout:
                fout.write('id\ttext\ttitle\n')
                for lid, did in enumerate(corpus):
                    title = clean_text_for_tsv(corpus[did].get('title'))
                    text = clean_text_for_tsv(corpus[did].get('text'))
                    assert '\n' not in title and '\n' not in text
                    fout.write(f'{did}\t{text}\t{title}\n')
                    l2dfout.write(f'{lid}\t{did}\n')

        examples: List[Dict] = []
        for qid, scores_dict in results.items():
            if add_qrel_as_answer:
                answer = [corpus[did].get(add_qrel_as_answer) for did, rel in qrels[qid].items() if rel]
            else:
                answer = beir_data.qid2answer[qid]
            if 'original_text' in beir_data.qid2meta[qid]:  # for wow dataset the query is the answer while the real query is in original_text
                query = beir_data.qid2meta[qid]['original_text']
            else:
                query = queries[qid]
            example = {'question': query, 'id': qid, 'answers': answer, 'ctxs': []}
            scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[:topk]
            if add_self:
                if beir_data.qid2meta[qid]['docid'] not in set(x[0] for x in scores):  # self doc not retrieved
                    scores.insert(0, (beir_data.qid2meta[qid]['docid'], scores[0][1] + 1.0))  # highest score
                    scores = scores[:topk]
                elif add_self_to_the_first:  # move this doc to the first position
                    pos = [x[0] for x in scores].index(beir_data.qid2meta[qid]['docid'])
                    if pos != 0:
                        ori = scores[0]
                        scores[0] = scores[pos]
                        scores[pos] = ori
            for rank in range(len(scores)):
                did = scores[rank][0]
                title = corpus[did].get('title')
                if add_self and did == beir_data.qid2meta[qid]['docid'] and 'context' in beir_data.qid2meta[qid]:  # avoid leaking
                    text = beir_data.qid2meta[qid]['context']
                else:
                    text = corpus[did].get('text')
                example['ctxs'].append({'id': did, 'title': title, 'text': text})
            examples.append(example)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(examples, open(os.path.join(out_dir, f'{split}.json'), 'w'), indent=2)

def dedup_translation(inp_file: str, out_file: str, dedup_field: str = 'decoder_prefix'):
    fields: Set[str] = set()
    with open(inp_file, 'r') as fin, open(out_file, 'w') as fout:
        for l in fin:
            example = json.loads(l)
            field = example['translation'][dedup_field]
            if field in fields:
                continue
            fields.add(field)
            fout.write(json.dumps(example) + '\n')

def layerhead(pt_file: str, transform: Callable = lambda x: x, topk: int = 10):
    for key in ['aggsmean', 'aggnormsmean']:
        if key in pt_file:
            tau = pt_file[pt_file.find(key) + len(key):].split('_', 1)[0]
            tau = float(tau[:1] + '.' + tau[1:])
            print(f'agg {key} with tau {tau}')
            transform = lambda x: torch.softmax(x.view(-1) / tau, -1).view(*x.size())
            break
    weights = torch.load(pt_file, map_location='cpu')
    lh_weight = [v for k, v in weights.items() if 'layerhead_weight' in k][0]
    try:
        lh_bias = [v for k, v in weights.items() if 'layerhead_bias' in k][0]
    except:
        lh_bias = None
    num_layers, num_heads = lh_weight.size()
    print('#layers', num_layers, '#heads', num_heads)
    for lh in [lh_weight, lh_bias]:
        if lh is None:
            continue
        lh = transform(lh)
        print('original', lh)
        indices = torch.sort(lh.view(-1), descending=True).indices[:topk]
        layer_indices, head_indices = indices // num_heads, indices % num_heads
        print('topk')
        for i, (li, hi) in enumerate(zip(layer_indices, head_indices)):
            print(f'{i} | {li.item()}, {hi.item()}: {lh[li, hi].item()}')

def split_ctxs(json_file: str, out_file: str):
    with open(json_file, 'r') as fin:
        data = json.load(fin)
        new_data = []
        for example in data:
            for ctx in example['ctxs']:
                _example = copy.deepcopy(example)
                _example['ctxs'] = [ctx]
                new_data.append(_example)
    with open(out_file, 'w') as fout:
        json.dump(new_data, fout, indent=True)

def convert_beir_corpus_to_translation(beir_corpus_file: str, out_file: str):
    with open(beir_corpus_file, 'r') as fin, open(out_file, 'w') as fout:
        for l in fin:
            l = json.loads(l)
            t = {'translation': {'en': '', 'zh': l['text']}}
            fout.write(json.dumps(t) + '\n')

def compare_logprob(
    file: str,
    sys_files: List[str],
    beir_dir: str,
    beir_split: str = 'dev'):

    corpus, queries, qrels = GenericDataLoader(
        data_folder=beir_dir).load(split=beir_split)

    data = torch.load(file, map_location='cpu')
    sys_datas = [torch.load(f, map_location='cpu') for f in sys_files]

    value_sym = '!!'
    categories = ['imp', '~imp', 'rel', '~rel', 'imp&rel', '~imp&rel', 'imp&~rel', '~imp&~rel']
    categories_values = [f'{c}{value_sym}' for c in categories]
    sys_reports = [{k: [] for k in categories + categories_values} for _ in sys_files]

    for i in range(len(data['labels'])):
        labels = data['labels'][i]  # (bs, seq_len)
        mask = labels != -100  # (bs, seq_len)
        logprobs = data['logprobs'][i]  # (bs, seq_len)
        probs = logprobs[mask].exp()  # (n_tokens)

        for data_sys, sys_report in zip(sys_datas, sys_reports):
            qids = data_sys['qids'][i]  # (bs)
            dids = data_sys['docids'][i]  # (bs, seq_len - 1, n_ctxs) bos doesn't require retrieval
            dids = torch.cat([torch.full(size=(dids.size(0), 1, dids.size(2)), fill_value=-1).to(dids), dids], 1)  # (bs, seq_len, n_ctxs)

            # convert doc ids to binary relevance
            rel = []
            for qid, _dids in zip(qids, dids):
                rel_dids = torch.tensor([int(k) for k, r in qrels[str(qid.item())].items() if r])
                rel.append(torch.isin(_dids, rel_dids))  # (seq_len, n_ctxs)
            rel = torch.stack(rel, 0)  # (bs, seq_len, n_ctxs)
            rel = rel.any(-1)  # (bs, seq_len)
            rel = rel[mask]  # (n_tokens)

            # compute prob margin
            logprobs_sys = data_sys['logprobs'][i]  # (bs, seq_len)
            probs_sys = logprobs_sys[mask].exp()  # (n_tokens)
            probs_sys = probs_sys - probs
            imp = probs_sys >= 0

            # categorize
            for c in categories:
                cate_mask = eval(c)
                sys_report[c].append(cate_mask.float().mean().item())
                sys_report[f'{c}{value_sym}'].append(probs_sys[cate_mask].float().mean().item())

    # aggregate over batches
    for resport in sys_reports:
        for k in resport:
            resport[k] = np.mean(resport[k])

    print('\t'.join(map(lambda x: '{:>10s}'.format(x), categories)))
    print('\t'.join(map(lambda x: '{:>10s}'.format(x), categories_values)))
    print()
    for sr in sys_reports:
        print('\t'.join(map(lambda x: '{:>10.4f}'.format(sr[x]), categories)))
        print('\t'.join(map(lambda x: '{:>10.4f}'.format(sr[x]), categories_values)))
        print()


def summary_to_beir(
    file_pattern: str,
    spm_path: str,
    beir_dir: str,
    split: str = 'dev',
    dedup_doc: bool = True,
    dedup_question: bool = True,
    max_num_examples: int = None,
):
    import sentencepiece
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(spm_path)

    files = glob.glob(file_pattern)

    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    question2qid: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for file in tqdm(files):
        data = torch.load(file)
        for example in data:
            if 'src_str' in example:
                docs = example['src_str']
            else:
                docs = [spm.decode_ids(doc).strip() for doc in example['src']]
            question = spm.decode_ids(example['query']).rstrip('<T>').strip()
            summary = example['tgt_str'].strip()

            if dedup_question and question in question2qid:
                qid = question2qid[question]
            else:
                qid = str(len(qid2dict))
                qid2dict[qid] = {'_id': qid, 'text': question, 'metadata': {'summary': summary}}
                question2qid[question] = qid

            for doc in docs:
                if dedup_doc and doc in doc2did:
                    did = doc2did[doc]
                else:
                    did = str(len(did2dict))
                    did2dict[did] = {'_id': did, 'title': '', 'text': doc}
                    doc2did[doc] = did
                split2qiddid[split].append((qid, did))

    if max_num_examples and max_num_examples < len(qid2dict.keys()):  # downsample
        qids = list(qid2dict.keys())
        random.shuffle(qids)
        qids = set(qids[:max_num_examples])
        for qid in list(qid2dict.keys()):
            if qid not in qids:
                del qid2dict[qid]
        split2qiddid[split] = [(qid, did) for qid, did in split2qiddid[split] if qid in qids]
        dids = set([did for _, did in split2qiddid[split]])
        for did in list(did2dict.keys()):
            if did not in dids:
                del did2dict[did]

    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def summary_to_beir_all(
    train_file_pattern: str,
    dev_file_pattern: str,
    test_file_pattern: str,
    spm_path: str,
    beir_dir: str,
    dedup_doc: bool = True,
    dedup_question: bool = True,
):
    import sentencepiece
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(spm_path)

    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    question2qid: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for file_pattern, split in [(test_file_pattern, 'test'), (dev_file_pattern, 'dev'), (train_file_pattern, 'train')]:
        files = glob.glob(file_pattern)
        for file in tqdm(files, desc=split):
            data = torch.load(file)
            for example in data:
                if 'src_str' in example:
                    docs = [doc.strip() for doc in example['src_str']]
                else:
                    docs = [spm.decode_ids(doc).strip() for doc in example['src']]
                question = spm.decode_ids(example['query']).rstrip('<T>').strip()
                summary = example['tgt_str'].strip()

                if dedup_question and question in question2qid:
                    qid = question2qid[question]
                else:
                    qid = str(len(qid2dict) + len(did2dict))
                    qid2dict[qid] = {'_id': qid, 'text': question, 'metadata': {'summary': summary}}
                    question2qid[question] = qid

                docids: List[str] = []
                for doc in docs:
                    if dedup_doc and doc in doc2did:
                        did = doc2did[doc]
                    else:
                        did = str(len(qid2dict) + len(did2dict))
                        did2dict[did] = {'_id': did, 'title': '', 'text': doc}
                        doc2did[doc] = did
                    docids.append(did)
                    split2qiddid[split].append((qid, did))

                qid2dict[qid]['metadata']['ctx_ids'] = docids

    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def compare(
        file1: str,
        file2: str,
        only_show_diff: bool = False,
        only_first_right: bool = False,
        show_final: bool = False,
        only_last_trace: bool = False,
        use_raw_data: bool = True,
    ):
    raw_data = json.load(open('data/asqa/ASQA.json'))
    get_ans = lambda x: x.strip().rsplit(' ', 1)[-1][:-1].lower()
    with open(file1) as fin1, open(file2) as fin2:
        id2examples1 = [json.loads(l) for l in fin1]
        id2examples2 = [json.loads(l) for l in fin2]
        id2examples1 = {e['qid']: e for e in id2examples1}
        id2examples2 = {e['qid']: e for e in id2examples2}

        for _id in id2examples1:
            example1 = id2examples1[_id]
            example2 = id2examples2[_id]

            q = example1['question']
            c = example1['ctxs'] if 'ctxs' in example1 else []
            a = example1['gold_output'] if 'gold_output' in example1 else example1['answer']
            a2 = example2['gold_output'] if 'gold_output' in example2 else example2['answer']
            #assert example1['question'] == example2['question'], f"{example1['question']}\n{example2['question']}"
            o1 = example1['output']
            o2 = example2['output']

            r1 = example1['retrieval']
            r2 = example2['retrieval']
            ts1 = example1['trace'] if 'trace' in example1 else []
            ts2 = example2['trace'] if 'trace' in example2 else []

            if show_final:
                #r1 = r1[-1:] if r1 else r1
                #r2 = r2[-1:] if r2 else r2
                ts1 = (ts1[-1:] if only_last_trace else ts1) if ts1 else ts1
                ts2 = (ts2[-1:] if only_last_trace else ts2) if ts2 else ts2

            o1a = get_ans(o1)
            o2a = get_ans(o2)
            ga = get_ans(a)

            if only_first_right:
                show = only_first_right and o1a == ga and o2a != ga
            elif only_show_diff:
                show = only_show_diff and o1 != o2
            else:
                show = False

            if show:
                print('^' * 100)
                for i, t1 in enumerate(ts1):
                    if type(t1) is str:
                        t1p, t1r = t1, None
                    else:
                        t1p, t1r = t1
                    print('-' * 30)
                    print(f'1.{i}->', t1p)
                    print(f'1.{i}->', t1r)
                print('')

                print('^' * 100)
                for i, t2 in enumerate(ts2):
                    if type(t2) is str:
                        t2p, t2r = t2, None
                    else:
                        t2p, t2r = t2
                    print('-' * 30)
                    print(f'2.{i}->', t2p)
                    print(f'2.{i}->', t2r)
                print('')

                print('^' * 100)
                print('ID->', _id)
                print('Q1->', q)
                print('Q2->', example2['question'])
                print('C->', c)
                print('A->', a)
                if use_raw_data:
                    print('subQ->', '\n', '\n'.join([qa['question'] + '     ' + ', '.join(qa['short_answers']) for qa in raw_data['dev'][_id]['qa_pairs']]))
                print('')

                print('-' * 30)
                print('1->', r1)
                print('1->', o1)
                print('')

                print('-' * 30)
                print('2->', r2)
                print('2->', o2)
                input('')


def kilt_to_beir(
    kilt_wiki_file: str,
    beir_dir: str,
    dedup_doc: bool = False,
    split: str = 'dev',
):
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    qid2dict['0'] = {'_id': '0', 'text': 'dummy'}
    with open(kilt_wiki_file, 'r') as fin:
        for l in tqdm(fin, desc='process kilt'):
            example = json.loads(l)
            title = example['wikipedia_title']
            wiki_id = example['wikipedia_id']
            for para_ind, para in enumerate(example['text']):
                if dedup_doc and para in doc2did:
                    did = doc2did[para]
                else:
                    did = str(len(did2dict))
                    did2dict[did] = {'_id': did, 'title': title, 'text': para, 'metadata': {'wiki_id': wiki_id, 'para_index': para_ind}}
                    doc2did[para] = did
                if len(split2qiddid) == 0:
                    split2qiddid[split].append(('0', did))
        save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def dpr_to_beir(
    dpr_file: str,
    beir_dir: str,
    split: str = 'dev',
):
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    qid2dict['0'] = {'_id': '0', 'text': 'dummy'}
    with open(dpr_file, 'r') as fin:
        reader = csv.reader(fin, delimiter='\t')
        header = next(reader)
        for row in tqdm(reader, desc='process dpr'):
            did, text, title = row
            did2dict[did] = {'_id': did, 'title': title, 'text': text}
            if len(split2qiddid) == 0:
                split2qiddid[split].append(('0', did))
        save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def strategyqa_to_beir(
    strategyqa_file: str,
    beir_dir: str,
    prompt_file: str = None,
    split: str = 'dev',
    dedup_question: bool = True,
    dedup_doc: bool = True,
):
    question2cot: Dict[str, str] = {}
    if prompt_file:
        with open(prompt_file, 'r') as fin:
            for l in fin:
                question, answer, cot = l.rstrip('\n').split('\t')
                question, answer, cot = question.strip(), answer.strip(), cot.strip()
                question2cot[question] = cot

    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    question2qid: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    with open(strategyqa_file, 'r') as fin:
        data = json.load(fin)
        for example in data:
            question = example['question'].strip()
            assert example['answer'] in {True, False}
            answer = 'yes' if example['answer'] else 'no'

            if dedup_question and question in question2qid:
                qid = question2qid[question]
            else:
                qid = str(len(qid2dict))
                cot = question2cot[question] if prompt_file else None
                qid2dict[qid] = {'_id': qid, 'text': question, 'metadata': {'answer': answer, 'cot': cot}}
                question2qid[question] = qid

            for fact in example['facts']:
                fact = fact.strip()
                if dedup_doc and fact in doc2did:
                    did = doc2did[fact]
                else:
                    did = str(len(did2dict))
                    did2dict[did] = {'_id': did, 'title': '', 'text': fact}
                    doc2did[fact] = did
                split2qiddid[split].append((qid, did))

        if prompt_file:
            assert len(qid2dict) == len(question2cot), '#rationales != #examples'

    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def hotpotqa_to_beir(
    input_file: str,
    beir_dir: str,
    split: str = 'dev',
    dedup_doc: bool = True,
):
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    question2qid: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    if input_file == 'hotpot_qa':
        dataset = load_dataset('hotpot_qa', 'distractor')['validation' if split == 'dev' else 'train']
    else:
        dataset = json.load(open(input_file, 'r'))

    for example_ind in range(len(dataset)):
        example = dataset[example_ind]

        question = example['question'].strip()
        answer = example['answer'].strip()
        answer_id = example['answer_id'] if 'answer_id' in example else None
        qid = example['id'] if 'id' in example else example['_id']
        ctxs: List[Tuple[str, str]] = []

        if type(example['context']) is dict:
            title2paras: Dict[str, List[str]] = dict(zip(example['context']['title'], example['context']['sentences']))
        elif type(example['context']) is list:
            title2paras: Dict[str, List[str]] = {title: sents for title, sents in example['context']}
        else:
            raise NotImplementedError

        if type(example['supporting_facts']) is dict:
            fact_gen = zip(example['supporting_facts']['title'], example['supporting_facts']['sent_id'])
        elif type(example['supporting_facts']) is list:
            fact_gen = example['supporting_facts']
        else:
            raise NotImplementedError

        for title, para_ind in fact_gen:
            if para_ind >= len(title2paras[title]):
                print(qid, 'support fact index oob')
                continue
            doc = title2paras[title][para_ind].strip()
            if dedup_doc and doc in doc2did:
                did = doc2did[doc]
            else:
                did = str(len(did2dict))
                did2dict[did] = {'_id': did, 'title': title, 'text': doc}
                doc2did[doc] = did
            split2qiddid[split].append((qid, did))
            ctxs.append((did, doc))

        qid2dict[qid] = {'_id': qid, 'text': question, 'metadata': {'answer': answer, 'answer_id': answer_id, 'ctxs': ctxs}}
        question2qid[question] = qid

    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def tsv_to_beir(
    tsv_file: str,
    beir_dir: str,
    split: str = 'dev',
    dedup_question: bool = True,
    dedup_doc: bool = True
):
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    question2qid: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with open(tsv_file, 'r') as fin:
        for l in fin:
            context, continual = l.rstrip('\n').split('\t')

            if dedup_question and context in question2qid:
                qid = question2qid[context]
            else:
                qid = str(len(qid2dict))
                qid2dict[qid] = {'_id': qid, 'text': context, 'metadata': {'continue': continual}}
                question2qid[context] = qid

            if dedup_doc and continual in doc2did:
                did = doc2did[continual]
            else:
                did = str(len(did2dict))
                did2dict[did] = {'_id': did, 'title': '', 'text': continual}
                doc2did[continual] = did
            split2qiddid[split].append((qid, did))

    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)

def self_consistency(examples: List[Dict], anchor_text: str):
    if len(examples) == 1:
        return examples[0]
    answer2indices: Dict[str, List[int]] = defaultdict(list)
    for i, example in enumerate(examples):
        pred = example['output'].split('\n\n', 1)[0].strip()
        position = pred.find(anchor_text)
        if position != -1:
            ans = pred[position + len(anchor_text):].strip().lower()
            answer2indices[ans].append(i)
    answer2indices: List[Tuple[str, List[int]]] = sorted(answer2indices.items(), key=lambda x: -len(x[1]) + random.random() / 2)
    if len(answer2indices):
        candidates = answer2indices[0][1]
        return examples[candidates[random.randint(0, len(candidates) - 1)]]
    else:  # all format error
        return examples[random.randint(0, len(examples) - 1)]

def eval(
    model: str,
    dataset: str,
    jsonl_files: List[str],
    anchor_text: List[str] = ['So the answer is'],
    prefix_to_remove: List[str] = [],
    retrieval_percentiles: List[Union[int, float]] = [1, 0.25, 0.5, 0.75, 1.0],
    remove_followup: Tuple[str, str] = ('Follow up[^:]*:', '?'),
    beir_dir: str = None,
    consistency_suffix: str = 'run',
    use_multi_ref: bool = False,
    debug: bool = False,
):
    if not anchor_text:
        anchor_text = []
    anchor_text = anchor_text if type(anchor_text) is list else [anchor_text]
    if beir_dir is not None:
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='dev')
    else:
        corpus = queries = qrels = None

    def add_metric_kvs(metric_dict):
        for k, v in metric_dict.items():
            final_metrics[k] += v

    def choose_reference(example):
        if 'answers' in example and use_multi_ref:  # multiple
            return example['answers']
        return example['gold_output'] if 'gold_output' in example else example['output']

    def choose_final_answer(example):
        if 'answers' in example and use_multi_ref:  # multiple
            answers = example['answers']
        else:
            answers = [example['answer']]
        for i, answer in enumerate(answers):
            for pattern in prefix_to_remove + anchor_text[:1]:
                if not pattern:
                    continue
                find = re.compile(pattern).search(answer)
                if find:
                    answer = find.group(1)
                    answers[i] = answer
                    if dataset == 'strategyqa':
                        answer = answer.lower()
                        assert answer in {'yes', 'no'}
                    elif dataset == 'mmlu':
                        answer = answer.lower()
                        assert answer in {'a', 'b', 'c', 'd', 'e'}
        return answers if len(answers) > 1 else answers[0]

    def choose_full_prediction(example):
        if ApiReturn.no_stop(model=model, dataset=dataset):
            pred = example['output'].strip()
        else:
            pred = example['output'].split('\n\n', 1)[0].strip()
        find = None
        if prefix_to_remove:
            for pattern in prefix_to_remove:
                find = re.compile(pattern).search(pred)
                if find:
                    pred = find.group(1)
                    break
        if find is None:
            logging.warning(f'format error "{pred}"')
        return pred

    def get_final_answer_from_pred(pred: str):
        final_ans = []
        for at in anchor_text:
            find = re.compile(at).search(pred)
            if find:
                final_ans.append(find.group(1))
        return ' '.join(final_ans).strip()

    metric_func = evaluate.load('rouge')

    scount = 0
    search_per_example: List[int] = []
    final_metrics = {k: 0 for k in [
        'correct', 'incorrect', 'wrongformat',
        'f1', 'precision', 'recall',
        'ent_f1', 'ent_precision', 'ent_recall', 'num_ent',
        'avg_nll', 'ppl', 'tokens']}
    ret_accs: List[List[float]] = []
    ret_covers: List[List[float]] = []
    predictions: List[str] = []
    followups: List[str] = []
    references: List[str] = []
    num_steps: List[int] = []
    retrieval_ratios: List[float] = []

    root_file = None
    if len(jsonl_files) > 1:  # consistency
        for jf in jsonl_files:
            assert jf.rsplit('.', 1)[1].startswith(consistency_suffix)
        root_file = jsonl_files[0].rsplit('.', 1)[0]
    examples_all_files = [[json.loads(l) for l in open(jf)] for jf in jsonl_files]
    assert len(set([len(examples) for examples in examples_all_files])) == 1
    total = len(examples_all_files[0])

    consistency_examples: List[Dict] = []
    for i in tqdm(range(total)):
        examples: List[Dict] = [file[i] for file in examples_all_files]

        # aggregate multiple examples with consistency
        example = self_consistency(examples, anchor_text=anchor_text)
        consistency_examples.append(example)

        # get necessary info for evaluation
        trace = example['trace'] if 'trace' in example else []
        qid = example['qid'] if 'qid' in example else example['id']
        question = example['question'] if 'question' in example else None
        ref = choose_reference(example)
        final_ans = choose_final_answer(example)
        ans_id = example['answer_id'] if 'answer_id' in example else None
        pred = choose_full_prediction(example)
        if remove_followup:
            raw_pred = pred
            rms, rme = remove_followup
            pred = re.sub(f'{rms}[^\{rme}]*\{rme}', '', raw_pred)
            fu = ' '.join(re.findall(f'{rms}[^\{rme}]*\{rme}', raw_pred))
            followups.append(fu)
        probs = -np.log(example['output_prob']) if 'output_prob' in example else []
        final_metrics['avg_nll'] += np.mean(probs)
        final_metrics['ppl'] += np.sum(probs)
        final_metrics['tokens'] += len(probs)

        references.append(ref)
        predictions.append(pred)
        num_steps.append(len(trace))
        retrieval_ratios.append(len((example['retrieval'] or []) if 'retrieval' in example else []) / (len(trace) or 1))
        if 'retrieval' in example and example['retrieval']:
            ret_dids = np.array([r if type(r[0]) is str else r[0] for r in example['retrieval']], dtype=np.str_)
        else:
            ret_dids = np.array([['placeholder']], dtype=np.str_)

        pred_ans = get_final_answer_from_pred(pred) if anchor_text else pred
        wrongformat = len(pred_ans) == 0
        if wrongformat:
            final_metrics['wrongformat'] += 1
        else:
            if dataset in {'strategyqa', 'mmlu'}:
                correct = int(final_ans.lower() in pred_ans.lower())
                final_metrics['correct'] += correct
                final_metrics['incorrect'] += 1 - correct
            elif dataset in {'hotpotqa'}:
                add_metric_kvs(HotpotQA.exact_match_score(pred_ans, final_ans))
                add_metric_kvs(HotpotQA.f1_score(pred_ans, final_ans))
            elif dataset in {'2wikihop'}:
                add_metric_kvs(WikiMultiHopQA.exact_match_score(pred_ans, final_ans, ground_truth_id=ans_id))
                add_metric_kvs(WikiMultiHopQA.f1_score(pred_ans, final_ans, ground_truth_id=ans_id))
            elif dataset in {'wikisum'}:
                add_metric_kvs(WikiSum.entity_f1_score(pred_ans, final_ans))
            elif dataset in {'wikiasp'}:
                add_metric_kvs(WikiAsp.entity_f1_score(pred_ans, final_ans))
            elif dataset in {'eli5'}:
                add_metric_kvs(ELI5.entity_f1_score(pred_ans, final_ans))
            elif dataset in {'asqa'}:
                add_metric_kvs(ASQA.entity_f1_score(pred_ans, final_ans))
            elif dataset in {'wow'}:
                add_metric_kvs(WoW.entity_f1_score(pred_ans, final_ans))
            elif dataset in {'lmdata'}:
                pass
            else:
                raise NotImplementedError

        has_search = '[Search(' in pred
        scount += has_search
        if has_search:
            search_per_example.append(len(re.findall('\[Search\(', pred)))

        if debug and (not correct or wrongformat):
            print('ID->', qid)
            print('Q->', question)
            print()
            print('T->')
            for prompt, cont in trace:
                print(prompt)
                print('->', cont)
                print('\n------------------\n')
            print()
            print('P->', pred)
            print()
            print('G->', ref)
            input()

        # retrieval
        ret_accs.append([])
        ret_covers.append([])
        if ret_dids is not None:
            ret_seq_len = len(ret_dids)
            rel_dids: List[str] = np.array([d for d, r in qrels[qid].items() if r]) if qrels else []
            rels = np.isin(ret_dids, rel_dids).any(-1)  # (ret_seq_len)
            prev_pt = 0
            for pt in retrieval_percentiles:
                if type(pt) is int:
                    pass
                elif type(pt) is float:
                    pt = int(ret_seq_len * pt)
                else:
                    raise NotImplementedError
                if pt <= prev_pt:  # at least one token
                    pt = prev_pt + 1
                ret_accs[-1].append(rels[prev_pt:pt].mean())
                ret_covers[-1].append(len(np.intersect1d(ret_dids[:pt].reshape(-1), rel_dids)) / (len(rel_dids) or 1))
                prev_pt = max(min(pt, ret_seq_len - 1), 0)

    if root_file:
        with open(root_file + '.merge', 'w') as fout:
            for e in consistency_examples:
                fout.write(json.dumps(e) + '\n')

    total = len(predictions)  # change total

    # rouge
    if dataset == 'lmdata':
        metrics = {}
    else:
        metrics = metric_func.compute(predictions=predictions, references=references)
    if remove_followup:
        metrics_followup = metric_func.compute(predictions=followups, references=references)

    ret_accs = np.array(ret_accs, dtype=float).mean(0)
    ret_covers = np.array(ret_covers, dtype=float).mean(0)
    format_list = lambda arr: ', '.join(map(lambda x: '{:.3f}'.format(x), arr.tolist()))
    print('\t'.join(final_metrics.keys()))
    print('\t'.join(map(lambda kv: str(np.exp(kv[1] / final_metrics['tokens'] or 1)) if kv[0] == 'ppl' else str(kv[1] / total), final_metrics.items())))
    print('')

    print('\t'.join(metrics.keys()))
    print('\t'.join(map(str, metrics.values())))
    print('#pred\t#gold\t#examples')
    print(f'{np.mean([len(p) for p in predictions])}\t{np.mean([len(r) if type(r) is str else np.mean([len(_r) for _r in r]) for r in references])}\t{total}')
    print('')

    if remove_followup:
        print('\t'.join(metrics_followup.keys()))
        print('\t'.join(map(str, metrics_followup.values())))
        print('#pred\t#gold')
        print(f'{np.mean([len(p) for p in followups])}\t{np.mean([len(r) for r in references])}')
        print('')

    print('retrieval acc\tcoverage')
    print(f'{format_list(ret_accs)}\t{format_list(ret_covers)}')
    print(f'#examples with search: {scount}, #avg search per example {np.mean(search_per_example)}, #steps {np.mean(num_steps)}, ret ratio {np.mean(retrieval_ratios)}')


def build_elasticsearch(
    beir_corpus_file_pattern: str,
    index_name: str,
    get_id: Callable = None,
):
    beir_corpus_files = glob.glob(beir_corpus_file_pattern)
    print(f'#files {len(beir_corpus_files)}')
    from beir.retrieval.search.lexical.elastic_search import ElasticSearch
    config = {
        "hostname": 'localhost',
        "index_name": index_name,
        "keys": {"title": "title", "body": "txt"},
        "timeout": 100,
        "retry_on_timeout": True,
        "maxsize": 24,
        "number_of_shards": 'default',
        "language": 'english',
    }
    es = ElasticSearch(config)

    # create index
    print(f'create index {index_name}')
    es.delete_index()
    time.sleep(5)
    es.create_index()

    get_id = get_id or (lambda x: str(x['_id']))
    # generator
    def generate_actions():
        for beir_corpus_file in beir_corpus_files:
            with open(beir_corpus_file, 'r') as fin:
                for l in fin:
                    doc = json.loads(l)
                    es_doc = {
                        "_id": get_id(doc),
                        "_op_type": "index",
                        "refresh": "wait_for",
                        config['keys']['body']: doc['text'],
                        config['keys']['title']: doc['title'],
                    }
                    yield es_doc

    # index
    progress = tqdm(unit='docs')
    es.bulk_add_to_index(
        generate_actions=generate_actions(),
        progress=progress)


def mmlu_ret(
        split: str = 'test',
        index_name: str = 'wikipedia_dpr',
        topk: int = 1000,
        output: str = None):
    from models.retriever import BM25
    letters = ['A', 'B', 'C', 'D']
    retriever = BM25(
        tokenizer=None,
        dataset=(None, None, None),
        index_name=index_name,
        use_decoder_input_ids=True,
        engine='elasticsearch',
        file_lock=None)
    topics = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
        'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
        'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
        'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
        'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
        'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
        'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
        'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
        'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
        'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
        'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology',
        'us_foreign_policy', 'virology', 'world_religions']
    with open(output, 'w') as fout:
        for topic in topics:
            dataset = load_dataset('lukaemon/mmlu', topic)[split]
            for i in tqdm(range(len(dataset)), desc=topic):
                q = dataset[i]['input'].strip()
                answer = dataset[i]['target'].strip()
                options = ''
                for letter in letters:
                    options += '(' + letter + ') ' + dataset[i][letter].strip() + ' '
                ctx_ids, ctx_texts = retriever.retrieve_and_prepare(
                    decoder_texts=[q + '\n' + options],
                    topk=topk,
                    max_query_length=None)
                result = {
                    'topic': topic,
                    'split': split,
                    'index': i,
                    'question': q,
                    'options': {letter: dataset[i][letter].strip() for letter in letters},
                    'answer': answer,
                    'docs': [(idx, text) for idx, text in zip(ctx_ids[0].tolist(), ctx_texts[0].tolist())]
                }
                fout.write(json.dumps(result) + '\n')


def prompt_dump(jsonl_file: str, out_file: str):
    with open(jsonl_file, 'r') as fin, open(out_file, 'w') as fout:
        for l in fin:
            example = json.loads(l)
            _id = example['qid']
            prompt, output = example['trace'][0]
            ans = example['answer']
            fout.write(json.dumps({'id': _id, 'prompt': prompt, 'answer': ans}) + '\n')


def kilt_dataset_to_beir(
        dataset: str,
        split: str,
        beir_dir: str):
    raise NotImplementedError
    rawdata = load_dataset('kilt_tasks', name='dataset')[split]

    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    question2qid: Dict[str, Dict] = {}
    doc2did: Dict[str, str] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for i, example in enumerate(rawdata):
        qid = example['id']
        question = example['input']
        answers: List[str] = []
        docs: List[str] = []
        for candidate in example['output']:
            ans = candidate['answer'].strip()
            if ans:
                answers.append(ans)
        assert len(answers) >= 1
        output = self.output_template(cot=None, ans=answers[0])
        dataset.append({
            'qid': qid,
            'question': question,
            'answer': answers[0],
            'answers': answers,
            'gold_output': output,
        })
    return Dataset.from_list(dataset)


def jsonl_to_keyvalue(
    jsonl_file: str,
    keyvalue_file: str,
    prefix_to_remove: List[str],
):
    with open(jsonl_file, 'r') as fin, open(keyvalue_file, 'w') as fout:
        key2output: Dict[str, str] = {}
        for l in fin:
            l = json.loads(l)
            pred = l['output'].strip()
            find = None
            if prefix_to_remove:
                for pattern in prefix_to_remove:
                    find = re.compile(pattern).search(pred)
                    if find:
                        pred = find.group(1)
                        break
            if find is None:
                logging.warning(f'format error "{pred}"')
            key2output[l['qid']] = pred
        json.dump(key2output, fout)


def mmlu_retrieval_usefulness(
        max_num_examples: int = 100,
        batch_size: int = 20,
        topk: int = 3,
        only_use_question: bool = True):
    from models.retriever import BM25
    retriever = BM25(
        tokenizer=None,
        dataset=(None, None, None),
        index_name='wikipedia_dpr',
        use_decoder_input_ids=True,
        engine='elasticsearch',
        file_lock=None)
    tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
        'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
        'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
        'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
        'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
        'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
        'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics',
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
        'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
        'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    task2contain: Dict[str, float] = {}
    for task in tqdm(tasks):
        # load data
        data = MMLU(tasks=[task], prompt_type='cot')
        data.format(fewshot=0)
        data = data.dataset
        if max_num_examples and max_num_examples < len(data):
            data = data.shuffle()
            data = data.select(range(max_num_examples))
        # retrieve
        queries = data['question']
        if only_use_question:
            queries = [q[:q.find('(A)')].strip() for q in queries]
        answers = data['answer_text']
        concat_text = []
        for b in range(0, len(queries), batch_size):
            ctx_ids, ctx_texts = retriever.retrieve_and_prepare(
                decoder_texts=queries[b:b + batch_size], topk=topk, max_query_length=None)
            concat_text.extend([' '.join(texts) for texts in ctx_texts])
        assert len(answers) == len(concat_text)
        # compute contain ratio
        num_contain = 0
        for ans, text in zip(answers, concat_text):
            num_contain += int(ans.lower() in text.lower())
        task2contain[task] = num_contain / len(answers)
        print(f'\nResult: {task}\t{task2contain[task]}\t{num_contain}\t{len(answers)}\n')
    # report
    for task, contain in sorted(task2contain.items(), key=lambda x: -x[1]):
        print(f'{task}\t{contain}')


def  wikiasp_match_title(
    output_dir: str,
    split: str = 'test',
    count_per_domain: int = 100,
    max_query_len: int = 20,
    max_n_toks: int = 400,
    min_n_toks: int = 50,
    min_n_toks_per_asp: int = 10,
    use_site_operator: bool = True,
    ):
    domains = ['album', 'animal', 'artist', 'building', 'company', 'educational_institution',
               'event', 'film', 'group', 'historic_place', 'infrastructure', 'mean_of_transportation',
               'office_holder', 'plant', 'single', 'soccer_player', 'software', 'television_show',
               'town', 'written_work']

    from models.retriever import BM25
    retriever = BM25(
        tokenizer=None,
        dataset=(None, None, None),
        index_name=None,
        use_decoder_input_ids=True,
        engine='bing',
        file_lock=None)

    def get_query(targets: List[Tuple[str, str]]):
        query: List[str] = []
        n_toks = 0
        for asp, text in targets:
            text = text.strip()
            toks = text.split()
            n_toks += len(toks)
            query.extend(toks)
            if len(query) >= max_query_len:
                query = query[:max_query_len]
                break
        query = ' '.join(query)
        if use_site_operator:
            query += ' site:en.wikipedia.org'
        return query
    def get_wiki_url(urls: List[str]):
        for url in urls:
            if 'wikipedia.org' in url:
                return url
        return None
    def wiki_url_to_title(url: str):
        title = ' '.join(unquote(url).rsplit('/wiki/', 1)[-1].split('_')) if url else url
        return title
    domain2success: Dict[str, int] = defaultdict(lambda: 0)
    def map_fn(example, domain: str):
        query = get_query(example['targets'])
        urls, snippets = retriever.retrieve_and_prepare(decoder_texts=[query], topk=50)
        url = get_wiki_url(urls[0])
        title = wiki_url_to_title(url)
        example['query'] = query
        example['url'] = url
        example['title'] = title
        example['domain'] = domain
        domain2success[domain] += int(title is not None)
        return example
    def filter_fn(example):
        targets = [(asp, text) for asp, text in example['targets'] if len(text.strip().split()) >= min_n_toks_per_asp]  # remove empty asp
        is_multi = len(targets) > 1
        n_toks = sum([len(t[1].strip().split()) for t in targets])
        has_len = n_toks <= max_n_toks and n_toks >= min_n_toks
        return is_multi and has_len

    processed = []
    for domain in domains:
        data = load_dataset('wiki_asp', domain)[split]
        data = data.filter(filter_fn)
        if len(data) > count_per_domain:  # downsample
            data = data.shuffle()
            data = data.select(range(count_per_domain))
        data = data.map(functools.partial(map_fn, domain=domain))
        processed.append(data)
    concat_data = concatenate_datasets(processed)
    concat_data.save_to_disk(output_dir)
    print(domain2success)


def wikiasp_corpus(beir_dir: str, paragraph_min_len: int = 100):
    domains = ['album', 'animal', 'artist', 'building', 'company', 'educational_institution',
               'event', 'film', 'group', 'historic_place', 'infrastructure', 'mean_of_transportation',
               'office_holder', 'plant', 'single', 'soccer_player', 'software', 'television_show',
               'town', 'written_work']
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    seen_exids: Set[str] = set()
    did = None
    for domain in domains:
        for split in ['test', 'validation', 'train']:
            data = load_dataset('wiki_asp', domain)[split]
            exids = data['exid']
            li_sents = data['inputs']
            for exid, sents in tqdm(zip(exids, li_sents), desc=f'{domain} {split}'):
                assert exid not in seen_exids, f'{exid} already seen'
                seen_exids.add(exid)
                prev: List[str] = []
                n_toks: int = 0
                start_sent_id: int = 0
                for i, sent in enumerate(sents):
                    sent = sent.strip()
                    n_toks += len(sent.split())
                    prev.append(sent)
                    if n_toks >= paragraph_min_len or (len(prev) and i == len(sents) - 1):
                        did = f'{exid}-{start_sent_id}.{i + 1}'
                        did2dict[did] = {'_id': did, 'title': '', 'text': ' '.join(prev)}
                        prev = []
                        n_toks = 0
                        start_sent_id = i + 1
    qid2dict['0'] =  {'_id': '0', 'text': 'dummy'}
    split2qiddid['dev'].append(('0', did))
    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def annotate_asqa(
    pred_json_file: str = None,
    raw_file: str = None,
    hint_file: str = None,
    out_file: str = None,
    split: str = 'dev'):
    raw_data = json.load(open('data/asqa/ASQA.json'))
    ids = None
    if pred_json_file:
        with open(pred_json_file, 'r') as fin:
            ids = set([json.loads(l)['qid'] for l in fin])
    id2hint = {}
    if hint_file:
        with open(hint_file, 'r') as fin:
            id2hint = {json.loads(l)['qid']: json.loads(l)['output'] for l in fin}
    with open(out_file, 'w') as fout:
        writer = csv.writer(fout, delimiter='\t')
        all_ids = list(raw_data[split].keys())
        random.shuffle(all_ids)
        for qid in all_ids:
            if ids is not None and qid not in ids:
                continue
            example = raw_data[split][qid]
            question = example['ambiguous_question']
            subqs = '\n'.join([qa['question'] + '     ' + ', '.join(qa['short_answers']) for qa in example['qa_pairs']])
            writer.writerow([qid, question, subqs] + ([id2hint[qid]] if qid in id2hint else []))


def annotate_asqa_get_hint(
    tsv_file: str,
    hint_file: str,
    out_file: str,
):
    id2hint = {}
    if hint_file:
        with open(hint_file, 'r') as fin:
            id2hint = {json.loads(l)['qid']: json.loads(l)['output'].strip() for l in fin}
    with open(tsv_file) as fin, open(out_file, 'w') as fout:
        header: List[str] = fin.readline().strip().split('\t')
        reader = csv.reader(fin, delimiter='\t')
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(header)
        for row in reader:
            assert len(row) == len(header)
            row = dict(zip(header, [x.strip() for x in row]))
            row['hint'] = id2hint[row['id']]
            writer.writerow(row.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=[
        'kilt', 'retrieval_track', 'head_analysis', 'shuffle_evidence', 'retrieval_acc',
        'translation_to_beir', 'convert_beir_to_fid_format', 'use_answer_as_query_in_beir',
        'dedup_translation', 'layerhead', 'split_ctxs', 'convert_beir_corpus_to_translation',
        'convert_fid_to_beir', 'compare_logprob', 'summary_to_beir', 'summary_to_beir_all', 'compare',
        'strategyqa_to_beir', 'hotpotqa_to_beir', 'tsv_to_beir', 'eval', 'kilt_to_beir',
        'build_elasticsearch', 'dpr_to_beir', 'mmlu_ret', 'prompt_dump', 'kilt_dataset_to_beir',
        'add_ref_to_kilt', 'jsonl_to_keyvalue', 'mmlu_retrieval_usefulness',
        'wikiasp_match_title', 'wikiasp_corpus', 'annotate_asqa', 'annotate_asqa_get_hint'])
    parser.add_argument('--inp', type=str, default=None, nargs='+', help='input file')
    parser.add_argument('--dataset', type=str, default='asqa', help='input dataset', choices=[
        'strategyqa', 'mmlu', 'hotpotqa', '2wikihop', 'wikisum', 'wikiasp', 'eli5', 'wow', 'asqa', 'lmdata'])
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0301', help='model name',
                        choices=['code-davinci-002', 'gpt-3.5-turbo-0301'])
    parser.add_argument('--out', type=str, default=None, help='output file')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    if args.task == 'kilt':
        prep_kilt(
            output_file=args.out,
            dataset_name='wow',
            split='validation',
            evidence_method='self_provenance',
            whole_paragraph_as_evidence=False,
            skip_answer_as_evidence=True,
            remove_wo_ctx=True,
            num_negative_evidence=10000,
            subsample=0,
            output_format='dpr')

    elif args.task == 'retrieval_track':
        n_heads = 32
        topk = 4
        pwrs = retrieval_track(args, n_heads=n_heads, topk=topk)
        print(f'total number of examples {len(pwrs)}')
        for head_idx in range(n_heads):
            print(head_idx, np.mean([pwr.get_ids_portion(i, head_idx) for i, pwr in enumerate(pwrs)]))

    elif args.task == 'head_analysis':
        head_analysis(args.inp[0])

    elif args.task == 'shuffle_evidence':
        shuffle_evidence(args.inp[0], args.out)

    elif args.task == 'retrieval_acc':
        retrieval_acc(args.inp[0], format='pt')

    elif args.task == 'translation_to_beir':
        translation_file = args.inp[0]
        beir_dir = args.out
        translation_to_beir(translation_file, beir_dir, split='dev', dedup_question=True, dedup_doc=True)

    elif args.task == 'convert_beir_to_fid_format':
        beir_dir = args.inp[0]
        out_dir = args.out
        convert_beir_to_fid_format(
            beir_dir,
            out_dir,
            dataset_name='strategyqa_cot',
            splits=['dev'],
            add_self=False,
            add_self_to_the_first=False,
            topk=100)

    elif args.task == 'use_answer_as_query_in_beir':
        beir_dir = args.inp[0]
        out_dir = args.out
        tokenizer = AutoTokenizer.from_pretrained('google/t5-xl-lm-adapt')
        use_answer_as_query_in_beir(beir_dir, out_dir, truncate_to=8, tokenizer=tokenizer)

    elif args.task == 'dedup_translation':
        inp_file = args.inp[0]
        out_file = args.out
        dedup_translation(inp_file, out_file)

    elif args.task == 'layerhead':
        pt_file = args.inp[0]
        layerhead(pt_file)

    elif args.task == 'split_ctxs':
        json_file = args.inp[0]
        out_file = args.out
        split_ctxs(json_file, out_file)

    elif args.task == 'convert_beir_corpus_to_translation':
        beir_corpus_file = args.inp[0]
        out_file = args.out
        convert_beir_corpus_to_translation(beir_corpus_file, out_file)

    elif args.task == 'convert_fid_to_beir':
        fid_file = args.inp[0]
        beir_dir = args.out
        convert_fid_to_beir(fid_file, beir_dir, split='dev')

    elif args.task == 'compare_logprob':
        compare_logprob(args.inp[0], sys_files=args.inp[1:], beir_dir='data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans')

    elif args.task == 'summary_to_beir':
        file_pattern, spm_path = args.inp
        beir_dir = args.out
        summary_to_beir(file_pattern, spm_path, beir_dir, max_num_examples=None)

    elif args.task == 'summary_to_beir_all':
        root_dir, spm_path = args.inp
        beir_dir = args.out
        train_file_pattern = f'{root_dir}/WIKI.train.*.pt'
        dev_file_pattern = f'{root_dir}/WIKI.valid.*.pt'
        test_file_pattern = f'{root_dir}/WIKI.test.*.pt'
        summary_to_beir_all(
            train_file_pattern=train_file_pattern,
            dev_file_pattern=dev_file_pattern,
            test_file_pattern=test_file_pattern,
            spm_path=spm_path,
            beir_dir=beir_dir)

    elif args.task == 'compare':
        file1, file2 = args.inp
        compare(file1, file2, only_show_diff=True, only_first_right=False, show_final=True)

    elif args.task == 'strategyqa_to_beir':
        strategyqa_file = args.inp[0]
        if len(args.inp) > 1:
            prompt_file = args.inp[1]
        beir_dir = args.out
        strategyqa_to_beir(strategyqa_file, beir_dir, prompt_file=prompt_file, split='dev')

    elif args.task == 'hotpotqa_to_beir':
        input_file = args.inp[0]
        beir_dir = args.out
        hotpotqa_to_beir(input_file, beir_dir, split='dev')

    elif args.task == 'tsv_to_beir':
        tsv_file = args.inp[0]
        beir_dir = args.out
        tsv_to_beir(tsv_file, beir_dir, split='dev')

    elif args.task == 'eval':
        dataset = args.dataset
        jsonl_files = glob.glob(args.inp[0])
        if dataset == 'hotpotqa':
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text='answer is',
                beir_dir='data/hotpotqa/dev_beir',)
        elif dataset == 'strategyqa':
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text=['answer is (.*)'],
                beir_dir='data/strategyqa/train_cot_beir')
        elif dataset == 'mmlu':
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text=['answer is \(([ABCDE]*)\)', '\(([ABCDE]*)\) is correct', '\(([ABCDE]*)\) is the correct answer'],
                beir_dir=None)
        elif dataset == '2wikihop':
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text='answer is',
                beir_dir='data/2wikimultihopqa/dev_beir')
        elif dataset in {'wikisum', 'wikiasp'}:
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text=None,
                beir_dir=None)  # 'data/wikisum/wikisum_all_beir'
        elif dataset in {'eli5', 'wow', 'lmdata'}:
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text='',
                beir_dir=None)
        elif dataset in {'asqa'}:
            eval(model=args.model,
                dataset=dataset,
                jsonl_files=jsonl_files,
                anchor_text=None,
                prefix_to_remove=[
                    'The answers to all interpretations are\: (.*)$',
                    'The answer to this interpretation is\: (.*)$',
                    'The answer to this interpretation is (.*)$'],
                beir_dir=None)

    elif args.task == 'kilt_to_beir':
        kilt_wiki_file = args.inp[0]
        beir_dir = args.out
        kilt_to_beir(kilt_wiki_file, beir_dir)

    elif args.task == 'dpr_to_beir':
        dpr_file = args.inp[0]
        beir_dir = args.out
        dpr_to_beir(dpr_file, beir_dir)

    elif args.task == 'build_elasticsearch':
        beir_corpus_file_pattern, index_name = args.inp  # 'wikipedia_dpr'
        get_id_default = lambda doc: str(doc['_id'])
        get_id_lm = lambda doc: doc['metadata']['line'] + '.' + str(doc['_id'])
        build_elasticsearch(beir_corpus_file_pattern, index_name, get_id=get_id_default)

    elif args.task == 'mmlu_ret':
        mmlu_ret(output=args.out)

    elif args.task == 'prompt_dump':
        jsonl_file = args.inp[0]
        out_file = args.out
        prompt_dump(jsonl_file, out_file)

    elif args.task == 'kilt_dataset_to_beir':
        dataset, split = args.inp
        beir_dir = args.out
        kilt_dataset_to_beir(dataset, split, beir_dir)

    elif args.task == 'add_ref_to_kilt':
        dataset, split = args.inp
        out_file = args.out
        add_ref_to_kilt(dataset, split, out_file, skip_empty_ctx=True)

    elif args.task == 'jsonl_to_keyvalue':
        jsonl_file = args.inp[0]
        keyvalue_file = args.out
        jsonl_to_keyvalue(
            jsonl_file,
            keyvalue_file,
            prefix_to_remove=[
                'The answers to all interpretations are\: (.*)$',
                'The answer to this interpretation is\: (.*)$',
                'The answer to this interpretation is (.*)$'])

    elif args.task == 'mmlu_retrieval_usefulness':
        mmlu_retrieval_usefulness()

    elif args.task == 'wikiasp_match_title':
        output_dir = args.out
        wikiasp_match_title(output_dir, split='test')

    elif args.task == 'wikiasp_corpus':
        beir_dir = args.out
        wikiasp_corpus(beir_dir)

    elif args.task == 'annotate_asqa':
        pred_json_file = args.inp[0]
        out_file = args.out
        annotate_asqa(
            pred_json_file=pred_json_file,
            raw_file='data/asqa/ASQA.json',
            hint_file='data/asqa/ASQA_test_general_hint.jsonl',
            out_file=out_file,
            split='dev')
        # annotate_asqa(
        #    raw_file='data/asqa/ASQA.json',
        #    out_file=out_file,
        #    split='train')

    elif args.task == 'annotate_asqa_get_hint':
        annotate_asqa_get_hint(
            tsv_file='data/asqa/annotation.tsv',
            hint_file='data/asqa/ASQA_test_specific_hint_keyword.jsonl',
            out_file='data/asqa/annotation_hint.tsv',
        )
