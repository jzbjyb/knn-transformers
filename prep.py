from typing import List, Tuple, Any, Union, Dict
import argparse
import random
import os
import json
import time
from collections import defaultdict
import csv
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from kilt.knowledge_source import KnowledgeSource
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

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
                        evidences.append(prov)
                if 'answer' in evidence_method and ans:
                    evidences.append(ans)
        
        if len(evidences) <= 0 and remove_wo_ctx:  # remove examples without ctx
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

def retrieval_acc_pt(filename: str):
    head2ids = torch.load(filename, map_location='cpu')
    head_accs: List[Tuple[int, float]] = []
    for head, ids in head2ids.items():
        acc = ids.eq(torch.arange(ids.size(0)).unsqueeze(-1)).any(-1).float().mean(0).item()
        head_accs.append((head, acc))
        print(f'{head}: {acc}')
    
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
            for did in did2dict:
                fout.write(json.dumps(did2dict[did]) + '\n')
    if split2qiddid is not None:
        os.makedirs(os.path.join(beir_dir, 'qrels'), exist_ok=True)
        for split in split2qiddid:
            with open(os.path.join(beir_dir, 'qrels', f'{split}.tsv'), 'w') as fout:
                fout.write('query-id\tcorpus-id\tscore\n')
                for qid, did in split2qiddid[split]:
                    fout.write(f'{qid}\t{did}\t1\n')

def translation_to_beir(translation_file: str, beir_dir: str, split: str = 'dev'):
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with open(translation_file, 'r') as fin:
        for l in fin:
            example = json.loads(l)['translation']
            question = example['en']
            answer = example['zh']
            evidence = example['decoder_prefix']
            qid = f'{str(len(qid2dict) + len(did2dict))}'
            qid2dict[qid] = {'_id': qid, 'text': question, 'metadata': {'answer': answer}}
            did = f'{str(len(qid2dict) + len(did2dict))}'
            did2dict[did] = {'_id': did, 'title': '', 'text': evidence}
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
    add_qrel_as_answer: str = None):
    
    assert add_qrel_as_answer in {None, 'title', 'text'}
    clean_text_for_tsv = lambda x: '' if x is None else x.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    beir_data = BEIRDataset(beir_dir, name=dataset_name)

    # build index
    hostname = 'localhost'
    number_of_shards = 1  # TODO
    corpus, _, _ = GenericDataLoader(data_folder=beir_dir).load(split=splits[0])
    model = BM25(index_name=dataset_name, hostname=hostname, initialize=True, number_of_shards=number_of_shards)
    model.index(corpus)
    time.sleep(5)

    for split_ind, split in enumerate(splits):
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split=split)

        # retrieve
        model = BM25(index_name=dataset_name, hostname=hostname, initialize=False, number_of_shards=number_of_shards)
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
                answer = [clean_text_for_tsv(corpus[did].get(add_qrel_as_answer)) for did, rel in qrels[qid].items() if rel]
            else:
                answer = beir_data.qid2answer[qid]
            query = clean_text_for_tsv(queries[qid])
            example = {'question': query, 'id': qid, 'answers': answer, 'ctxs': []}
            scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[:topk]
            if add_self:
                if beir_data.qid2meta[qid]['docid'] not in set(x[0] for x in scores):  # self doc not retrieved
                    scores.insert(0, (beir_data.qid2meta[qid]['docid'], scores[0][1] + 1.0))  # highest score
                    scores = scores[:topk]
            for rank in range(len(scores)):
                did = scores[rank][0]
                title = clean_text_for_tsv(corpus[did].get('title'))
                if add_self and did == beir_data.qid2meta[qid]['docid']:
                    text = clean_text_for_tsv(beir_data.qid2meta[qid]['context'])
                else:
                    text = clean_text_for_tsv(corpus[did].get('text'))
                example['ctxs'].append({'id': did, 'title': title, 'text': text})
            examples.append(example)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(examples, open(os.path.join(out_dir, f'{split}.json'), 'w'), indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=[
        'kilt', 'retrieval_track', 'head_analysis', 'shuffle_evidence', 'retrieval_acc', 'translation_to_beir', 'convert_beir_to_fid_format', 'use_answer_as_query_in_beir'])
    parser.add_argument('--inp', type=str, default=None, help='input file')
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
            num_negative_evidence=99,
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
        head_analysis(args.inp)
    
    elif args.task == 'shuffle_evidence':
        shuffle_evidence(args.inp, args.out)
    
    elif args.task == 'retrieval_acc':
        retrieval_acc(args.inp, format='pt')
    
    elif args.task == 'translation_to_beir':
        translation_file = args.inp
        beir_dir = args.out
        translation_to_beir(translation_file, beir_dir, split='dev')
    
    elif args.task == 'convert_beir_to_fid_format':
        beir_dir = args.inp
        out_dir = args.out
        convert_beir_to_fid_format(
            beir_dir, 
            out_dir, 
            dataset_name='wow',
            splits=['dev'],
            topk=100)
    
    elif args.task == 'use_answer_as_query_in_beir':
        beir_dir = args.inp
        out_dir = args.out
        tokenizer = AutoTokenizer.from_pretrained('google/t5-xl-lm-adapt')
        use_answer_as_query_in_beir(beir_dir, out_dir, truncate_to=None, tokenizer=tokenizer)
