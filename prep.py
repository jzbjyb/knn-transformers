from typing import List, Tuple, Any, Union, Dict, Set, Callable
import argparse
import random
import os
import json
import time
import glob
from collections import defaultdict
import csv
import copy
import evaluate
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

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
            for did in did2dict:
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


def compare(file1: str, file2: str, only_show_diff: bool = False):
    with open(file1) as fin1, open(file2) as fin2:
        for l in fin1:
            example1 = json.loads(l)
            example2 = json.loads(fin2.readline())
            q = example1['question']
            c = example1['ctxs'][0] if 'ctxs' in example1 else example1['references']
            a = example1['gold'] if 'gold' in example1 else example1['answer']
            a2 = example2['gold'] if 'gold' in example2 else example2['answer']
            assert a == a2
            assert example1['question'] == example2['question']
            o1 = example1['output']
            o2 = example2['output']

            if not only_show_diff or o1 != o2:
                print('Q->', q)
                print('C->', c)
                print('A->', a)
                print('1->', o1)
                print('2->', o2)
                input()


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

def eval(
    jsonl_file: str,
    anchor_text: str = 'So the answer is',
    retrieval_percentiles: List[Union[int, float]] = [1, 0.25, 0.5, 0.75, 1.0],
    beir_dir: str = None,
):
    if beir_dir is not None:
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='dev')

    metric_func = evaluate.load('rouge')

    correct = incorrect = wrongformat = total = 0
    ret_accs: List[List[float]] = []
    ret_covers: List[List[float]] = []
    predictions: List[str] = []
    references: List[str] = []
    with open(jsonl_file, 'r') as fin:
        for l in fin:
            total += 1
            l = json.loads(l)
            qid = l['qid']
            question = l['question']
            ref = l['gold_output']
            pred = l['output'].split('\n\n', 1)[0].strip()
            yesno_ans = l['answer']
            if anchor_text in yesno_ans:
                yesno_ans = yesno_ans[yesno_ans.find(anchor_text) + len(anchor_text):].strip()[:-1].strip().lower()
            assert yesno_ans in {'yes', 'no'}

            references.append(ref)
            predictions.append(pred)
            if 'retrieval' in l and l['retrieval']:
                ret_dids = np.array(l['retrieval'], dtype=np.str_)
            else:
                ret_dids = np.array([['placeholder']], dtype=np.str_)

            # yes/no
            position = pred.find(anchor_text)
            if position == -1:
                #print(json.dumps(l, indent=True))
                #input()
                wrongformat += 1
            elif yesno_ans in pred[position + len(anchor_text):].strip().lower():
                correct += 1
            else:
                incorrect += 1

            # retrieval
            ret_accs.append([])
            ret_covers.append([])
            if ret_dids is not None:
                ret_seq_len = len(ret_dids)
                rel_dids: List[str] = np.array([d for d, r in qrels[qid].items() if r])
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

        # rouge
        metrics = metric_func.compute(predictions=predictions, references=references)

    ret_accs = np.array(ret_accs, dtype=float).mean(0)
    ret_covers = np.array(ret_covers, dtype=float).mean(0)
    format_list = lambda arr: ', '.join(map(lambda x: '{:.3f}'.format(x), arr.tolist()))
    print('correct\tincorrect\twrongformat')
    print(f'{correct / total}\t{incorrect / total}\t{wrongformat / total}')
    print('\t'.join(metrics.keys()))
    print('\t'.join(map(str, metrics.values())))
    print('#pred\t#gold')
    print(f'{np.mean([len(p) for p in predictions])}\t{np.mean([len(r) for r in references])}')
    print('retrieval acc\tcoverage')
    print(f'{format_list(ret_accs)}\t{format_list(ret_covers)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=[
        'kilt', 'retrieval_track', 'head_analysis', 'shuffle_evidence', 'retrieval_acc',
        'translation_to_beir', 'convert_beir_to_fid_format', 'use_answer_as_query_in_beir',
        'dedup_translation', 'layerhead', 'split_ctxs', 'convert_beir_corpus_to_translation',
        'convert_fid_to_beir', 'compare_logprob', 'summary_to_beir', 'compare',
        'strategyqa_to_beir', 'tsv_to_beir', 'eval'])
    parser.add_argument('--inp', type=str, default=None, nargs='+', help='input file')
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
        summary_to_beir(file_pattern, spm_path, beir_dir, max_num_examples=1000)

    elif args.task == 'compare':
        file1, file2 = args.inp
        compare(file1, file2, only_show_diff=True)

    elif args.task == 'strategyqa_to_beir':
        strategyqa_file = args.inp[0]
        if len(args.inp) > 1:
            prompt_file = args.inp[1]
        beir_dir = args.out
        strategyqa_to_beir(strategyqa_file, beir_dir, prompt_file=prompt_file, split='dev')

    elif args.task == 'tsv_to_beir':
        tsv_file = args.inp[0]
        beir_dir = args.out
        tsv_to_beir(tsv_file, beir_dir, split='dev')

    elif args.task == 'eval':
        jsonl_file = args.inp[0]
        eval(jsonl_file,
            anchor_text='So the final answer is:',
            beir_dir='data/strategyqa/train_cot_beir')
