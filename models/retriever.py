from typing import List, Callable
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search


class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        collator = None,
        dataset: GenericDataLoader = None,
        index_name: str = None,
        encode_retrieval_in: str = 'encoder',
        use_encoder_input_ids: bool = False,
        use_decoder_input_ids: bool = True,
    ):
        self.tokenizer = tokenizer
        self.collator = collator
        self.corpus, self.queries, self.qrels = dataset
        # load bm25 index
        model = BM25Search(index_name=index_name, hostname='localhost', initialize=False, number_of_shards=1)
        self.retriever = EvaluateRetrieval(model)
        self.encode_retrieval_in = encode_retrieval_in
        assert encode_retrieval_in in {'encoder', 'decoder'}
        self.use_encoder_input_ids = use_encoder_input_ids
        self.use_decoder_input_ids = use_decoder_input_ids
        assert use_encoder_input_ids or use_decoder_input_ids, 'nothing used as queries'

    def retrieve_and_prepare(
        self,
        encoder_input_ids: torch.LongTensor = None,  # (bs, encoder_seq_len)
        decoder_input_ids: torch.LongTensor = None,  # (bs, decoder_seq_len)
        encoder_texts: List[str] = None,  # (bs, encoder_seq_len)
        decoder_texts: List[str] = None,  # (bs, encoder_seq_len)
        ctx_ids: np.ndarray = None,  # (bs, topk)
        qids: np.ndarray = None,  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
        use_gold: bool = False,
        joint_encode_retrieval: bool = False,
        merge_ctx: bool = False,
    ):
        device = None
        if self.use_encoder_input_ids and encoder_input_ids is not None:
            device = encoder_input_ids.device
        if self.use_decoder_input_ids and decoder_input_ids is not None:
            device = decoder_input_ids.device

        if self.use_encoder_input_ids and encoder_input_ids is not None:
            encoder_texts: List[str] = self.tokenizer.batch_decode(encoder_input_ids, skip_special_tokens=True)
        if self.use_decoder_input_ids and decoder_input_ids is not None:
            decoder_texts: List[str] = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)

        if encoder_texts is not None:
            bs = len(encoder_texts)
        if decoder_texts is not None:
            bs = len(decoder_texts)

        if ctx_ids is not None:  # use doc ids passed in
            docids: List[str] = ctx_ids.reshape(-1)
            docs: List[str] = [self.corpus[did]['text'] for did in docids]
        elif use_gold:  # use qrels annotations to find gold ctxs
            docids: List[str] = []
            docs: List[str] = []
            for qid in qids:
                rel_dids = [did for did, r in self.qrels[qid].items() if r]
                rel_docs = [self.corpus[did]['text'] for did in rel_dids]
                if merge_ctx:
                    rel_dids = rel_dids[:1]
                    rel_docs = [' '.join(rel_docs)]
                assert len(rel_dids) == len(rel_docs) == topk, f'{len(rel_dids)} {len(rel_docs)} {topk}'
                docids.extend(rel_dids)
                docs.extend(rel_docs)
        else:
            # prepare queries
            queries: List[str] = []
            if self.use_encoder_input_ids and encoder_texts is not None:
                queries = list(encoder_texts)
            if self.use_decoder_input_ids and decoder_texts is not None:
                if queries:
                    assert len(queries) == len(decoder_texts), 'inconsistent length'
                    queries = [f'{q} {t}' for q, t in zip(queries, decoder_texts)]
                else:
                    queries = list(decoder_texts)

            # truncate queries
            if max_query_length:
                ori_ps = self.tokenizer.padding_side
                ori_ts = self.tokenizer.truncation_side
                self.tokenizer.padding_side = 'left'
                self.tokenizer.truncation_side = 'left'
                tokenized = self.tokenizer(
                    queries,
                    truncation=True,
                    padding=True,
                    max_length=max_query_length,
                    add_special_tokens=False,
                    return_tensors='pt')['input_ids']
                self.tokenizer.padding_side = ori_ps
                self.tokenizer.truncation_side = ori_ts
                queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

            # retrieve
            results = self.retriever.retrieve(self.corpus, dict(zip(range(len(queries)), queries)), disable_tqdm=True)

            # prepare outputs
            docids: List[str] = []
            docs: List[str] = []
            for qid, query in enumerate(queries):
                _docids: List[str] = list(results[qid].keys())[:topk] if qid in results else []
                _docs = [self.corpus[did]['text'] for did in _docids]
                if len(_docids) < topk:  # add dummy docs
                    _docids += ['-1'] * (topk - len(_docids))
                    _docs += [''] * (topk - len(_docs))
                docids.extend(_docids)
                docs.extend(_docs)

        if device is None:
            docids = np.array(docids).reshape(bs, topk)  # (batch_size, topk)
            docs = np.array(docs).reshape(bs, topk)  # (batch_size, topk)
            return docids, docs

        # tokenize
        if joint_encode_retrieval:
            for i in range(bs):
                for j in range(topk):
                    if self.encode_retrieval_in == 'encoder':
                        docs[i * topk + j] = f'{docs[i * topk + j]}\n{encoder_texts[i]}'
                    elif self.encode_retrieval_in == 'decoder':
                        docs[i * topk + j] = f'{docs[i * topk + j]}\n'
                    else:
                        raise NotImplementedError
            if self.encode_retrieval_in == 'encoder':
                ctxs = self.collator.encode_context(docs, max_length=self.collator.max_context_len + self.collator.max_question_len)
            elif self.encode_retrieval_in == 'decoder':
                ctxs = self.collator.encode_context(docs)
                assert ctxs.input_ids[:, 0].eq(self.collator.get_real_decoder_start_token_id).all()
                assert topk == 1
                #decoder_input_ids = decoder_input_ids[:, 1:]  # skip decoder_start_token
                ctxs.input_ids = torch.cat([ctxs.input_ids.to(device), decoder_input_ids], 1)
                ctxs.attention_mask = torch.cat([ctxs.attention_mask.to(device), torch.ones_like(decoder_input_ids).to(ctxs.attention_mask.dtype)], 1)
            else:
                raise NotImplementedError
        else:
            ctxs = self.collator.encode_context(docs)
        input_ids = ctxs.input_ids.view(bs, topk, -1).to(device)  # (batch_size, topk, seq_length)
        attention_mask = ctxs.attention_mask.view(bs, topk, -1).to(device)  # (batch_size, topk, seq_length)
        docids = np.array(docids).reshape(bs, topk)  # (batch_size, topk)
        return docids, input_ids, attention_mask
