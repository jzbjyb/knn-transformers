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
        tokenizer: AutoTokenizer,
        collator,
        corpus: GenericDataLoader,
        index_name: str,
        use_encoder_input_ids: bool = False,
        use_decoder_input_ids: bool = True,
    ):
        self.tokenizer = tokenizer
        self.collator = collator
        self.corpus = corpus
        # load bm25 index
        model = BM25Search(index_name=index_name, hostname='localhost', initialize=False, number_of_shards=1)
        self.retriever = EvaluateRetrieval(model)
        self.use_encoder_input_ids = use_encoder_input_ids
        self.use_decoder_input_ids = use_decoder_input_ids
        assert use_encoder_input_ids or use_decoder_input_ids, 'nothing used as queries'

    def retrieve_and_prepare(
        self,
        encoder_input_ids: torch.LongTensor = None,  # (bs, encoder_seq_len)
        decoder_input_ids: torch.LongTensor = None,  # (bs, decoder_seq_len)
        ctx_input_ids: torch.LongTensor = None,  # (bs, n_ctxs, ctx_seq_len)
        ctx_attention_mask: torch.FloatTensor = None,  # (bs, n_ctxs, ctx_seq_len)
        decoder_ctx_ids: np.ndarray = None,  # (bs, topk)
        topk: int = 1,
        use_ctx: bool = False,
    ):
        if use_ctx:
            return np.zeros((ctx_input_ids.size(0), topk)), ctx_input_ids[:, :topk], ctx_attention_mask[:, :topk]

        device = None
        if self.use_encoder_input_ids and encoder_input_ids is not None:
            device = encoder_input_ids.device
            bs = len(encoder_input_ids)
        if self.use_decoder_input_ids and decoder_input_ids is not None:
            device = decoder_input_ids.device
            bs = len(decoder_input_ids)

        if decoder_ctx_ids is not None:  # use doc ids passed in
            docids: List[str] = decoder_ctx_ids.reshape(-1)
            docs: List[str] = [self.format_func(self.corpus[did]) for did in docids]
        else:
            # prepare queries
            queries: List[str] = []
            if self.use_encoder_input_ids and encoder_input_ids is not None:
                queries = self.tokenizer.batch_decode(encoder_input_ids, skip_special_tokens=True)
            if self.use_decoder_input_ids and decoder_input_ids is not None:
                decoder_texts = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
                if len(queries):
                    assert len(queries) == len(decoder_texts), 'inconsistent length'
                    queries = [f'{q} {t}' for q, t in zip(queries, decoder_texts)]
                else:
                    queries = decoder_texts

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

        # tokenize
        ctxs = self.collator.encode_context(docs)
        input_ids = ctxs.input_ids.view(bs, topk, -1).to(device)  # (batch_size, topk, seq_length)
        attention_mask = ctxs.attention_mask.view(bs, topk, -1).to(device)  # (batch_size, topk, seq_length)
        docids = np.array(docids).reshape(bs, topk)  # (batch_size, topk)
        return docids, input_ids, attention_mask
