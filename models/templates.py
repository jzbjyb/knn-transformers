from typing import List, Dict, Any, Tuple
from operator import itemgetter
import spacy


class CtxPrompt:
    ctx_position: str = 'begin'
    ret_instruction: "RetrievalInstruction" = None
    format_reference_method: str = 'default'
    add_ref_suffix: str = None
    add_ref_prefix: str = None

    def __init__(
        self,
        demo: List["CtxPrompt"] = [],
        ctx: str = None,
        ctxs: List[Tuple[str, str]] = [],
        case: str = None,
        question: str = None,
        qid: str = None,
        gold_output: str = None,
    ):
        assert self.ctx_position in {'before_case', 'begin'}
        self.demo = demo
        self.did = None
        self.ctx = ctx
        self.ctxs = ctxs
        self.ctxs_idx = 0
        self.case = case
        self.question = question or case
        self.qid = qid
        self.gold_output = gold_output
        self.ind = 1  # ctx index
        self.gen_len = 0
        self.gold_used_len = 0

    @staticmethod
    def get_append_retrieval(ret_to_append: str, index: int = None):
        if index is not None:
            return f'Reference {index}: {ret_to_append}\n'
        return f'Reference: {ret_to_append}\n'

    @classmethod
    def from_dict(cls, adict):
        adict = dict(adict)
        if 'demo' in adict:
            adict['demo'] = [cls.from_dict(d) for d in adict['demo']]
        return cls(**{k: adict[k] for k in ['demo', 'ctx', 'ctxs', 'case', 'question', 'qid', 'gold_output'] if k in adict})

    @classmethod
    def clean_rets(cls, rets: List[str]) -> List[str]:
        return [ret.replace('\n', ' ').strip() for ret in rets if ret.replace('\n', ' ').strip()]

    def get_query_for_retrieval(self):
        if self.gen_len == 0:
            return self.question
        else:
            return self.case

    def get_all_ctxs(self) -> List[str]:
        return list(map(itemgetter(1), self.ctxs))

    def add_generation(self, cont: str):
        self.case += cont
        self.gen_len += len(cont)
        if self.gold_used_len != 0:  # use gold
            self.gold_output = self.gold_output[self.gold_used_len:]
            self.gold_used_len = 0

    def reset_generation(self):
        if self.gen_len <= 0:
            return
        self.case = self.case[:-self.gen_len]
        self.gen_len = 0

    def change_ctx(self):
        assert len(self.ctxs)
        if self.ctxs_idx >= len(self.ctxs):
            return self.did, self.ctx
        self.did, self.ctx = self.ctxs[self.ctxs_idx]
        self.ctxs_idx += 1
        return self.did, self.ctx

    def append_retrieval(self, rets: List[str], add_index: bool = False):
        rets = self.clean_rets(rets)
        self.case += self.get_append_retrieval(rets, index=self.ind if add_index else None)  # TODO: fix list bug
        self.ind = (self.ind + 1) if add_index else self.ind

    def update_retrieval(self, rets: List[str], method: str = 'replace', dedup: bool = True, add_index: bool = True):
        rets = self.clean_rets(rets)
        def merge_rets():
            if add_index:
                return '\n'.join(f'[{self.ind + i}]: {ret}' for i, ret in enumerate(rets))
            return '\n'.join(rets)
        assert method in {'replace', 'append'}
        merge_ret = merge_rets()
        if self.ctx is None:
            self.ctx = merge_ret
        else:
            if method == 'replace':
                self.ctx = merge_ret
            elif method == 'append':
                if dedup:
                    if merge_ret.lower() not in self.ctx.lower():
                        self.ctx += '\n' + merge_ret
                        self.ind += len(rets)
                else:
                    self.ctx += '\n' + merge_ret
                    self.ind += len(rets)
            else:
                raise NotImplementedError

    @classmethod
    def format_reference(cls, ref: str):
        if cls.add_ref_suffix and not ref.endswith(cls.add_ref_suffix):
            ref += cls.add_ref_suffix
        if cls.add_ref_prefix and not ref.startswith(cls.add_ref_prefix):
            ref = cls.add_ref_prefix + ref
        method = cls.format_reference_method
        assert method in {'default', 'searchresults', 'ignore', 'ignore_for_retrieval_instruct', 'short_ignore'}
        if method == 'default':
            return 'Reference: ' + ref
        if method == 'searchresults':
            return 'Search results:\n' + ref
        if method == 'ignore':
            formatted = [
                '1. The reference below might be helpful when answering questions but it is noisy. Free free to ignore irrelevant information in it.', ref.strip(),
                '2. You should write out the reasoning steps and then draw your conclusion, where the reasoning steps should utilize the Search API "[Search(term)]" to look up information about "term" whenever possible. For example:']
            return '\n\n'.join(formatted)
        if method == 'ignore_for_retrieval_instruct':
            formatted = ['The reference below might be helpful when answering questions but it is noisy. Free free to ignore irrelevant information in it.', ref.strip()]
            return '\n\n'.join(formatted)
        if method == 'short_ignore':
            formatted = ['The reference below might be helpful but it is noisy. Free free to ignore irrelevant information in it:', ref.strip()]
            return ' '.join(formatted)
        raise NotImplementedError

    def get_prefix(
            self,
            qagent: "QueryAgent",
            prefix_method: str = 'sentence') -> Tuple[str, int]:
        if not self.gold_output:  # finish
            return qagent.final_stop_sym, 0
        if prefix_method == 'sentence':
            prefix, self.gold_used_len = ApiReturn.get_sent(self.gold_output, position='begin')
            return prefix, 0
        elif prefix_method == 'all':
            prefix, self.gold_used_len = self.gold_output, len(self.gold_output)
            return prefix, 0
        elif prefix_method.startswith('sentence_first:'):
            firstk = int(prefix_method[len('sentence_first:'):])
            prefix, self.gold_used_len = ApiReturn.get_sent(self.gold_output, position='begin')
            prefix = qagent.get_tokens(prefix, topk=firstk)[0]
            return prefix, None
        else:
            raise NotImplementedError

    def format(
        self,
        use_ctx: bool = False,
        use_ret_instruction: bool = True
    ):
        # run on demo
        demo_formatted: str = '\n\n'.join([d.format(use_ctx=use_ctx, use_ret_instruction=False)[0] for d in self.demo])  # TODO: no retrieval for demo

        use_ctx = use_ctx and bool(self.ctx)  # do not use ctx when it's None or empty string
        use_ret_instruction = use_ret_instruction and self.ret_instruction is not None
        ref = self.format_reference(self.ctx) if use_ctx else None
        task, ret, ensemble = self.ret_instruction.format(use_ctx=use_ctx) if use_ret_instruction else (None, None, None)
        elements: List[str] = []

        if use_ctx and self.ctx_position == 'begin':
            elements.append(ref)

        # append retrieval instructionj
        if use_ret_instruction:
            elements.append(ret)

        # append task instruction
        if use_ret_instruction:
            elements.append(task)

        # append demo
        if len(demo_formatted):
            elements.append(demo_formatted)

        # append ensemble
        if use_ret_instruction:
            elements.append(ensemble)

        if use_ctx and self.ctx_position == 'before_case':
            elements.append(ref + '\n' + self.case)
        else:
            elements.append(self.case)

        return '\n\n'.join(elements), self.gen_len


class ApiReturn:
    EOS = '<|endoftext|>'
    nlp = spacy.load('en_core_web_sm')
    min_sent_len = 5

    def __init__(
        self,
        prompt: str,
        text: str,
        tokens: List[str] = [],
        probs: List[float] = [],
        offsets: List[int] = [],
        finish_reason: str = 'stop',
        skip_len: int = 0,
    ):
        self.prompt = prompt
        self.text = text
        assert len(tokens) == len(probs) == len(offsets)
        self.tokens = tokens
        self.probs = probs
        self.offsets = offsets
        self.finish_reason = finish_reason
        if self.finish_reason is None:
            self.finish_reason = 'stop'  # TODO: a bug from openai?

        if skip_len:  # skip `skip_len` chars at the beginning
            self.text = self.text[skip_len:]
            i = 0
            for i, off in enumerate(self.offsets):
                if off == skip_len:
                    break
                elif off > skip_len:  # the previous token span across the boundary
                    i = i - 1
                    assert i >= 0
                    break
            self.tokens = self.tokens[i:]
            self.probs = self.probs[i:]
            self.offsets = self.offsets[i:]

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def has_endoftext(self):
        return self.EOS in self.tokens

    @classmethod
    def get_sent(cls, text: str, position: str = 'begin'):
        doc = cls.nlp(text)
        if position == 'begin':
            break_at = len(text)
            for sent in doc.sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= cls.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break
            return text[:break_at], break_at
        if position == 'end':
            sents = list(doc.sents)
            break_at = 0
            for i in range(len(sents)):
                sent = sents[len(sents) - i - 1]
                if len(text) - sent.start_char >= cls.min_sent_len:  # TODO: argument
                    break_at = sent.start_char
                    break
            return text[break_at:], break_at
        raise NotImplementedError

    def truncate_at_prob(self, low: float):
        if self.num_tokens <= 1:
            return self

        break_point = self.num_tokens
        for i in range(self.num_tokens):
            t, p, o = self.tokens[i], self.probs[i], self.offsets[i]
            if p <= low:
                break_point = i
                break
        if break_point == 0 and self.num_tokens > 0:  # avoid deadlock
            break_point = 1

        while break_point < self.num_tokens:  # truncation
            assert break_point > 0
            keep = self.offsets[break_point] - len(self.prompt)
            if keep <= 0:
                break_point += 1
                continue

            self.text = self.text[:keep]
            self.tokens = self.tokens[:break_point]
            self.probs = self.probs[:break_point]
            self.offsets = self.offsets[:break_point]
            self.finish_reason = 'boundary'
            break

        return self

    def truncate_at_boundary(self, unit: str = 'sentence'):
        if self.num_tokens <= 1:
            return self

        if unit == 'sentence':
            doc = self.nlp(self.text)
            break_at = len(self.text)
            for sent in doc.sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= self.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break

            if break_at > 0 and break_at < len(self.text):  # truncation
                i = 0
                for i in range(self.num_tokens):
                    if self.offsets[i] - len(self.prompt) >= break_at:
                        break_at = self.offsets[i] - len(self.prompt)
                        break
                assert i > 0 and break_at > 0
                self.text = self.text[:break_at]
                self.tokens = self.tokens[:i]
                self.probs = self.probs[:i]
                self.offsets = self.offsets[:i]
                self.finish_reason = 'boundary'
        else:
            raise NotImplementedError
        return self

    def truncate_at_substring(self, substr: str):
        position = self.text.find(substr)
        if position == -1:
            return
        self.text = self.text[:position]
        i = 0
        for i, off in enumerate(self.offsets):
            if off - len(self.prompt) == position:
                break
            elif off - len(self.prompt) > position:  # the previous token span across the boundary
                i = i - 1
                assert i >= 0
                break
        self.tokens = self.tokens[:i]
        self.probs = self.probs[:i]
        self.offsets = self.offsets[:i]

    def use_as_query(
        self,
        low: float = None,
        mask: float = None
    ):
        if low is None and mask is None:
            return self.text
        if low:
            ok = False
            for p in self.probs:
                if p <= low:
                    ok = True
                    break
            return self.text if ok else ''
        elif mask:
            keep = [(t if p > mask else ' ') for t, p in zip(self.tokens, self.probs)]
            keep = ''.join(keep).strip()
            return keep


class RetrievalInstruction:
    cot_instruction: Dict[str, Any] = {
        'retrieval': '1. You should use a Search API to look up information. You can do so by writing "[Search(term)]" where "term" is the search term you want to look up. For example:',
        'task': '2. You should answer questions by thinking step-by-step. You can do so by first write out the reasoning steps and then draw the conclusion. For example:',
        'ensemble': '3. Now, you should combine the aforementioned two abilities. You should first write out the reasoning steps and then draw then conclusion, where the reasoning steps should also utilize the Search API "[Search(term)]" whenever possible.',
        #'ensemble': '3. Now, you should combine the aforementioned two abilities. You should first write out the reasoning steps and then draw you conclusion, where the reasoning steps should also utilize the Search API "[Search(term)]" whenever possible. However, you should not directly copy chunks of words from "reference".',
        'examplars': [
            {
                'question': 'But what are the risks during production of nanomaterials?',
                'ctxs': [(None, 'The increased production of manufactured nanomaterials (MNMs) and their use in consumer and industrial products means that workers in all countries will be at the front line of any exposure, placing...')],
                'answer': '[Search("nanomaterial production risks")] Some nanomaterials may give rise to various kinds of lung damage.',
            },
            {
                'question': 'The colors on the flag of Ghana have the following meanings.',
                'ctxs': [(None, "The flag of Ghana comprises of the Pan-African colors of red, yellow and green. These colors are horizontal stripes that make up the background of the flag. Red is represents the nation's fight for independence, the gold is a sign of the country's mineral wealth, and the green is a representation of the country's natural wealth...")],
                'answer': 'Red is for [Search("Ghana flag red meaning")] the blood of martyrs, green for forests, and gold for mineral wealth.',
            },
            {
                'question': 'Metformin is the first-line drug for what?',
                'ctxs': [(None, "Metformin, sold under the brand name Glucophage, among others, is the main first-line medication for the treatment of type 2 diabetes,[6][7][8][9] particularly in people who are overweight.[7] It is also used in the treatment of polycystic ovary syndrome...")],
                'answer': '[Search("Metformin first-line drug")] patients with type 2 diabetes and obesity.'
            }
        ]
    }

    summary_instruction: Dict[str, Any] = {
        'task': '2. You should generate a short paragraph of summary for an entity. For example:',
        'ensemble': '3. Now, you should combine the aforementioned two abilities. You should generate a short paragraph of summary for an entity and utilize the Search API "[Search(term)]" whenever possible.',
    }

    def __init__(self, method: str = 'cot', fewshot: int = None):
        self.instruction = getattr(self, f'{method}_instruction')
        for k, v in self.cot_instruction.items():
            if k not in self.instruction:
                self.instruction[k] = v
        self.fewshot = len(self.instruction['examplars']) if fewshot is None else self.fewshot

    def format(self, use_ctx: bool = False) -> Tuple[str, str]:
        demos: List[str] = []
        for i in range(self.fewshot):
            q = self.instruction['examplars'][i]['question']
            a = self.instruction['examplars'][i]['answer']
            if use_ctx:
                ctxs = self.instruction['examplars'][i]['ctxs']
                assert CtxPrompt.ctx_position == 'before_case'
                ref = CtxPrompt.format_reference(' '.join(map(itemgetter(1), ctxs)))
                demo = f'{ref}\nQuestion: {q}\nAnswer (with Search): {a}'
            else:
                demo = f'Question: {q}\nAnswer (with Search): {a}'
            demos.append(demo)
        task = self.instruction['task']
        ret = self.instruction['retrieval'] + '\n\n' + '\n\n'.join(demos)
        ensemble = self.instruction['ensemble']
        return task, ret, ensemble
