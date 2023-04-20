from typing import List, Dict, Any, Tuple
from operator import itemgetter
from collections import namedtuple
import spacy
from nltk.tokenize.punkt import PunktSentenceTokenizer
import tiktoken
import openai
from .utils import openai_api_call


class CtxPrompt:
    ctx_position: str = 'begin'
    ret_instruction: "RetrievalInstruction" = None
    instruction: str = None
    format_reference_method: str = 'default'
    clean_reference: bool = False
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

    @classmethod
    def chatgpt_get_response(cls, prompt: str, examplars: List[Tuple[str, str]] = [], max_tokens: int = 2048, api_key: str = None):
        assert len(prompt.split()) <= max_tokens
        responses = openai_api_call(
            api_key=api_key,
            model='gpt-3.5-turbo-0301',
            messages=[
                {'role': 'user' if i == 0 else 'assistant', 'content': examplar[i]} for examplar in examplars for i in range(2)
            ] + [
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.0,
            top_p=0.0,
            max_tokens=max_tokens)
        return responses['choices'][0]['message']['content']

    @classmethod
    def canonicalize_text(cls, text: str, field: str = 'paragraph', api_key: str = None):
        prompt = f'For the following {field}, remove unnecessary spaces and capitalize words properly.\n{field.capitalize()}\n{text}'
        clean_text = cls.chatgpt_get_response(prompt, api_key=api_key)
        return clean_text

    @classmethod
    def annotate_low_confidence_terms(cls, tokens: List[str], probs: List[float], low: float = 0.0, special_symbol: str = '*', min_gap: int = 5):
        # mark with symbol
        text = []
        prev_is_low = -1
        has = False
        for i, (token, prob) in enumerate(zip(tokens, probs)):
            if prob <= low:
                if prev_is_low == -1 or i - prev_is_low >= min_gap:
                    has = True
                    leading_spaces = len(token) - len(token.lstrip())
                    if leading_spaces <= 0:
                        text.append(f'*{token}')
                    else:
                        text.append(f'{token[:leading_spaces]}*{token[leading_spaces:]}')
                    prev_is_low = i
                else:
                    text.append(token)
            else:
                text.append(token)
        text = ''.join(text)
        return text, has

    @classmethod
    def extract_low_confidence_terms(cls, context: str, tokens: List[str], probs: List[float], low: float = 0.0, api_key: str = None, special_symbol: str = '*', debug: bool = False):
        examplars = [
            ('*Egypt has one of the longest histories of any country, tracing its heritage along *the Nile Delta back to the *6th–4th millennia BCE.', '*Egypt\n*the Nile Delta\n*6th–4th'),
            ('The settlement, which *legal experts said was *the largest struck by an American media company, was *announced by the two sides and the judge in the case at the 11th hour.', '*legal experts\n*the largest struck\n*announced'),
            ('In his only *surviving love letter to her, written a few months before their wedding, Tyler promised, "*Whether I *float or sink in the stream of fortune, you may be assured of this, that I shall never *cease to love you."', '*surviving love letter\n*Whether\n*float or sink\n*cease to love you')
        ]
        original_text = ''.join(tokens)
        text, has = cls.annotate_low_confidence_terms(tokens=tokens, probs=probs, low=low, special_symbol=special_symbol)
        if not has:
            return []
        # extract terms
        #prompt_format = lambda x: f'Given the previous context and the last sentence, extract all terms/entities in the last sentence starting with the symbol "{special_symbol}", one at a line.\nPrevious context:\n{context}\nLast sentence:\n{x}'
        prompt_format = lambda x: f'Given the following sentence, extract all terms/entities starting with the symbol "{special_symbol}", one at a line.\n{x}'
        examplars = [(prompt_format(inp), out) for inp, out in examplars]
        prompt = prompt_format(text)
        response = cls.chatgpt_get_response(prompt, examplars=examplars, api_key=api_key)
        terms = [t.strip() for t in response.strip().split('\n') if t.strip().startswith(special_symbol)]  # remove outlier
        terms = [t.lstrip(special_symbol) for t in terms if t in text and t.lstrip(special_symbol) in original_text]  # remove non-exist terms
        if debug:
            print('-' * 10)
            print(prompt)
            print('-' * 10)
            print(response)
            print('-' * 10)
            print(terms)
            print('-' * 10)
        return terms

    @classmethod
    def replace_low_confidence_terms(cls, context: str, tokens: List[str], probs: List[float], low: float = 0.0, api_key: str = None, special_symbol: str = '*', replace_symbol: str = 'XXX', debug: bool = False):
        text, has = cls.annotate_low_confidence_terms(tokens=tokens, probs=probs, low=low, special_symbol=special_symbol)
        if not has:
            return text
        # replace terms
        prompt = f'Given the previous context and the last sentence, detect all terms/entities in the last sentence starting with the symbol "{special_symbol}", then replace them with "{replace_symbol}".\nPrevious context:\n{context}\nLast sentence:\n{text}'
        replaced_text = cls.chatgpt_get_response(prompt, api_key=api_key)
        if debug:
            print('-' * 10)
            print(prompt)
            print('-' * 10)
            print(replaced_text)
            print('-' * 10)
        return replaced_text

    @classmethod
    def replace_low_confidence_terms_by_extract(cls, context: str, tokens: List[str], probs: List[float], low: float = 0.0, api_key: str = None, special_symbol: str = '*', replace_symbol: str = 'XXX', min_term_length: int = 0):
        text = ''.join(tokens)
        terms = cls.extract_low_confidence_terms(context=context, tokens=tokens, probs=probs, low=low, api_key=api_key, special_symbol=special_symbol)
        for term in terms:
            if min_term_length and len(term) <= min_term_length:  # ignore short terms
                continue
            text = text.replace(term, replace_symbol)
        return text

    @classmethod
    def decontextualize_text(cls, context: str, text: str, api_key: str = None, debug: bool = False):
        examplars = [
            ("The first American author to use natural diction and a pioneer of colloquialism, John Neal is the first to use the phrase son-of-a-bitch in a work of fiction.", "He attained his greatest literary achievements between 1817 and 1835, during which time he was America's first daily newspaper columnist, the first American published in British literary journals, author of the first history of American literature, America's first art critic, a short story pioneer, a children's literature pioneer, and a forerunner of the American Renaissance.", "John Neal attained his greatest literary achievements between 1817 and 1835, during which time he was America's first daily newspaper columnist, the first American published in British literary journals, author of the first history of American literature, America's first art critic, a short story pioneer, a children's literature pioneer, and a forerunner of the American Renaissance."),
            ("The Scottish wildcat is a European wildcat (Felis silvestris silvestris) population in Scotland.", "It was once widely distributed across Great Britain, but the population has declined drastically since the turn of the 20th century due to habitat loss and persecution.", "The Scottish wildcat was once widely distributed across Great Britain, but the population has declined drastically since the turn of the 20th century due to habitat loss and persecution."),
        ]
        examplars = []
        #prompt = f'Given the previous context and the last sentence, make minimal changes to the last sentence to make it self-contained by resolving pronoun references.\nPrevious context:\n{context}\nLast sentence:\n{text}'
        #prompt_format = lambda x, y: f'Given the previous context and the last text, copy the last text and only replace pronouns (if any) with corresponding references to make the text self-contained.\n=== Previous context ===\n{x.strip()}\n=== Last text ===\n{y.strip()}'
        #indicator = '---'
        #prompt_format = lambda x, y: f'Replace pronouns in the following text with their corresponding references.\n\n=== Text (start) ===\n{x.strip()}\n{indicator}\n{y.strip()}\n=== Text (end) ==='
        prompt_format = lambda x, y: f'Replace pronouns in the following text with their corresponding references.\n\n{x.strip()}\n=== Text (start) ===\n{y.strip()}\n=== Text (end) ==='
        examplars = [(prompt_format(e[0], e[1]), e[2]) for e in examplars]
        prompt = prompt_format(context, text)
        #decontext_text = cls.chatgpt_get_response(prompt, examplars=examplars, api_key=api_key).split(indicator, 1)[-1].strip()
        decontext_text = cls.chatgpt_get_response(prompt, examplars=examplars, api_key=api_key).strip()
        if debug:
            print('-' * 10)
            print(prompt)
            print('-' * 10)
            print(decontext_text)
            print('-' * 10)
        return decontext_text

    @classmethod
    def process_text_for_retrieval(
        cls,
        context: str,
        tokens: List[str],
        probs: List[float],
        low: float = 0.0,
        api_key: str = None,
        replace_symbol: str = 'XXX',
        detect_low_terms: bool = False,
        decontextualize: bool = False,
        debug: bool = False
    ):
        text = ''.join(tokens)
        if debug:
            print('0->', context)
            print('1->', text)
            print(list(zip(tokens, probs)))
        if detect_low_terms:
            #text = cls.replace_low_confidence_terms_by_extract(context=context, tokens=tokens, probs=probs, low=low, api_key=api_key, replace_symbol=replace_symbol)
            terms = cls.extract_low_confidence_terms(context=context, tokens=tokens, probs=probs, low=low, api_key=api_key)
        if debug:
            print('2->', terms)
        if decontextualize:
            text = cls.decontextualize_text(context=context, text=text, api_key=api_key)
        if debug:
            print('3->', text)
        if detect_low_terms:
            #text = text.replace(replace_symbol, ' ')
            for term in terms:
                text = text.replace(term, ' ')
        if debug:
            print('4->', text)
            input()
        return text

    def get_query_for_retrieval(self):
        if self.gen_len == 0:
            return self.question
            #question = self.question[:self.question.find('(A)')].strip()  # TODO: debug
            #return question
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

    def reinit_ctx(self):
        self.ctx = None
        self.ind = 1

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
    def format_reference(cls, ref: str, api_key: str = None):
        if cls.add_ref_suffix and not ref.endswith(cls.add_ref_suffix):
            ref += cls.add_ref_suffix
        if cls.add_ref_prefix and not ref.startswith(cls.add_ref_prefix):
            ref = cls.add_ref_prefix + ref
        if cls.clean_reference:
            ref = cls.canonicalize_text(ref, field='text', api_key=api_key)
        method = cls.format_reference_method
        assert method in {'default', 'searchresults', 'searchresultsrank', 'ignore', 'ignore_for_retrieval_instruct', 'short_ignore'}
        if method == 'default':
            return 'Reference: ' + ref
        if method == 'searchresults':
            return 'Search results :\n' + ref
        if method == 'searchresultsrank':
            return 'Search results ranked based on relevance in descending order:\n' + ref
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
        elif prefix_method.startswith('freq:'):
            firstk = int(prefix_method[len('freq:'):])
            prefix, self.gold_used_len = qagent.get_tokens(self.gold_output, topk=firstk)
            return prefix, 0
        else:
            raise NotImplementedError

    def format(
        self,
        use_ctx: bool = False,
        use_ret_instruction: bool = True,
        use_instruction: bool = True,
        is_chat_model: bool = False,
        api_key: str = None
    ):
        # run on demo
        demo_formatted: List[str] = [d.format(use_ctx=use_ctx, use_ret_instruction=False, use_instruction=False)[0] for d in self.demo]

        use_ctx = use_ctx and bool(self.ctx)  # do not use ctx when it's None or empty string
        use_ret_instruction = use_ret_instruction and self.ret_instruction is not None
        ref = self.format_reference(self.ctx, api_key=api_key) if use_ctx else None
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

        # append additional instruction
        if use_instruction and self.instruction is not None:
            elements.append(self.instruction)

        # append demo
        if len(demo_formatted) and not is_chat_model:
            elements.extend(demo_formatted)

        # append ensemble
        if use_ret_instruction:
            elements.append(ensemble)

        if use_ctx and self.ctx_position == 'before_case':
            elements.append(ref + '\n' + self.case)
        else:
            elements.append(self.case)

        return '\n\n'.join(elements), self.gen_len, demo_formatted


Sentence = namedtuple('Sentence', 'text start_char end_char')


class ApiReturn:
    EOS = '<|endoftext|>'
    #nlp = spacy.load('en_core_web_sm')
    psentencizer = PunktSentenceTokenizer()
    min_sent_len = 5

    def __init__(
        self,
        prompt: str,
        text: str,
        tokens: List[str] = None,
        probs: List[float] = None,
        offsets: List[int] = None,
        finish_reason: str = 'stop',
        model: str = None,
        skip_len: int = 0,
    ):
        self.model = model
        self.prompt = prompt
        self.text = text

        self.tokens = tokens
        self.probs = probs
        self.offsets = offsets
        if self.has_tokens:
            assert len(tokens) == len(probs) == len(offsets)

        self.finish_reason = finish_reason
        if self.finish_reason is None:
            self.finish_reason = 'stop'  # TODO: a bug from openai?

        if skip_len:  # skip `skip_len` chars at the beginning
            self.text = self.text[skip_len:]
            if self.has_tokens:
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
    def has_tokens(self):
        return self.tokens is not None

    @property
    def token_probs(self):
        if self.has_tokens:
            return self.probs
        else:
            return []

    @property
    def num_tokens(self):
        if self.has_tokens:
            return len(self.tokens)
        else:
            return len(tiktoken.encoding_for_model(self.model).encode(self.text))

    @property
    def has_endoftext(self):
        return self.EOS in self.tokens


    @classmethod
    def get_sent(cls, text: str, position: str = 'begin'):
        #sents = list(cls.nlp(text).sents)
        sents = [Sentence(text[s:e], s, e) for s, e in cls.psentencizer.span_tokenize(text)]
        if position == 'begin':
            break_at = len(text)
            for sent in sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= cls.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break
            return text[:break_at], break_at
        if position == 'end':
            break_at = 0
            for i in range(len(sents)):
                sent = sents[len(sents) - i - 1]
                if len(text) - sent.start_char >= cls.min_sent_len:  # TODO: argument
                    break_at = sent.start_char
                    break
            return text[break_at:], break_at
        raise NotImplementedError

    def truncate_at_prob(self, low: float):
        assert self.has_tokens, 'not supported'

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
            #sents = list(self.nlp(self.text).sents)
            sents = [Sentence(self.text[s:e], s, e) for s, e in self.psentencizer.span_tokenize(self.text)]
            break_at = len(self.text)
            for sent in sents:
                # remove trailing spaces which is usually tokenized into the next token of the next sentence by GPT tokeniers
                num_trail_spaces = len(sent.text) - len(sent.text.rstrip())
                if sent.end_char - num_trail_spaces >= self.min_sent_len:
                    break_at = sent.end_char - num_trail_spaces
                    break

            if break_at > 0 and break_at < len(self.text):  # truncation
                if self.has_tokens:
                    i = 0
                    for i in range(self.num_tokens):
                        if self.offsets[i] - len(self.prompt) >= break_at:
                            break_at = self.offsets[i] - len(self.prompt)
                            break
                    assert i > 0
                    self.tokens = self.tokens[:i]
                    self.probs = self.probs[:i]
                    self.offsets = self.offsets[:i]
                assert break_at > 0
                self.text = self.text[:break_at]
                self.finish_reason = 'boundary'
        else:
            raise NotImplementedError
        return self

    def truncate_at_substring(self, substr: str):
        position = self.text.find(substr)
        if position == -1:
            return
        self.text = self.text[:position]
        if self.has_tokens:
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
        low_prob: float = None,
        mask_prob: float = None,
        mask_method: str = 'simple',
        n_gen_char_in_prompt: int = 0,
        api_key: str = None,
    ):
        if not low_prob and not mask_prob:
            return self.text
        assert self.has_tokens, 'not supported'
        if low_prob:
            ok = False
            for p in self.probs:
                if p <= low_prob:
                    ok = True
                    break
            if not ok:
                return ''
        if mask_prob:
            if mask_method == 'simple':
                keep = [(t if p > mask_prob else ' ') for t, p in zip(self.tokens, self.probs)]
                keep = ''.join(keep).strip()
                return keep
            elif mask_method == 'wholeterm-decontextualize':
                if n_gen_char_in_prompt == 0:
                    context = ''
                else:
                    context = self.prompt[-n_gen_char_in_prompt:]
                keep = CtxPrompt.process_text_for_retrieval(
                    context=context,
                    tokens=self.tokens,
                    probs=self.probs,
                    low=mask_prob,
                    api_key=api_key,
                    detect_low_terms=True,
                    decontextualize=True)
                return keep
            else:
                raise NotImplementedError
        else:
            return self.text


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
