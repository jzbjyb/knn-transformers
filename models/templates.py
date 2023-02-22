from typing import List, Dict, Any, Tuple
from operator import itemgetter


class CtxPrompt:
    ctx_position: str = 'begin'
    ret_instruction: "RetrievalInstruction" = None
    format_reference_method: str = 'default'
    add_ref_suffix: str = '...'

    def __init__(
        self,
        demo: List["CtxPrompt"] = [],
        ctx: str = None,
        ctxs: List[Tuple[str, str]] = [],
        case: str = None,
        qid: str = None,
    ):
        assert self.ctx_position in {'before_case', 'begin'}
        self.demo = demo
        self.did = None
        self.ctx = ctx
        self.ctxs = ctxs
        self.ctxs_idx = 0
        self.case = case
        self.qid = qid
        self.ind = 0

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
        return cls(**{k: adict[k] for k in ['demo', 'ctx', 'ctxs', 'case', 'qid'] if k in adict})

    def change_ctx(self):
        assert len(self.ctxs)
        if self.ctxs_idx >= len(self.ctxs):
            return self.did, self.ctx
        self.did, self.ctx = self.ctxs[self.ctxs_idx]
        self.ctxs_idx += 1
        return self.did, self.ctx

    def append_retrieval(self, ret_to_append: str, add_index: bool = False):
        self.case += self.get_append_retrieval(ret_to_append, index=self.ind if add_index else None)
        self.ind = (self.ind + 1) if add_index else self.ind

    def update_retrieval(self, ret: str, method: str = 'replace', dedup: bool = True):
        assert method in {'replace', 'append'}
        if self.ctx is None:
            self.ctx = ret
        else:
            if method == 'replace':
                self.ctx = ret
            elif method == 'append':
                if dedup:
                    if ret.lower() not in self.ctx.lower():
                        self.ctx += ' ' + ret
                else:
                    self.ctx += ' ' + ret
            else:
                raise NotImplementedError

    @classmethod
    def format_reference(cls, ref: str):
        if cls.add_ref_suffix and not ref.endswith(cls.add_ref_suffix):
            ref += cls.add_ref_suffix
        method = cls.format_reference_method
        assert method in {'default', 'ignore', 'ignore_for_retrieval_instruct'}
        if method == 'default':
            return 'Reference:\n' + ref
        if method == 'ignore':
            formatted = [
                '1. The reference below might be helpful when answering questions but it is noisy. Free free to ignore irrelevant information in it.', ref.strip(),
                '2. You should write out the reasoning steps and then draw your conclusion, where the reasoning steps should utilize the Search API "[Search(term)]" to look up information about "term" whenever possible. For example:']
            return '\n\n'.join(formatted)
        if method == 'ignore_for_retrieval_instruct':
            formatted = ['The reference below might be helpful when answering questions but it is noisy. Free free to ignore irrelevant information in it.', ref.strip()]
            return '\n\n'.join(formatted)
        raise NotImplementedError

    def format(
        self,
        use_ctx: bool = False,
        use_ret_instruction: bool = True
    ):
        # run on demo
        demo_formatted: str = '\n\n'.join([d.format(use_ctx=use_ctx, use_ret_instruction=use_ret_instruction) for d in self.demo])  # TODO: no retrieval for demo

        if use_ctx and self.ctx is None and len(self.ctxs):  # default is use all ctxs
            self.ctx = ' '.join([ctx for _, ctx in self.ctxs])
        use_ctx = use_ctx and self.ctx
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

        return '\n\n'.join(elements)


class RetrievalInstruction:
    toolformer_instruction: Dict[str, Any] = {
        'retrieval': '1. You should use a Search API to look up information. You can do so by writing "[Search(term)]" where "term" is the search term you want to look up. For example:',
        'task': '2. You should answer a question by thinking step-by-step. You can do so by first write out the reasoning steps and then draw you conclusion. For example:',
        'ensemble': '3. Now, you should combine the aforementioned two abilities. You should first write out the reasoning steps and then draw you conclusion, where the reasoning steps should also utilize the Search API "[Search(term)]" whenever possible.',
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

    def __init__(self, method: str = 'toolformer', fewshot: int = None):
        self.instruction = getattr(self, f'{method}_instruction')
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
