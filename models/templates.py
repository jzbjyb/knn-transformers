from typing import List, Dict, Any, Tuple


class CtxPrompt:
    ctx_position: str = 'begin'
    ret_instruction: "RetrievalInstruction" = None

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

    def update_retrieval(self, ret: str, dedup: bool = True):
        if self.ctx is None:
            self.ctx = ret
        else:
            if dedup:
                if ret.lower() not in self.ctx.lower():
                    self.ctx += ' ' + ret
            else:
                self.ctx += ' ' + ret

    def format(
        self,
        use_ctx: bool = False,
        use_ret_instruction: bool = True
    ):
        if use_ctx and self.ctx is None:  # default is use all ctxs
            self.ctx = ' '.join([ctx for _, ctx in self.ctxs])
        use_ret_instruction = use_ret_instruction and self.ret_instruction is not None

        demo_formatted: str = '\n\n'.join([d.format(use_ctx=False, use_ret_instruction=False) for d in self.demo])  # TODO: no retrieval for demo
        ref = ('Reference:\n' + self.ctx) if use_ctx else None
        task, ret, ensemble = self.ret_instruction.format() if use_ret_instruction else (None, None, None)
        elements: List[str] = []

        if use_ctx and self.ctx_position == 'begin':
            elements.append(ref)

        # append retrieval instruction
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
            elements.append(ref)

        # append test case
        elements.append(self.case)

        return '\n\n'.join(elements)


class RetrievalInstruction:
    toolformer_instruction: Dict[str, Any] = {
        'retrieval': '1. You should use a Search API to look up information. You can do so by writing "[Search(term)]" where "term" is the search term you want to look up. For example:',
        'task': '2. You should answer a question by thinking step-by-step. You can do so by first write out the reasoning steps and then draw you conclusion. For example:',
        'ensemble': '3. Now, you should combine the aforementioned two abilities. You should first write out the reasoning steps and then draw you conclusion, where the reasoning steps should also utilize the Search API "[Search(term)]" whenever possible.',
        'examplars': [
            {
                'question': 'But what are the risks during production of nanomaterials?',
                'answer': '[Search("nanomaterial production risks")] Some nanomaterials may give rise to various kinds of lung damage.',
            },
            {
                'question': 'The colors on the flag of Ghana have the following meanings.',
                'answer': 'Red is for [Search("Ghana flag red meaning")] the blood of martyrs, green for forests, and gold for mineral wealth.',
            },
            {
                'question': 'Metformin is the first-line drug for what?',
                'answer': '[Search("Metformin first-line drug")] patients with type 2 diabetes and obesity.'
            }
        ]
    }

    def __init__(self, method: str = 'toolformer', fewshot: int = None):
        self.instruction = getattr(self, f'{method}_instruction')
        self.fewshot = len(self.instruction['examplars']) if fewshot is None else self.fewshot

    def format(self) -> Tuple[str, str]:
        demos: List[str] = []
        for i in range(self.fewshot):
            q = self.instruction['examplars'][i]['question']
            a = self.instruction['examplars'][i]['answer']
            demos.append(f'Question: {q}\nAnswer (with Search): {a}')
        task = self.instruction['task']
        ret = self.instruction['retrieval'] + '\n\n' + '\n\n'.join(demos)
        ensemble = self.instruction['ensemble']
        return task, ret, ensemble
