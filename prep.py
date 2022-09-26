from typing import List, Tuple
import argparse
import random
import json
from collections import defaultdict
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

def prep_eli5(
    args, 
    split: str = 'validation', 
    evidence_method: str = 'provenance', 
    skip_answer_as_evidence: bool = True):
    assert evidence_method in {'provenance', 'self_provenance', 'answer', 'self_answer'}

    qa_file = f'{args.out_file}_qa.json'
    prov_file = f'{args.out_file}_evidence.json'
    eli5 = load_dataset('kilt_tasks', name='eli5')
    wikipedia = Wikipedia()

    with open(qa_file, 'w') as qfin, open(prov_file, 'w') as pfin:
        for i, example in enumerate(eli5[split]):
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
                            prov = wikipedia.get_provenance(wiki_id, ps, pe, cs, ce, whole_paragraph=True)  # always use the whole paragraph
                            evidences.append(prov)
                    if 'answer' in evidence_method and ans:
                        evidences.append(ans)
            
            # write qa pairs
            qfin.write(json.dumps({'translation': {'en': inp, 'zh': answer}}) + '\n')

            # write evidences
            if 'self' in evidence_method:
                for evi in evidences:
                    pfin.write(json.dumps({'translation': {'en': inp, 'zh': answer, 'decoder_prefix': evi}}) + '\n')
            else:
                for evi in evidences:
                    pfin.write(json.dumps({'translation': {'en': inp, 'zh': evi}}) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=['eli5'])
    parser.add_argument('--out_file', type=str, default=None, help='output file')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    if args.task == 'eli5':
        prep_eli5(args, evidence_method='answer', skip_answer_as_evidence=False)
