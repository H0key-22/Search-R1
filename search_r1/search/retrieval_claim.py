# retrieve.py
import json
import os
from typing import List, Dict
from tqdm import tqdm
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
import datasets

def load_corpus(corpus_path: str):
    """
    Load a JSON corpus for retrieval.
    """
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus


def read_jsonl(file_path: str) -> List[Dict]:
    """
    Read a JSONL file into a list of dicts.
    Each record is expected to contain at least a 'claim' field.
    """
    data = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


class Encoder:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        pooling_method: str,
        max_length: int,
        use_fp16: bool
    ):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model, self.tokenizer = self._load_model(model_path, use_fp16)

    def _load_model(self, model_path: str, use_fp16: bool):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval().cuda()
        if use_fp16:
            model = model.half()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        return model, tokenizer

    @torch.no_grad()
    def encode(self, texts: List[str], is_query: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        # add prefixes for certain models
        name = self.model_name.lower()
        if "e5" in name:
            prefix = "query:" if is_query else "passage:"
            texts = [f"{prefix} {t}" for t in texts]
        if "bge" in name and is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)
        output = self.model(**inputs, return_dict=True)
        if hasattr(output, 'pooler_output'):
            emb = output.pooler_output
        else:
            emb = output.last_hidden_state[:, 0]
        if self.pooling_method == 'mean':
            mask = inputs['attention_mask'].unsqueeze(-1).bool()
            h = output.last_hidden_state.masked_fill(~mask, 0.0)
            emb = h.sum(1) / mask.sum(1)
        if 'dpr' not in name:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype(np.float32)


def load_docs(corpus, indices: List[int]) -> List[Dict]:
    """
    Given a datasets.Dataset-like corpus and a list of integer indices,
    return the corresponding document records.
    """
    return [corpus[int(i)] for i in indices]


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def search(self, query: str, return_score: bool=False):
        return self._search(query, self.topk, return_score)

    def batch_search(self, queries: List[str], return_score: bool=False):
        return self._batch_search(queries, self.topk, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)

    def _check_contain_doc(self) -> bool:
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int, return_score: bool):
        hits = self.searcher.search(query, num)
        if not hits:
            return ([],[]) if return_score else []
        docs, scores = [], []
        for h in hits[:num]:
            scores.append(h.score)
            if self.contain_doc:
                obj = json.loads(self.searcher.doc(h.docid).raw())
                docs.append({'contents': obj['contents']})
            else:
                docs.append(load_docs(self.corpus, [h.docid])[0])
        return (docs, scores) if return_score else docs


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co)
        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=config.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )

    def _search(self, query: str, num: int, return_score: bool):
        emb = self.encoder.encode(query)
        scores, idxs = self.index.search(emb, k=num)
        docs = load_docs(self.corpus, idxs[0])
        return (docs, scores[0].tolist()) if return_score else docs

    def _batch_search(self, queries: List[str], num: int, return_score: bool):
        all_docs, all_scores = [], []
        for i in range(0, len(queries), config.retrieval_batch_size):
            batch = queries[i:i+config.retrieval_batch_size]
            emb = self.encoder.encode(batch)
            scores, idxs = self.index.search(emb, k=num)
            for s, ids in zip(scores.tolist(), idxs.tolist()):
                all_scores.append(s)
                all_docs.append(load_docs(self.corpus, ids))
        return (all_docs, all_scores) if return_score else all_docs


def get_retriever(config):
    return BM25Retriever(config) if config.retrieval_method == 'bm25' else DenseRetriever(config)


def get_dataset(config):
    """
    Load dataset split file. Expects records with a 'claim' field.
    """
    path = os.path.join(config.dataset_path, f"{config.data_split}.jsonl")
    return read_jsonl(path)


# run_retrieval.py
if __name__ == '__main__':
    import argparse
    from retrieve import get_retriever, get_dataset

    parser = argparse.ArgumentParser("Run retrieval on claim dataset")
    parser.add_argument('--retrieval_method', required=True)
    parser.add_argument('--retrieval_topk', type=int, default=10)
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--corpus_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--data_split', default='train')
    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--retrieval_model_path', required=True)
    parser.add_argument('--retrieval_pooling_method', default='mean')
    parser.add_argument('--retrieval_query_max_length', type=int, default=256)
    parser.add_argument('--retrieval_use_fp16', action='store_true')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)

    args = parser.parse_args()
    # adjust index path for dense
    if args.retrieval_method != 'bm25':
        args.index_path = os.path.join(args.index_path, f"{args.retrieval_method}_Flat.index")
    else:
        args.index_path = os.path.join(args.index_path, 'bm25')

    samples = get_dataset(args)
    # use 'claim' instead of 'question'
    queries = [s['claim'] for s in samples[:512]]

    retriever = get_retriever(args)
    print('Start Retrieving ...')
    results, scores = retriever.batch_search(queries, return_score=True)

    out = 'retrieval_results.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'scores': scores}, f, ensure_ascii=False, indent=2)
    print(f'Retrieval results written to {out}')
