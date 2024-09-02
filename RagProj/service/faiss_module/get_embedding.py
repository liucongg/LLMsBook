import faiss
import numpy as np
import json
from FlagEmbedding import FlagModel
from sklearn.preprocessing import normalize
from tqdm import tqdm


class Service:
    def __init__(self, model_path, doc_path):
        self.model = FlagModel(model_path,
                               query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                               use_fp16=True)
        self.faiss_index, self.all_lines = self.load_doc(doc_path)

    def predict_vector(self, text):
        vector = self.model.encode(text)
        return vector

    def init_file_vector(self, lines):
        all_vectors = []
        for line in lines:
            vector = self.predict_vector(line)
            all_vectors.append(vector)
        vectors = np.array(all_vectors)
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        return faiss_index

    def load_doc(self, doc_path):
        """
        加载doc相关文档，本次直接采用拆解完毕的jsonl 文件来运行，请注意您文档的格式
        :param doc_path: jsonl 文件目录
        :return:
        """
        all_lines = []
        all_vectors = []
        with open(doc_path, 'r', encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line)
                text = sample["text"]
                vector = self.predict_vector(text)
                all_lines.append(text)
                all_vectors.append(vector)
        vectors = np.array(all_vectors)
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        return faiss_index, all_lines

    def find_topk_text(self, faiss_index, text, topk):
        this_vector = self.predict_vector(text)
        source_vecs = normalize(this_vector, axis=1)
        res_distance, res_index = faiss_index.search(source_vecs, topk)
        lines = []
        for i, idx in enumerate(res_index):
            score = res_distance[i]
            text = self.all_lines[idx]
            lines.append(text)
        return lines
