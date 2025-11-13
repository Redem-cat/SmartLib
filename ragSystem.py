import pickle
from pathlib import Path
import re
import os
import joblib
import string
import requests
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RagSystem:
    def __init__(self, api_key: str, docs_dir: str = "docs"):
        self.api_key = api_key
        self.docs_dir = Path(docs_dir)
        self.documents = []  # 原始文档
        self.doc_chunks = []  # 文档分块
        self.vectorizer = None
        self.doc_vectors = None  # 向量矩阵
        self.stopwords = self.load_stopwords()

        # 缓存目录
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

        # 缓存文件路径
        self.chunks_cache = self.cache_dir / "doc_chunks.pkl"
        self.vectorizer_cache = self.cache_dir / "vectorizer_cache.pkl"
        self.vector_matrix_cache = self.cache_dir / "vector_matrix_cache.pkl"

        # SiliconFlow API endpoint
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"

    # ------------------ 初始化 ------------------
    def initialize(self):
        print("🔧 智能图书检索系统初始化中...")
        self.load_documents()
        self.process_documents()
        self.build_vector_index()
        print("✅ 初始化完成，可以开始提问！")

    # ------------------ 文档加载 ------------------
    def load_documents(self):
        print("📘 正在加载资料文档...")
        if not self.docs_dir.exists() or not self.docs_dir.is_dir():
            raise FileNotFoundError(f"文档目录不存在: {self.docs_dir.absolute()}")

        for file_path in self.docs_dir.glob("*.txt"):
            print(f"找到文档: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        self.documents.append({
                            "filename": file_path.name,
                            "content": content,
                            "path": str(file_path)
                        })
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

        print(f"✅ 共加载 {len(self.documents)} 个文档。")

    # ------------------ 文本分块 ------------------
    def split_text_chunks(self, text: str, chunk_size: int = 300):
        """将文本按句号分块"""
        sentences = re.split(r"(?<=[。！？])", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # ------------------ 文档预处理（带缓存） ------------------
    def process_documents(self):
        if self.chunks_cache.exists():
            print("💾 发现文档分块缓存，正在加载...")
            with open(self.chunks_cache, "rb") as f:
                self.doc_chunks = pickle.load(f)
            print(f"✅ 从缓存中加载了 {len(self.doc_chunks)} 个分块。")
            return

        print("🧩 开始处理文档分块...")
        for doc in self.documents:
            chunks = self.split_text_chunks(doc["content"])
            for i, chunk in enumerate(chunks):
                self.doc_chunks.append({
                    "content": chunk,
                    "source": doc["filename"],
                    "chunk_id": i,
                    "full_path": doc["path"]
                })

        with open(self.chunks_cache, "wb") as f:
            pickle.dump(self.doc_chunks, f)
        print(f"✅ 文档分块完成，共生成 {len(self.doc_chunks)} 个chunk。")

    # ------------------ 停用词加载 ------------------
    def load_stopwords(self):
        stopwords_file = Path("中文停用词库.txt")
        stopwords = set()
        if stopwords_file.exists():
            with open(stopwords_file, "r", encoding="utf-8") as f:
                stopwords = {line.strip() for line in f if line.strip()}
            print(f"✅ 已加载 {len(stopwords)} 个停用词。")
        else:
            print("⚠️ 未找到停用词库文件，将使用默认停用词表。")
            stopwords.update({
                "的", "了", "和", "是", "在", "我", "有", "就", "不", "人",
                "都", "一个", "上", "也", "很", "到", "说", "要", "去",
                "你", "会", "着", "没有", "看", "自己", "这", "那", "还", "什么"
            })
        return stopwords

    # ------------------ 中文分词 ------------------
    def chinese_tokenizer(self, text: str):
        words = list(jieba.cut(text))
        cleaned_words = []
        for word in words:
            word = word.strip()
            if not word or word in self.stopwords:
                continue
            if word in string.punctuation or re.match(r"^[\W_]+$", word):
                continue
            if len(word) == 1:
                continue
            cleaned_words.append(word)
        return cleaned_words

    # ------------------ TF-IDF 向量索引 ------------------
    def build_vector_index(self):
        """构建或加载 TF-IDF 向量索引"""
        if self.vectorizer_cache.exists() and self.vector_matrix_cache.exists():
            print("💾 检测到向量缓存文件，正在加载...")
            self.vectorizer = joblib.load(self.vectorizer_cache)
            self.doc_vectors = joblib.load(self.vector_matrix_cache)
            print(f"✅ 从缓存加载 TF-IDF 矩阵，形状: {self.doc_vectors.shape}")
            return self.doc_vectors

        print("⚙️ 正在构建 TF-IDF 向量索引...")
        corpus = [chunk["content"] for chunk in self.doc_chunks]
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.chinese_tokenizer,
            token_pattern=None,
            max_features=None,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

        joblib.dump(self.vectorizer, self.vectorizer_cache)
        joblib.dump(self.doc_vectors, self.vector_matrix_cache)
        print(f"✅ 向量索引构建完成！形状: {self.doc_vectors.shape}")
        return self.doc_vectors

    # ------------------ 从缓存加载 ------------------
    def load_vector_cache(self):
        """从缓存加载向量化模型与矩阵"""
        if not (self.vectorizer_cache.exists() and self.vector_matrix_cache.exists()):
            print("⚠️ 未检测到向量缓存文件，请先调用 build_vector_index() 构建索引。")
            return False
        try:
            self.vectorizer = joblib.load(self.vectorizer_cache)
            self.doc_vectors = joblib.load(self.vector_matrix_cache)
            print(f"✅ 从缓存加载 TF-IDF 矩阵，形状: {self.doc_vectors.shape}")
            return True
        except Exception as e:
            print(f"❌ 向量缓存加载失败: {e}")
            return False

    # ------------------ 文档检索 ------------------
    def search_chunks(self, query: str, top_k: int = 10, similarity_threshold: float = 0.05):
        if self.vectorizer is None or self.doc_vectors is None:
            raise ValueError("向量索引未构建，请先调用 build_vector_index()")

        query_vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_vectors).flatten()
        valid_index = np.where(similarity > similarity_threshold)[0]
        if len(valid_index) == 0:
            return []

        sorted_index = valid_index[np.argsort(similarity[valid_index])[::-1]]
        top_index = sorted_index[:top_k]

        results = []
        for idx in top_index:
            chunk = self.doc_chunks[idx].copy()
            chunk["similarity"] = float(similarity[idx])
            results.append(chunk)
        return results

    # ------------------ SiliconFlow 模型调用 ------------------
    def generate_answer(self, query: str, context_chunks):
        context = "\n\n".join([f"文档片段{i + 1}: {chunk['content']}" for i, chunk in enumerate(context_chunks)])
        user_prompt = f"基于以下文档回答问题：\n{context}\n\n问题：{query}"

        payload = {
            "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "messages": [
                {"role": "system", "content": "你是一位专业的文学分析专家。"},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return str(result)
        except requests.exceptions.RequestException as e:
            return f"❌ 调用 SiliconFlow API 出错: {e}"

    # ------------------ 主问答流程 ------------------
    def ask(self, question: str):
        relevant_chunks = self.search_chunks(question)
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "抱歉，在相关文档中无法找到与您问题相关的内容。",
                "source": []
            }
        print(f"找到 {len(relevant_chunks)} 个相关文档片段，正在生成答案...")
        answer = self.generate_answer(question, relevant_chunks)
        return {"question": question, "answer": answer, "source": relevant_chunks}


# ------------------ 主函数 ------------------
def main():
    rag = RagSystem("sk-ukzszmjmdpsurolcgrjhlhfgrnqvljaczcfgldezhvhkxsvg")
    rag.initialize()

    print("=== 智能图书问答系统 ===")
    print("输入您的问题，输入 'quit' 或 'exit' 退出。")

    while True:
        question = input("请输入您的问题：").strip()
        if question.lower() in ["quit", "exit", "退出"]:
            print("感谢使用系统，再见！")
            break
        if not question:
            continue

        result = rag.ask(question)
        print("\n--- 答案 ---")
        print(result["answer"])
        print("\n--- 来源 ---")
        for src in result["source"]:
            print(f"- {src['source']} (chunk {src['chunk_id']}, 相似度: {src['similarity']:.3f})")
        print("\n")


if __name__ == "__main__":
    main()
