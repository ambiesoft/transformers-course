from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale

# --- 1. モデルの設定 ---
lm_model_name = "gpt2"          # 言語モデル
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 埋め込み用

# トークナイザーとモデル
tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name)
lm_model.eval()

# 埋め込みモデル
embed_model = SentenceTransformer(embed_model_name)

# --- 2. 入力文 ---
input_text = "日本の首都はどこですか？"

# --- 3. モデルの平均トークン対数確率（log prob）計算 ---
with torch.no_grad():
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = lm_model(**inputs, labels=inputs["input_ids"])
    # loss は -平均対数尤度
    avg_log_prob = -outputs.loss.item()  # 高いほどモデルが得意とする入力

# --- 4. 外部知識（Wikipediaなど）との埋め込み類似度 ---
# ここでは例として単純に1文だけ（実運用なら複数文で最大コサイン類似度）
input_embedding = embed_model.encode([input_text])[0]
# 仮に外部知識ベース1件の埋め込み
kb_texts = ["東京は日本の首都である。"]
kb_embeddings = embed_model.encode(kb_texts)
cos_sim = np.dot(input_embedding, kb_embeddings[0]) / (
    np.linalg.norm(input_embedding) * np.linalg.norm(kb_embeddings[0]))

# --- 5. 特徴量正規化 ---
# 単純 min-max 正規化（実運用では学習データから範囲を決定）
avg_log_prob_norm = minmax_scale([avg_log_prob], feature_range=(0, 1))[0]
cos_sim_norm = cos_sim  # 0~1 で正規化済みと仮定

# --- 6. Closeness Score の統合 ---
w1, w2 = 0.5, 0.5  # 重み（例）
closeness_score = w1 * avg_log_prob_norm + w2 * cos_sim_norm

print(f"Closeness score: {closeness_score:.3f}")
