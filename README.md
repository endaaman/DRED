
# Dify + Ollama RAGシステム

DifyとOllamaを使用した行政文書向けRAGシステムの構築環境

## 必要なもの

- Docker & Docker Compose
- NVIDIA GPU（推奨）
- 16GB以上のメモリ

## セットアップ

### 1. クローン

```bash
git clone <repository-url>
cd DRED
```

### 2. モデルファイルダウンロード

```bash
# LLaMa3 ELYZA JP 8B
wget -P models/llama-elyzsa-jp-8b/ \
  https://huggingface.co/QuantFactory/Llama-3-ELYZA-JP-8B-GGUF/resolve/main/Llama-3-ELYZA-JP-8B.Q4_K_M.gguf

# Mistral Nemo JP
wget -P models/mistral-nemo-jp/ \
  https://huggingface.co/QuantFactory/Mistral-Nemo-Japanese-Instruct-2408-GGUF/resolve/main/Mistral-Nemo-Japanese-Instruct-2408.Q4_K_M.gguf
```

### 3. Ollamaセットアップ

```bash
# Ollamaインストール
curl -fsSL https://ollama.com/install.sh | sh

# モデル読み込み
ollama create llama-elyzsa-jp-8b -f models/llama-elyzsa-jp-8b/Modelfile
ollama create mistral-nemo-jp -f models/mistral-nemo-jp/Modelfile

# Embeddingモデル
ollama pull nomic-embed-text
```

### 4. Dify起動

```bash
cd dify-setup
echo "EXPOSE_NGINX_PORT=8080" >> .env
docker compose up -d
```

### 5. 初期設定

1. http://localhost:8080/install にアクセス
2. 管理者アカウント作成
3. ワークスペース作成
4. モデルプロバイダー設定：
   - Model Name: `llama-elyzsa-jp-8b`
   - Base URL: `http://host.docker.internal:11434`
   - Model Type: Chat
5. Embedding設定：
   - Model Name: `nomic-embed-text`
   - Base URL: `http://host.docker.internal:11434`
   - Model Type: Text Embedding

### 6. RAGシステム構築

1. Knowledge → Create Knowledge
2. Embedding Model: nomic-embed-text を選択
3. PDFファイルをアップロード
4. Studio → Chatbot作成
5. Model: llama-elyzsa-jp-8b を選択
6. Knowledge Base を追加

## トラブルシューティング

### コンテナエラー
```bash
# 問題のあるコンテナを停止
docker compose stop nginx ssrf_proxy sandbox
```

### Ollamaモデル確認
```bash
ollama list
curl http://localhost:11434/api/tags
```

### ログ確認
```bash
docker compose logs -f
```
