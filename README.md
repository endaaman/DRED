
# Dify + Ollama RAGシステム

DifyとOllamaを使用した行政文書向けRAGシステムの構築環境

## 必要なもの

- Docker & Docker Compose
- NVIDIA GPU（推奨）
- 16GB以上のメモリ

## セットアップ

### 自動セットアップ（推奨）

```bash
git clone <repository-url>
cd DRED
./scripts/setup_all.sh
```

### 手動セットアップ

```bash
git clone <repository-url>
cd DRED

# 1. モデルファイルダウンロード
./scripts/download_models.sh

# 2. Ollamaセットアップ
./scripts/setup_ollama.sh

# 3. Dify起動
./scripts/setup_dify.sh
```

## 初期設定

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

## RAGシステム構築

1. Knowledge → Create Knowledge
2. Embedding Model: nomic-embed-text を選択
3. PDFファイルをアップロード
4. Studio → Chatbot作成
5. Model: llama-elyzsa-jp-8b を選択
6. Knowledge Base を追加

## トラブルシューティング

### root権限ファイル問題
Dockerが作成したroot権限ファイルで困った場合：
```bash
./cleanup.sh  # 自動検出して解決方法を案内
```

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
