
# Dify + Ollama RAGシステム

DifyとOllamaを使用した行政文書向けRAGシステムの構築環境

## 必要なもの

- Docker & Docker Compose
- GPU（推奨）
- 16GB以上のメモリ

## セットアップ

### 自動セットアップ
```bash
./scripts/setup_all.sh
```

### 手動セットアップ
```bash
cd dify-setup
docker compose up -d
```

## Dify設定

1. http://localhost/install にアクセス
2. 管理者アカウント作成
3. Ollama設定：
   - Base URL: `http://host.docker.internal:11434`
   - Chat Model: `llama-elyzsa-jp-8b`
   - Embedding Model: `nomic-embed-text`
4. Knowledge Base作成 → PDFアップロード
5. Studio → Chatbot作成

## トラブルシューティング

### マーケットプレイスからプラグインインストール失敗
- エラー: `Reached maximum retries (3) for URL https://marketplace.dify.ai/api/v1/plugins/download`
- 原因: コミュニティプラグインの検査
- 対処: `FORCE_VERIFYING_SIGNATURE=false`

```bash
# コンテナ確認
docker compose ps

# ログ確認
docker compose logs -f

# クリーンアップ
./cleanup.sh
```
