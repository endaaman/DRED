# 行政文書RAGシステム構築 - Dify + Ollama

## 目標
A4 700ページ×十数個の行政文書に対するRAGベースの質問応答システムを、DifyとOllamaを使って構築する。

## 環境
- GPU: A6000 Ada (利用可能)
- 予算: 最小限（Ollamaローカル実行でコスト削減）

## タスク1: Difyセットアップ

### 1.1 Dify環境構築
- Difyをローカルでセルフホスト方式でセットアップ
- Docker Composeを使用した構築を推奨
- 必要な依存関係とポート設定を含める
- 日本語環境での動作確認

### 1.2 初期設定
- 管理画面へのアクセス設定
- 基本的なワークスペース作成
- セキュリティ設定の確認

## タスク2: Ollama統合

### 2.1 Ollama環境構築
- A6000 Adaを活用するOllamaインストール
- GPU認識とメモリ設定の最適化
- 日本語対応の適切なモデル選択と導入
  - LLM: japanese-stablelm-instruct、calm2、elyza等から選択
  - Embedding: multilingual-e5-large等の多言語モデル

### 2.2 Dify-Ollama連携設定
- DifyでOllamaをLLMプロバイダーとして追加
- APIエンドポイント設定 (通常 http://localhost:11434)
- モデル設定とパラメータ調整
- 接続テストと動作確認

## タスク3: RAGシステム基本構築

### 3.1 Knowledge Base作成
- Dify上でKnowledge Base新規作成
- 文書処理設定（チャンクサイズ、オーバーラップ等）
- Embedding設定の最適化

### 3.2 テスト用文書での動作確認
- 小規模なPDFでの動作テスト
- チャンク分割とベクトル化の確認
- 検索精度の初期評価

## タスク4: アプリケーション作成
- Dify上でChatbotアプリケーション作成
- Knowledge Baseとの連携設定
- プロンプトテンプレートの日本語最適化
- ソース情報表示の設定

## 成果物
1. 動作するDify + Ollamaローカル環境
2. テスト用文書での質問応答デモ
3. セットアップ手順書
4. 今後の大量文書投入に向けた最適化指針

## 注意事項
- GPU利用率の監視
- メモリ使用量の最適化
- 日本語文書処理の精度確認
- レスポンス速度の測定

この指示に従って、段階的に構築を進めてください。各ステップでの問題や最適化ポイントがあれば報告してください。
# Dify + Ollama RAGシステム構築ガイド（詳細版）

このドキュメントでは、DifyとOllamaを使用したRAG（Retrieval-Augmented Generation）システムの詳細なセットアップ手順を説明します。

> **注意**: このファイルは詳細な技術手順を記載しています。クイックスタートについては、プロジェクトルートの [README.md](../README.md) を先に確認してください。

## 目次

- [システム要件](#システム要件)
- [Difyのセットアップ](#difyのセットアップ)
- [Ollamaのセットアップ](#ollamaのセットアップ)
- [Dify-Ollama連携設定](#dify-ollama連携設定)
- [RAGシステムの構築](#ragシステムの構築)
- [トラブルシューティング](#トラブルシューティング)

## システム要件

### ハードウェア
- GPU: NVIDIA GPU（A6000 Ada推奨）
- メモリ: 16GB以上（推奨32GB以上）
- ストレージ: 50GB以上の空き容量

### ソフトウェア
- Docker 20.10以上
- Docker Compose v2.0以上
- NVIDIA Container Toolkit（GPU利用時）
- curl, jq（オプション）

## Difyのセットアップ

### 1. プロジェクトディレクトリの作成

```bash
mkdir -p dify-setup && cd dify-setup
```

### 2. 必要なファイルのダウンロード

```bash
# Docker Composeファイルのダウンロード
curl -L https://raw.githubusercontent.com/langgenius/dify/main/docker/docker-compose.yaml -o docker-compose.yaml

# 環境変数ファイルのダウンロード
curl -L https://raw.githubusercontent.com/langgenius/dify/main/docker/.env.example -o .env

# nginxテンプレートファイルのダウンロード
mkdir -p nginx
curl -L https://raw.githubusercontent.com/langgenius/dify/main/docker/nginx/docker-entrypoint.sh -o nginx/docker-entrypoint.sh
curl -L https://raw.githubusercontent.com/langgenius/dify/main/docker/nginx/nginx.conf.template -o nginx/nginx.conf.template
curl -L https://raw.githubusercontent.com/langgenius/dify/main/docker/nginx/proxy.conf.template -o nginx/proxy.conf.template
curl -L https://raw.githubusercontent.com/langgenius/dify/main/docker/nginx/https.conf.template -o nginx/https.conf.template
chmod +x nginx/docker-entrypoint.sh
mkdir -p nginx/conf.d
```

### 3. 環境変数の設定

`.env`ファイルを編集して必要な設定を行います：

```bash
# セキュアなシークレットキーの生成
SECRET_KEY="sk-$(openssl rand -hex 32)"
sed -i "s|SECRET_KEY=.*|SECRET_KEY=${SECRET_KEY}|" .env

# 管理者パスワードの設定（必要に応じて変更）
sed -i "s|INIT_PASSWORD=.*|INIT_PASSWORD=Admin123456|" .env

# ポート設定（デフォルトの80番ポートが使用中の場合）
echo "EXPOSE_NGINX_PORT=8080" >> .env
```

### 4. nginx設定ファイルの作成

`nginx/conf.d/default.conf`を作成：

```nginx
server {
    listen 80;
    server_name _;

    location /console/api {
        proxy_pass http://api:5001;
        include proxy.conf;
    }

    location /api {
        proxy_pass http://api:5001;
        include proxy.conf;
    }

    location /v1 {
        proxy_pass http://api:5001;
        include proxy.conf;
    }

    location /files {
        proxy_pass http://api:5001;
        include proxy.conf;
    }

    location / {
        proxy_pass http://web:3000;
        include proxy.conf;
    }
}
```

### 5. Difyの起動

```bash
# すべてのサービスを起動
docker compose up -d

# 起動状態の確認
docker compose ps

# ログの確認（問題がある場合）
docker compose logs -f
```

### 6. 初期設定

1. ブラウザで `http://localhost:8080/install` にアクセス
2. 管理者アカウントを作成（パスワード: `.env`で設定したINIT_PASSWORD）
3. ワークスペースを作成

## Ollamaのセットアップ

### 1. Ollamaのインストール

Ollamaが未インストールの場合：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. 利用可能なモデルの確認

```bash
# インストール済みモデルの一覧
ollama list

# 実行中のモデルを確認
curl -s http://localhost:11434/api/tags | jq -r '.models[] | .name'
```

### 3. 必要なモデルのダウンロード

#### LLMモデル（チャット用）
日本語対応モデルの例：

```bash
# 日本語対応LLaMAモデル
ollama pull llama-elyzsa-jp-8b

# または他の日本語対応モデル
ollama pull mistral-nemo-jp
```

#### Embeddingモデル（ベクトル化用）

```bash
# 多言語対応Embeddingモデル
ollama pull nomic-embed-text

# または他のEmbeddingモデル
ollama pull multilingual-e5-large
```

### 4. Ollamaサービスの確認

```bash
# Ollamaが正常に動作しているか確認
curl http://localhost:11434/api/tags
```

## Dify-Ollama連携設定

### 1. モデルプロバイダーの追加

1. Difyにログイン後、設定 → モデルプロバイダー へ移動
2. 「Ollama」を選択して追加

### 2. LLMモデルの設定

以下の情報を入力：

- **Model Name**: `llama-elyzsa-jp-8b`（使用するモデル名）
- **Base URL**: `http://host.docker.internal:11434`
- **Model Type**: Chat
- **Vision support**: オフ
- **Function call support**: オフ

### 3. Embeddingモデルの設定

新しいモデルを追加：

- **Model Name**: `nomic-embed-text`
- **Base URL**: `http://host.docker.internal:11434`
- **Model Type**: Text Embedding

### 4. 接続テスト

各モデルの「テスト」ボタンをクリックして接続を確認

## RAGシステムの構築

### 1. Knowledge Base（ナレッジベース）の作成

1. Difyメニューから「Knowledge」を選択
2. 「Create Knowledge」をクリック
3. 以下を設定：
   - **Name**: 任意の名前（例：行政文書DB）
   - **Description**: 説明文
   - **Embedding Model**: 設定したOllamaのEmbeddingモデルを選択

### 2. 文書のアップロード

1. 作成したKnowledge Baseを開く
2. 「Add Document」をクリック
3. PDFファイルをアップロード
4. 以下のチャンク設定を調整：
   - **Chunk Size**: 500-1000（文書に応じて調整）
   - **Chunk Overlap**: 50-100
   - **Retrieval Settings**: Top K = 3-5

### 3. Chatbotアプリケーションの作成

1. 「Studio」メニューから「Create from blank」を選択
2. 「Chatbot」を選択
3. 以下を設定：
   - **Model**: 設定したOllamaのLLMモデル
   - **Context**: 作成したKnowledge Baseを追加
   - **Prompt**: 日本語用のプロンプトテンプレートを設定

#### プロンプトテンプレート例

```
あなたは行政文書に関する質問に答えるアシスタントです。
提供されたコンテキストに基づいて、正確で簡潔な回答を日本語で提供してください。

コンテキスト:
{{context}}

質問: {{query}}

回答:
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. nginx/sandbox/ssrf_proxyコンテナが再起動を繰り返す

これらのコンテナは必須ではないため、停止しても問題ありません：

```bash
docker compose stop nginx ssrf_proxy sandbox
```

#### 2. プラグインデーモンエラー

```bash
# サービスの再起動
docker compose restart api worker plugin_daemon

# ログの確認
docker compose logs plugin_daemon
```

#### 3. Ollamaへの接続エラー

- Dockerコンテナからホストへの接続には `host.docker.internal` を使用
- ファイアウォール設定を確認
- Ollamaが11434ポートで動作していることを確認

#### 4. メモリ不足エラー

大きなモデルを使用する場合：

```bash
# Docker Desktopのメモリ割り当てを増やす
# または、より小さいモデルを使用
```

### ログの確認方法

```bash
# 全サービスのログ
docker compose logs

# 特定のサービスのログ
docker compose logs api
docker compose logs worker

# リアルタイムログ
docker compose logs -f
```

### サービスの管理

```bash
# すべてのサービスを停止
docker compose down

# 特定のサービスの再起動
docker compose restart api worker

# サービスの状態確認
docker compose ps
```

## セキュリティに関する注意事項

1. `.env`ファイルの`SECRET_KEY`は必ず変更してください
2. 本番環境では`INIT_PASSWORD`を強固なものに設定
3. 必要に応じてファイアウォール設定を行う
4. HTTPSの設定を検討（本番環境）

## 次のステップ

- より高度なRAG設定（リランキング、ハイブリッド検索など）
- カスタムツールの追加
- APIエンドポイントの活用
- マルチテナント設定

詳細な設定やカスタマイズについては、[Dify公式ドキュメント](https://docs.dify.ai/)を参照してください。
