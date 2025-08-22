# DRED - 行政文書RAGシステム

行政文書に対するMap-Reduce方式質問応答システム

## 概要

従来のRAGシステムの課題を解決するため、Mistral Nemoの大コンテキスト（128k tokens）を活用したMap-Reduce方式を採用。各文書全体を理解した上で質問応答を行い、結果を統合して包括的な回答を生成します。

## セットアップ

```bash
# 依存関係のインストール
uv sync

# Ollamaモデルの確認
ollama list
```

## 基本的な使用方法

### 1. 単一文書質問応答 (Single QA)

```bash
# 基本実行
uv run python map_reduce/single_doc_qa.py "data/空き家/空き家ガイドライン●.txt" "管理不全空家等の定義は何ですか？"

# プロンプトテンプレートを指定
uv run python map_reduce/single_doc_qa.py -t structured "data/空き家/住宅市街地総合整備事業制度要綱.txt" "支援制度はありますか？"

# JSON形式で出力
uv run python map_reduce/single_doc_qa.py --format json "data/空き家/空家等対策特別措置法.txt" "対象となる建物は？"

# 利用可能なテンプレート一覧
uv run python map_reduce/single_doc_qa.py --list-templates

# 対話継続モード
uv run python map_reduce/single_doc_qa.py -i "data/空き家/空き家ガイドライン●.txt" -t structured
```

### 2. Map-Reduce質問応答 (Aggregate QA)

```bash
# 基本実行（全文書対象）
uv run python map_reduce/aggregate_qa.py "３年前に家を相続した。売却のため、土地の上の建屋を取り壊したい。利用出来る支援金等は何かあるか？"

# テンプレートと並列数を指定
uv run python map_reduce/aggregate_qa.py "空き家の管理に関する支援制度を教えて" \
  --single-template structured \
  --aggregate-template consensus \
  --parallel 2

# 特定サブディレクトリのみを対象
uv run python map_reduce/aggregate_qa.py "立地適正化計画の策定手順は？" \
  --subdir 立地適正化計画 \
  --parallel 3
```

### 3. 実行管理・履歴確認

```bash
# 実行履歴一覧
uv run python map_reduce/aggregate_qa.py --list-runs

# 特定実行結果の確認
uv run python map_reduce/aggregate_qa.py --run-id 2025-08-25_0001

# 文書インデックス確認
uv run python map_reduce/document_indexer.py --stats
```

## 実行結果の保存先

```
run/
├── 2025-08-25_0001/           # 実行ID（日付_連番）
│   ├── single_qa/             # 各文書の個別回答
│   │   ├── 001_空き家_住宅市街地総合整備事業制度要綱.json
│   │   └── ...
│   ├── aggregate_result.txt   # 統合回答
│   └── metadata.json         # 実行情報・統計
└── 2025-08-25_0002/
    └── ...
```

## プロンプトテンプレート

### Single QA用
- `baseline`: シンプルな基本形
- `structured`: 構造化回答形式  
- `qualitative`: 市民相談員の視点
- `strict`: 厳密検証アプローチ

### Aggregate QA用
- `baseline`: シンプルな統合形式
- `consensus`: 専門統合アナリストによる包括回答

## システム特徴

- **大コンテキスト活用**: 文書全体（最大128k tokens）を一度に処理
- **並列処理**: プログレスバー付きで複数文書を効率的に処理
- **結果管理**: 日付ベース実行IDで履歴を体系的に管理
- **柔軟な設定**: プロンプトテンプレートで回答品質を調整可能
