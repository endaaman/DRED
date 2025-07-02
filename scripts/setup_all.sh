#!/bin/bash

echo "🚀 DRED - 完全自動セットアップ"
echo "============================"

# プロジェクトルートかチェック
if [ ! -f "../CLAUDE.md" ] && [ ! -f "CLAUDE.md" ]; then
    echo "❌ エラー: プロジェクトルートまたはscriptsディレクトリで実行してください"
    exit 1
fi

# プロジェクトルートに移動
if [ -f "../CLAUDE.md" ]; then
    cd ..
fi

echo ""
echo "📋 実行内容:"
echo "   1. モデルファイルダウンロード"
echo "   2. Ollamaセットアップ"
echo "   3. Difyセットアップ"
echo ""

# スクリプトの実行権限設定
chmod +x scripts/*.sh

# ステップ1: モデルダウンロード
echo "🔧 ステップ1: モデルファイルダウンロード"
echo "======================================="
./scripts/download_models.sh
if [ $? -ne 0 ]; then
    echo "❌ モデルダウンロードに失敗しました"
    exit 1
fi

echo ""

# ステップ2: Ollamaセットアップ
echo "🔧 ステップ2: Ollamaセットアップ"
echo "=============================="
./scripts/setup_ollama.sh
if [ $? -ne 0 ]; then
    echo "❌ Ollamaセットアップに失敗しました"
    exit 1
fi

echo ""

# ステップ3: Difyセットアップ
echo "🔧 ステップ3: Difyセットアップ"
echo "============================"
./scripts/setup_dify.sh
if [ $? -ne 0 ]; then
    echo "❌ Difyセットアップに失敗しました"
    exit 1
fi

echo ""
echo "🎉 全セットアップ完了!"
echo "==================="
echo ""
echo "📍 次のステップ:"
echo "   1. ブラウザで http://localhost:8080/install にアクセス"
echo "   2. 管理者アカウントを作成"
echo "   3. ワークスペースを作成"
echo "   4. モデルプロバイダー設定:"
echo "      - LLM: llama-elyzsa-jp-8b"
echo "      - Base URL: http://host.docker.internal:11434"
echo "      - Embedding: nomic-embed-text"
echo "   5. Knowledge Base作成してPDFアップロード"
echo "   6. Chatbotアプリ作成"
echo ""
echo "🛠️  便利なスクリプト:"
echo "   - 完全クリーンアップ: ./cleanup.sh"
echo "   - 個別Ollamaセットアップ: ./scripts/setup_ollama.sh"
echo "   - 個別Difyセットアップ: ./scripts/setup_dify.sh"