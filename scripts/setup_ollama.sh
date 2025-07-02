#!/bin/bash

echo "🤖 Ollamaセットアップ"
echo "=================="

# プロジェクトルートかチェック
if [ ! -f "../CLAUDE.md" ] && [ ! -f "CLAUDE.md" ]; then
    echo "❌ エラー: プロジェクトルートまたはscriptsディレクトリで実行してください"
    exit 1
fi

# プロジェクトルートに移動
if [ -f "../CLAUDE.md" ]; then
    cd ..
fi

# Ollamaの確認
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollamaがインストールされていません"
    echo "💡 以下のコマンドでインストールしてください:"
    echo "   curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

echo "✅ Ollama確認済み"

# Ollamaサービス確認
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollamaサービスが動作していません"
    echo "💡 Ollamaを起動してから再実行してください"
    exit 1
fi

echo "✅ Ollamaサービス確認済み"

# モデルファイル存在確認
if [ ! -f "models/llama-elyzsa-jp-8b/Llama-3-ELYZA-JP-8B.Q4_K_M.gguf" ]; then
    echo "❌ LLaMa3 ELYZA JP 8B モデルファイルが見つかりません"
    echo "💡 先に scripts/download_models.sh を実行してください"
    exit 1
fi

if [ ! -f "models/mistral-nemo-jp/Mistral-Nemo-Japanese-Instruct-2408.Q4_K_M.gguf" ]; then
    echo "❌ Mistral Nemo JP モデルファイルが見つかりません"
    echo "💡 先に scripts/download_models.sh を実行してください"
    exit 1
fi

# モデルの作成
echo "🔧 Ollamaモデルの作成"

if ! ollama list | grep -q "llama-elyzsa-jp-8b"; then
    echo "📝 llama-elyzsa-jp-8b モデルを作成中..."
    ollama create llama-elyzsa-jp-8b -f models/llama-elyzsa-jp-8b/Modelfile
    if [ $? -eq 0 ]; then
        echo "✅ llama-elyzsa-jp-8b モデル作成完了"
    else
        echo "❌ llama-elyzsa-jp-8b モデル作成失敗"
        exit 1
    fi
else
    echo "✅ llama-elyzsa-jp-8b モデルは既に存在"
fi

if ! ollama list | grep -q "mistral-nemo-jp"; then
    echo "📝 mistral-nemo-jp モデルを作成中..."
    ollama create mistral-nemo-jp -f models/mistral-nemo-jp/Modelfile
    if [ $? -eq 0 ]; then
        echo "✅ mistral-nemo-jp モデル作成完了"
    else
        echo "❌ mistral-nemo-jp モデル作成失敗"
        exit 1
    fi
else
    echo "✅ mistral-nemo-jp モデルは既に存在"
fi

if ! ollama list | grep -q "nomic-embed-text"; then
    echo "📥 nomic-embed-text Embeddingモデルをダウンロード中..."
    ollama pull nomic-embed-text
    if [ $? -eq 0 ]; then
        echo "✅ nomic-embed-text ダウンロード完了"
    else
        echo "❌ nomic-embed-text ダウンロード失敗"
        exit 1
    fi
else
    echo "✅ nomic-embed-text は既に存在"
fi

echo ""
echo "🎉 Ollamaセットアップ完了!"
echo ""
echo "📊 利用可能なモデル:"
ollama list