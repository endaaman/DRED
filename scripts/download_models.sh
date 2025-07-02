#!/bin/bash

echo "📦 モデルファイルダウンロード"
echo "=========================="

# プロジェクトルートかチェック
if [ ! -f "../CLAUDE.md" ] && [ ! -f "CLAUDE.md" ]; then
    echo "❌ エラー: プロジェクトルートまたはscriptsディレクトリで実行してください"
    exit 1
fi

# プロジェクトルートに移動
if [ -f "../CLAUDE.md" ]; then
    cd ..
fi

echo "📁 モデルファイルの確認とダウンロード"

if [ ! -f "models/llama-elyzsa-jp-8b/Llama-3-ELYZA-JP-8B.Q4_K_M.gguf" ]; then
    echo "📥 LLaMa3 ELYZA JP 8B をダウンロード中..."
    wget -P models/llama-elyzsa-jp-8b/ \
      https://huggingface.co/QuantFactory/Llama-3-ELYZA-JP-8B-GGUF/resolve/main/Llama-3-ELYZA-JP-8B.Q4_K_M.gguf
    if [ $? -eq 0 ]; then
        echo "✅ LLaMa3 ELYZA JP 8B ダウンロード完了"
    else
        echo "❌ LLaMa3 ELYZA JP 8B ダウンロード失敗"
        exit 1
    fi
else
    echo "✅ LLaMa3 ELYZA JP 8B は既に存在"
fi

if [ ! -f "models/mistral-nemo-jp/Mistral-Nemo-Japanese-Instruct-2408.Q4_K_M.gguf" ]; then
    echo "📥 Mistral Nemo JP をダウンロード中..."
    wget -P models/mistral-nemo-jp/ \
      https://huggingface.co/QuantFactory/Mistral-Nemo-Japanese-Instruct-2408-GGUF/resolve/main/Mistral-Nemo-Japanese-Instruct-2408.Q4_K_M.gguf
    if [ $? -eq 0 ]; then
        echo "✅ Mistral Nemo JP ダウンロード完了"
    else
        echo "❌ Mistral Nemo JP ダウンロード失敗"
        exit 1
    fi
else
    echo "✅ Mistral Nemo JP は既に存在"
fi

echo ""
echo "🎉 モデルファイルダウンロード完了!"
echo ""
echo "📊 ダウンロード済みファイル:"
ls -lh models/*/Llama-3-ELYZA-JP-8B.Q4_K_M.gguf 2>/dev/null
ls -lh models/*/Mistral-Nemo-Japanese-Instruct-2408.Q4_K_M.gguf 2>/dev/null