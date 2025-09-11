#!/bin/bash

echo "🧹 DRED プロジェクトクリーンアップスクリプト"
echo "=============================================="

# 現在のディレクトリがプロジェクトルートかチェック
if [ ! -f "CLAUDE.md" ] || [ ! -d "dify_setup" ]; then
    echo "❌ エラー: プロジェクトルートディレクトリで実行してください"
    exit 1
fi

echo ""
echo "📍 実行場所: $(pwd)"
echo ""

# Dockerコンテナとネットワークの停止・削除
echo "🐳 Dockerコンテナの停止と削除..."
cd dify_setup
if [ -f "docker-compose.yaml" ]; then
    docker compose down --remove-orphans --volumes 2>/dev/null || true

    # 強制的にコンテナを削除
    CONTAINERS=$(docker ps -a --filter "name=dify_setup" --format "{{.Names}}" 2>/dev/null)
    if [ -n "$CONTAINERS" ]; then
        echo "🗑️  残存コンテナを削除中..."
        echo "$CONTAINERS" | xargs -r docker rm -f
    fi

    # ネットワークの削除
    NETWORKS=$(docker network ls --filter "name=dify_setup" --format "{{.Name}}" 2>/dev/null)
    if [ -n "$NETWORKS" ]; then
        echo "🌐 残存ネットワークを削除中..."
        echo "$NETWORKS" | xargs -r docker network rm 2>/dev/null || true
    fi

    echo "✅ Dockerコンテナとネットワークの削除完了"
else
    echo "⚠️  docker-compose.yamlが見つかりません"
fi

cd ..

# root権限で作成されたディレクトリの確認と削除
echo ""
echo "🔍 root権限ディレクトリの確認..."

ROOT_DIRS=""

# dify_setup下のroot権限ディレクトリをチェック
if [ -d "dify_setup" ]; then
    # よくあるroot権限ディレクトリ
    DIRS_TO_CHECK=(
        "dify_setup/volumes"
        "dify_setup/nginx/conf.d"
        "dify_setup/logs"
        "dify_setup/storage"
    )

    for dir in "${DIRS_TO_CHECK[@]}"; do
        if [ -d "$dir" ]; then
            OWNER=$(stat -c '%U' "$dir" 2>/dev/null || echo "unknown")
            if [ "$OWNER" = "root" ] || [ ! -w "$dir" ]; then
                ROOT_DIRS="$ROOT_DIRS $dir"
            fi
        fi
    done

    # ファイルもチェック
    if [ -f "dify_setup/.env" ]; then
        OWNER=$(stat -c '%U' "dify_setup/.env" 2>/dev/null || echo "unknown")
        if [ "$OWNER" = "root" ] || [ ! -w "dify_setup/.env" ]; then
            ROOT_DIRS="$ROOT_DIRS dify_setup/.env"
        fi
    fi
fi

if [ -n "$ROOT_DIRS" ]; then
    echo "🔒 以下のroot権限ファイル/ディレクトリが見つかりました:"
    for dir in $ROOT_DIRS; do
        echo "   - $dir"
    done
    echo ""
    echo "⚠️  これらのファイル/ディレクトリを削除するには管理者権限が必要です"
    echo "💡 以下のコマンドを手動で実行してください:"
    echo ""
    echo "sudo rm -rf$ROOT_DIRS"
    echo ""
else
    echo "✅ root権限のファイル/ディレクトリは見つかりませんでした"
fi

# Ollamaモデルの削除（オプション）
echo ""
read -p "🤖 Ollamaモデルも削除しますか？ (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Ollamaモデルを削除中..."
    ollama rm llama-elyzsa-jp-8b 2>/dev/null || echo "   - llama-elyzsa-jp-8b は存在しないか削除済み"
    ollama rm mistral-nemo-jp 2>/dev/null || echo "   - mistral-nemo-jp は存在しないか削除済み"
    ollama rm nomic-embed-text 2>/dev/null || echo "   - nomic-embed-text は存在しないか削除済み"
    echo "✅ Ollamaモデルの削除完了"
else
    echo "⏭️  Ollamaモデルはそのまま残します"
fi

# .envファイルのバックアップと削除
echo ""
if [ -f "dify_setup/.env" ]; then
    if [ -w "dify_setup/.env" ]; then
        echo "💾 .envファイルをバックアップ中..."
        cp "dify_setup/.env" "dify_setup/.env.backup.$(date +%Y%m%d_%H%M%S)"
        rm "dify_setup/.env"
        echo "✅ .envファイルをバックアップして削除しました"
    else
        echo "⚠️  .envファイルは削除できません（権限不足）"
    fi
fi

# 削除可能なディレクトリを削除
echo ""
echo "🧼 一般ファイルのクリーンアップ..."
rm -rf dify_setup/volumes 2>/dev/null || true
rm -rf dify_setup/storage 2>/dev/null || true
rm -rf dify_setup/logs 2>/dev/null || true
rm -f dify_setup/nginx.conf 2>/dev/null || true

echo ""
echo "🎉 クリーンアップ完了!"
echo ""
echo "📋 残っているファイル:"
echo "   - Modelfile (models/)"
echo "   - docker-compose.yaml, nginx設定"
echo "   - README.md, CLAUDE.md"
echo ""
echo "🔄 再セットアップの場合は README.md の手順に従ってください"

if [ -n "$ROOT_DIRS" ]; then
    echo ""
    echo "⚠️  root権限ファイルがある場合は手動削除してください:"
    echo "   sudo rm -rf$ROOT_DIRS"
fi
