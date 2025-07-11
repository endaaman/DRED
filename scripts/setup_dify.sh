#!/bin/bash

echo "🐳 Difyセットアップ"
echo "================"

# プロジェクトルートかチェック
if [ ! -f "../CLAUDE.md" ] && [ ! -f "CLAUDE.md" ]; then
    echo "❌ エラー: プロジェクトルートまたはscriptsディレクトリで実行してください"
    exit 1
fi

# プロジェクトルートに移動
if [ -f "../CLAUDE.md" ]; then
    cd ..
fi

# Dockerの確認
if ! command -v docker &> /dev/null; then
    echo "❌ Dockerがインストールされていません"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Composeがインストールされていません"
    exit 1
fi

echo "✅ Docker & Docker Compose確認済み"

cd dify-setup

# 必要なディレクトリの事前作成（権限問題対策）
echo "📁 必要なディレクトリを作成中..."
mkdir -p volumes/{app/storage,db/data,redis/data,weaviate,plugin_daemon,sandbox}
echo "✅ ボリュームディレクトリ作成完了（rootにならない対策）"

# .env設定
if ! grep -q "EXPOSE_NGINX_PORT" .env 2>/dev/null; then
    echo "⚙️  ポート設定を追加中..."
    echo "EXPOSE_NGINX_PORT=8080" >> .env
    echo "✅ ポート8080に設定完了"
else
    echo "✅ ポート設定は既に存在"
fi

echo ""
echo "🚀 Difyサービス起動中..."
docker compose up -d

# 起動確認
sleep 5
echo ""
echo "📊 サービス状況:"
docker compose ps

echo ""
echo "🎉 Difyセットアップ完了!"
echo ""
echo "📍 アクセス情報:"
echo "   - Dify管理画面: http://localhost:8080/install"
echo ""
echo "🛠️  管理コマンド:"
echo "   - ログ確認: docker compose logs -f"
echo "   - 停止: docker compose down"
echo "   - 再起動: docker compose restart"

if docker compose ps | grep -q "Restarting"; then
    echo ""
    echo "⚠️  一部のコンテナが再起動中です"
    echo "💡 nginx/sandbox/ssrf_proxyで問題が出る場合:"
    echo "   docker compose stop nginx ssrf_proxy sandbox"
fi

cd ..
