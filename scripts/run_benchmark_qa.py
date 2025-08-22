#!/usr/bin/env python3
"""
模範QA自動実行スクリプト
data/QA/のExcelファイルから質問を読み込み、Map-Reduce Q&Aを順次実行
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import time
import argparse

def load_questions(excel_path: Path) -> list:
    """Excelファイルから質問列を読み込み"""
    try:
        # ヘッダーが3行目にある（0-indexed で 2）
        df = pd.read_excel(excel_path, header=2)
        if '質問' not in df.columns:
            print(f"❌ エラー: {excel_path} に '質問' 列が見つかりません", file=sys.stderr)
            print(f"   利用可能な列: {df.columns.tolist()}", file=sys.stderr)
            return []
        
        questions = df['質問'].dropna().tolist()
        return questions
    except Exception as e:
        print(f"❌ エラー: {excel_path} の読み込みに失敗: {e}", file=sys.stderr)
        return []

def run_aggregate_qa(question: str, run_id: str, single_template: str = "legal_sandwich", 
                    aggregate_template: str = "focused", model: str = None) -> bool:
    """aggregate_qa.pyを実行"""
    cmd = [
        "uv", "run", "python", "map_reduce/aggregate_qa.py",
        "--parallel", "3",
        "--single-template", single_template,
        "--aggregate-template", aggregate_template,
        "--run-id", run_id,
        question
    ]
    
    # モデル指定がある場合は環境変数で渡す
    env = None
    if model:
        import os
        env = os.environ.copy()
        env['OLLAMA_MODEL'] = model
    
    # 実行コマンドを表示（デバッグ用にコピペ可能）
    cmd_str = ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
    print(f"\n🔧 実行コマンド:", file=sys.stderr)
    print(f"   {cmd_str}", file=sys.stderr)
    
    print(f"📝 実行中: {run_id}", file=sys.stderr)
    print(f"   質問: {question[:50]}...", file=sys.stderr)
    
    try:
        # 進捗表示を見るためcapture_output=Falseに変更
        result = subprocess.run(cmd, text=True, env=env)
        if result.returncode == 0:
            print(f"✅ 完了: {run_id}", file=sys.stderr)
            return True
        else:
            print(f"❌ 失敗: {run_id} (exit code: {result.returncode})", file=sys.stderr)
            return False
    except Exception as e:
        print(f"❌ 実行エラー: {e}", file=sys.stderr)
        return False

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="模範QA自動実行スクリプト")
    parser.add_argument("--single-template", default="legal_sandwich", 
                       help="Single QA用テンプレート (default: legal_sandwich)")
    parser.add_argument("--aggregate-template", default="focused", 
                       help="Aggregate用テンプレート (default: focused)")
    parser.add_argument("--model", help="使用するモデル名 (環境変数OLLAMA_MODELで指定)")
    parser.add_argument("--category", choices=["空き家", "立地適正化計画"], 
                       help="実行するカテゴリを指定 (指定しない場合は全て実行)")
    
    args = parser.parse_args()
    
    qa_dir = Path("data/QA")
    
    if not qa_dir.exists():
        print(f"❌ エラー: {qa_dir} が見つかりません", file=sys.stderr)
        sys.exit(1)
    
    # QAファイルのリスト
    qa_files = {
        "空き家": qa_dir / "QA_空き家.xlsx",
        "立地適正化計画": qa_dir / "QA_立地適正化計画.xlsx"
    }
    
    # カテゴリフィルタがある場合
    if args.category:
        qa_files = {args.category: qa_files[args.category]}
    
    total_count = 0
    success_count = 0
    
    print("=" * 60, file=sys.stderr)
    print("模範QA自動実行開始", file=sys.stderr)
    print(f"Single Template: {args.single_template}", file=sys.stderr)
    print(f"Aggregate Template: {args.aggregate_template}", file=sys.stderr)
    if args.model:
        print(f"Model: {args.model}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    start_time = time.time()
    
    for category, excel_path in qa_files.items():
        if not excel_path.exists():
            print(f"⚠️  {excel_path} が見つかりません。スキップします。", file=sys.stderr)
            continue
        
        print(f"\n【{category}】", file=sys.stderr)
        questions = load_questions(excel_path)
        
        if not questions:
            print(f"  質問が見つかりません", file=sys.stderr)
            continue
        
        print(f"  質問数: {len(questions)}", file=sys.stderr)
        
        for i, question in enumerate(questions, 1):
            run_id = f"{category}_Q{i}"
            total_count += 1
            
            if run_aggregate_qa(question, run_id, args.single_template, 
                               args.aggregate_template, args.model):
                success_count += 1
            
            # 短い休憩（Ollama負荷軽減）
            if i < len(questions):
                time.sleep(2)
    
    # 実行結果サマリー
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60, file=sys.stderr)
    print("実行完了", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"総実行数: {total_count}", file=sys.stderr)
    print(f"成功: {success_count}", file=sys.stderr)
    print(f"失敗: {total_count - success_count}", file=sys.stderr)
    print(f"実行時間: {elapsed_time:.1f}秒", file=sys.stderr)
    
    if success_count < total_count:
        sys.exit(1)

if __name__ == "__main__":
    main()