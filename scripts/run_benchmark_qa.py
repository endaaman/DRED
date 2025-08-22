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

def run_aggregate_qa(question: str, run_id: str) -> bool:
    """aggregate_qa.pyを実行"""
    cmd = [
        "uv", "run", "python", "map_reduce/aggregate_qa.py",
        "--parallel", "3",
        "--force-run-id", run_id,
        question
    ]
    
    # 実行コマンドを表示（デバッグ用にコピペ可能）
    cmd_str = ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
    print(f"\n🔧 実行コマンド:", file=sys.stderr)
    print(f"   {cmd_str}", file=sys.stderr)
    
    print(f"📝 実行中: {run_id}", file=sys.stderr)
    print(f"   質問: {question[:50]}...", file=sys.stderr)
    
    try:
        # 進捗表示を見るためcapture_output=Falseに変更
        result = subprocess.run(cmd, text=True)
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
    qa_dir = Path("data/QA")
    
    if not qa_dir.exists():
        print(f"❌ エラー: {qa_dir} が見つかりません", file=sys.stderr)
        sys.exit(1)
    
    # QAファイルのリスト
    qa_files = {
        "空き家": qa_dir / "QA_空き家.xlsx",
        "立地適正化計画": qa_dir / "QA_立地適正化計画.xlsx"
    }
    
    total_count = 0
    success_count = 0
    
    print("=" * 60, file=sys.stderr)
    print("模範QA自動実行開始", file=sys.stderr)
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
            
            if run_aggregate_qa(question, run_id):
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