#!/usr/bin/env python3
"""
Map-Reduce結果と模範解答の比較・結合スクリプト

benchmark実行結果と模範解答（Excelファイル）を結合し、
比較可能な形式で出力する
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

def load_reference_answers(excel_path: Path) -> Dict[int, str]:
    """模範解答をExcelから読み込み"""
    try:
        # ヘッダーが3行目（0-indexed で 2）
        df = pd.read_excel(excel_path, header=2)
        
        # 実際の列名に合わせて調整
        question_col = '質問'
        answer_col = '回答'  # '模範解答' ではなく '回答'
        process_col = '過程（対応思考、参照文献、参照箇所）'
        
        if question_col not in df.columns or answer_col not in df.columns:
            available_cols = df.columns.tolist()
            print(f"❌ 必要な列が見つかりません", file=sys.stderr)
            print(f"   必要: ['{question_col}', '{answer_col}']", file=sys.stderr)  
            print(f"   利用可能: {available_cols}", file=sys.stderr)
            return {}
        
        reference_answers = {}
        for idx, row in df.iterrows():
            if pd.notna(row[question_col]) and pd.notna(row[answer_col]):
                # 質問番号は1-indexed
                reference_answers[idx + 1] = {
                    'question': str(row[question_col]).strip(),
                    'reference_answer': str(row[answer_col]).strip(),
                    'process': str(row[process_col]).strip() if pd.notna(row.get(process_col)) else ""
                }
        
        return reference_answers
        
    except Exception as e:
        print(f"❌ Excel読み込みエラー: {e}", file=sys.stderr)
        return {}

def load_benchmark_result(run_dir: Path) -> Optional[Dict]:
    """benchmark実行結果を読み込み"""
    try:
        # メタデータ読み込み
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 統合回答読み込み
        aggregate_path = run_dir / "aggregate_result.txt"
        aggregate_answer = ""
        if aggregate_path.exists():
            with open(aggregate_path, 'r', encoding='utf-8') as f:
                aggregate_answer = f.read()
        
        # Single QA結果読み込み（.txtファイル）
        single_qa_dir = run_dir / "single_qa"
        single_results = []
        if single_qa_dir.exists():
            for txt_file in sorted(single_qa_dir.glob("*.txt")):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    single_results.append({
                        'filename': txt_file.name,
                        'content': content
                    })
        
        return {
            'metadata': metadata,
            'aggregate_answer': aggregate_answer,
            'single_results': single_results
        }
        
    except Exception as e:
        print(f"❌ benchmark結果読み込みエラー ({run_dir}): {e}", file=sys.stderr)
        return None

def extract_question_number(run_id: str) -> Optional[int]:
    """run_idから質問番号を抽出（例: "空き家_Q1" -> 1）"""
    try:
        if '_Q' in run_id:
            return int(run_id.split('_Q')[1])
        return None
    except:
        return None

def create_combined_excel(category: str, reference_answers: Dict[int, str], 
                         benchmark_results: Dict[str, Dict]) -> pd.DataFrame:
    """結合データをExcel用DataFrameとして作成"""
    
    rows = []
    
    # 質問ごとに比較データ作成
    for q_num in sorted(reference_answers.keys()):
        ref_data = reference_answers[q_num]
        
        # 対応するbenchmark結果を探す
        matching_run_id = None
        benchmark_data = None
        
        for run_id, result_data in benchmark_results.items():
            if extract_question_number(run_id) == q_num:
                matching_run_id = run_id
                benchmark_data = result_data
                break
        
        # 1行のデータを作成
        row = {
            'NO': q_num,
            '質問': ref_data['question'],
            '模範解答': ref_data['reference_answer'],
            '模範解答_過程': ref_data.get('process', ''),
        }
        
        if benchmark_data:
            metadata = benchmark_data['metadata']
            timing = metadata.get('results', {}).get('timing', {})
            
            # aggregate_answer の内容処理
            aggregate_answer = benchmark_data['aggregate_answer']
            
            row.update({
                'システム回答': aggregate_answer,
                '実行ID': matching_run_id,
                '実行時間_秒': round(timing.get('total_time', 0), 1),
                'トークン数': metadata.get('results', {}).get('total_tokens', 0),
                'Single_QA平均時間_秒': round(timing.get('single_qa_avg_time', 0), 1),
                '集約時間_秒': round(timing.get('aggregate_time', 0), 1)
            })
        else:
            row.update({
                'システム回答': '❌ benchmark結果なし',
                '実行ID': '',
                '実行時間_秒': 0,
                'トークン数': 0,
                'Single_QA平均時間_秒': 0,
                '集約時間_秒': 0
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="benchmark結果と模範解答を結合")
    parser.add_argument("--category", choices=["空き家", "立地適正化計画"], 
                       help="処理するカテゴリ")
    parser.add_argument("--output-dir", type=Path, default="evaluation",
                       help="出力ディレクトリ (デフォルト: evaluation)")
    
    args = parser.parse_args()
    
    # QAディレクトリ
    qa_dir = Path("data/QA")
    run_dir = Path("run")
    
    if not qa_dir.exists() or not run_dir.exists():
        print("❌ data/QA または run ディレクトリが見つかりません", file=sys.stderr)
        sys.exit(1)
    
    # カテゴリ指定がない場合は対話的に選択
    category = args.category
    if not category:
        categories = ["空き家", "立地適正化計画"]
        print("カテゴリを選択してください:")
        for i, cat in enumerate(categories, 1):
            print(f"  {i}. {cat}")
        
        try:
            choice = int(input("番号を入力: ")) - 1
            category = categories[choice]
        except (ValueError, IndexError):
            print("❌ 無効な選択です", file=sys.stderr)
            sys.exit(1)
    
    print(f"📊 カテゴリ: {category}")
    
    # 模範解答読み込み
    excel_path = qa_dir / f"QA_{category}.xlsx"
    reference_answers = load_reference_answers(excel_path)
    
    if not reference_answers:
        print(f"❌ {excel_path} から模範解答を読み込めませんでした", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ 模範解答: {len(reference_answers)}件")
    
    # benchmark結果読み込み
    benchmark_results = {}
    pattern = f"{category}_Q*"
    
    for run_path in run_dir.glob(pattern):
        if run_path.is_dir():
            result_data = load_benchmark_result(run_path)
            if result_data:
                benchmark_results[run_path.name] = result_data
    
    print(f"✅ benchmark結果: {len(benchmark_results)}件")
    
    # 結合データフレーム作成
    combined_df = create_combined_excel(category, reference_answers, benchmark_results)
    
    # 出力
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"combined_{category}.xlsx"
    
    # CSVが主力、Excelは参考用として出力
    
    # まずCSV形式で出力してテスト
    csv_path = output_dir / f"combined_{category}.csv"
    combined_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV結合レポートを出力: {csv_path}")
    
    # Excel形式で出力 - xlsxwriterエンジンを使用
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            combined_df.to_excel(writer, sheet_name='比較結果', index=False)
            
            # xlsxwriterでの列幅調整
            worksheet = writer.sheets['比較結果']
            for idx, col in enumerate(combined_df.columns):
                # 列のデータから最大長を計算
                max_len = max(combined_df[col].astype(str).map(len).max(), len(col))
                # 適度な幅に調整（最大50文字）
                worksheet.set_column(idx, idx, min(max_len + 2, 50))
    except Exception as e:
        print(f"❌ Excel出力エラー: {e}", file=sys.stderr)
    
    print(f"✅ Excel結合レポートを出力: {output_path}")
    
    # 要約統計
    matched_count = sum(1 for q_num in reference_answers.keys() 
                       if any(extract_question_number(run_id) == q_num 
                             for run_id in benchmark_results.keys()))
    
    print(f"\n📈 マッチング統計:")
    print(f"   模範解答: {len(reference_answers)}件")
    print(f"   benchmark: {len(benchmark_results)}件")
    print(f"   マッチ: {matched_count}件")

if __name__ == "__main__":
    main()