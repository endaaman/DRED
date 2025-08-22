#!/usr/bin/env python3
"""
Map-Reduce質問応答システム - Aggregate処理

機能:
- 複数ドキュメントに対するsingle_qa並列実行
- 結果の統合とaggregate処理
- 実行管理とログ出力
"""

import asyncio
import concurrent.futures
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from document_indexer import DocumentIndexer
from execution_manager import ExecutionManager
from single_doc_qa import single_document_qa, load_prompt_template


def create_aggregate_prompt(question: str, single_results: List[Dict[str, Any]], 
                          template_name: str = "consensus") -> str:
    """
    aggregate用プロンプトを作成
    
    Args:
        question: 元の質問
        single_results: single_qa結果のリスト
        template_name: aggregateテンプレート名
        
    Returns:
        str: aggregateプロンプト
    """
    # aggregate用プロンプトテンプレート読み込み
    template_path = Path(__file__).parent / "prompts" / "aggregate_qa" / f"{template_name}.txt"
    
    if not template_path.exists():
        # デフォルトテンプレートを使用
        template = """以下は複数の行政文書から得られた回答です。これらを統合して、最も適切で包括的な回答を作成してください。

元の質問: 「{question}」

各文書からの回答:
{document_answers}

上記の回答を分析し、以下の形式で統合回答を作成してください：

## 統合回答

[最も適切で包括的な回答]

## 根拠・出典

[どの文書のどの部分に基づいているかを明記]

## 注意事項

[制約事項や追加で確認が必要な事項があれば記載]
"""
    else:
        template = template_path.read_text(encoding='utf-8')
    
    # 各文書の回答を整理
    document_answers = []
    for i, result in enumerate(single_results):
        doc_name = Path(result['document_path']).stem
        answer = result['answer']
        document_answers.append(f"【文書{i+1}: {doc_name}】\n{answer}\n")
    
    return template.format(
        question=question,
        document_answers='\n'.join(document_answers)
    )


def run_single_qa_batch(documents: List[Dict[str, Any]], question: str, 
                       template_name: str, max_workers: int = 3,
                       show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    single_qaを並列実行
    
    Args:
        documents: 対象ドキュメントリスト
        question: 質問内容
        template_name: single_qa用テンプレート名
        max_workers: 最大並列数
        show_progress: プログレスバー表示フラグ
        
    Returns:
        List[Dict]: single_qa結果のリスト
    """
    def run_single_qa(doc_info, pbar=None):
        """単一ドキュメントでsingle_qaを実行"""
        try:
            # サイレントモードを設定（ログ混雑回避）
            import single_doc_qa
            single_doc_qa._SILENT_MODE = show_progress
            
            result = single_document_qa(doc_info['path'], question, template_name)
            
            if pbar:
                pbar.set_postfix_str(f"{doc_info['filename'][:20]}...")
                pbar.update(1)
            
            return result
        except Exception as e:
            if pbar:
                pbar.set_postfix_str(f"ERROR: {doc_info['filename'][:15]}...")
                pbar.update(1)
            
            # エラー時はダミー結果を返す
            return {
                'document_path': doc_info['path'],
                'question': question,
                'template': template_name,
                'answer': f"エラー: {str(e)}",
                'metadata': {'error': True}
            }
    
    results = []
    
    # プログレスバー付き並列実行
    if show_progress:
        from tqdm import tqdm
        with tqdm(total=len(documents), desc="Single QA", ncols=100) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 全タスクを投入
                future_to_doc = {
                    executor.submit(run_single_qa, doc, pbar): doc 
                    for doc in documents
                }
                
                # 結果を順次収集
                for future in concurrent.futures.as_completed(future_to_doc):
                    result = future.result()
                    results.append(result)
    else:
        # プログレスバーなしの実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全タスクを投入
            future_to_doc = {
                executor.submit(run_single_qa, doc): doc 
                for doc in documents
            }
            
            # 結果を順次収集
            for future in concurrent.futures.as_completed(future_to_doc):
                result = future.result()
                results.append(result)
    
    # 元の文書順序に合わせてソート
    doc_path_to_index = {doc['path']: doc['index'] for doc in documents}
    results.sort(key=lambda x: doc_path_to_index.get(x['document_path'], 999))
    
    return results


def run_aggregate_qa(question: str, single_template: str = "baseline",
                    aggregate_template: str = "consensus", parallel: int = 3,
                    subdir_filter: List[str] = None) -> str:
    """
    Map-Reduce質問応答の完全実行
    
    Args:
        question: 質問内容
        single_template: single_qa用テンプレート
        aggregate_template: aggregate用テンプレート
        parallel: 並列実行数
        subdir_filter: 対象サブディレクトリフィルタ
        
    Returns:
        str: 実行ID
    """
    # 実行管理開始
    exec_manager = ExecutionManager()
    run_id = exec_manager.create_run()
    
    print(f"実行開始 - Run ID: {run_id}", file=sys.stderr)
    
    try:
        # ドキュメントスキャン
        indexer = DocumentIndexer()
        documents = indexer.scan_documents()
        
        if subdir_filter:
            documents = indexer.filter_by_subdir(documents, subdir_filter)
            print(f"サブディレクトリフィルタ適用: {subdir_filter}", file=sys.stderr)
        
        print(f"対象ドキュメント数: {len(documents)}", file=sys.stderr)
        
        # メタデータ更新
        exec_manager.update_metadata(run_id, {
            'status': 'running',
            'parameters': {
                'question': question,
                'single_template': single_template,
                'aggregate_template': aggregate_template,
                'parallel': parallel,
                'subdir_filter': subdir_filter
            },
            'documents': [{'index': doc['index'], 'path': doc['relative_path']} 
                         for doc in documents]
        })
        
        # Single QA並列実行
        print("Single QA実行開始...", file=sys.stderr)
        single_results = run_single_qa_batch(documents, question, single_template, parallel)
        
        # Single QA結果保存
        for result in single_results:
            doc_index = next((doc['index'] for doc in documents 
                            if doc['path'] == result['document_path']), 0)
            doc_path = Path(result['document_path'])
            exec_manager.save_single_qa_result(run_id, doc_index, doc_path, result)
        
        # Aggregate実行
        print("Aggregate処理開始...", file=sys.stderr)
        aggregate_prompt = create_aggregate_prompt(question, single_results, aggregate_template)
        
        # LLMでaggregate処理
        from single_doc_qa import query_llm
        aggregate_answer, aggregate_metadata = query_llm(aggregate_prompt)
        
        # Aggregate結果保存
        aggregate_result = f"""=== MAP-REDUCE質問応答結果 ===

実行ID: {run_id}
実行日時: {datetime.now().isoformat()}

質問: {question}

パラメータ:
- Single Template: {single_template}
- Aggregate Template: {aggregate_template}  
- 並列数: {parallel}
- 対象文書数: {len(documents)}

=== 統合回答 ===

{aggregate_answer}

=== 実行統計 ===

Aggregateトークン使用量: {aggregate_metadata.get('total_tokens', 'N/A')}
処理対象文書:
"""
        
        for doc in documents:
            aggregate_result += f"  {doc['index']:2d}: {doc['subdir']}/{doc['filename']}\n"
        
        exec_manager.save_aggregate_result(run_id, aggregate_result)
        
        # メタデータ最終更新
        exec_manager.update_metadata(run_id, {
            'status': 'completed',
            'results': {
                'single_qa_count': len(single_results),
                'aggregate_tokens': aggregate_metadata.get('total_tokens'),
                'total_processing_time': 'N/A'  # 実装時に計算
            }
        })
        
        print(f"実行完了 - Run ID: {run_id}", file=sys.stderr)
        return run_id
        
    except Exception as e:
        # エラー時の処理
        exec_manager.update_metadata(run_id, {
            'status': 'failed',
            'error': str(e)
        })
        raise e


def main():
    """コマンドライン実行のメイン処理"""
    parser = argparse.ArgumentParser(
        description="Map-Reduce質問応答システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python aggregate_qa.py "３年前に家を相続した。売却のため、土地の上の建屋を取り壊したい。利用出来る支援金等は何かあるか？" --single-template structured --parallel 2 --subdir 空き家
        """
    )
    
    parser.add_argument("question", help="質問内容")
    parser.add_argument("--single-template", default="baseline",
                       help="single_qa用プロンプトテンプレート (default: baseline)")
    parser.add_argument("--aggregate-template", default="consensus", 
                       help="aggregate用プロンプトテンプレート (default: consensus)")
    parser.add_argument("--parallel", type=int, default=3,
                       help="並列実行数 (default: 3)")
    parser.add_argument("--subdir", action="append",
                       help="対象サブディレクトリ（複数指定可）")
    parser.add_argument("--run-id", 
                       help="既存の実行結果を表示")
    parser.add_argument("--list-runs", action="store_true",
                       help="実行履歴一覧を表示")
    
    args = parser.parse_args()
    
    try:
        if args.list_runs:
            # 実行履歴表示
            exec_manager = ExecutionManager()
            runs = exec_manager.list_runs()
            print("実行履歴:")
            for run_id in runs:
                summary = exec_manager.get_run_summary(run_id)
                print(f"  {run_id}: {summary['status']} "
                      f"({summary['single_qa_count']} docs)")
                      
        elif args.run_id:
            # 既存結果表示
            exec_manager = ExecutionManager()
            result_path = exec_manager.get_run_dir(args.run_id) / "aggregate_result.txt"
            if result_path.exists():
                print(result_path.read_text(encoding='utf-8'))
            else:
                print(f"結果ファイルが見つかりません: {args.run_id}")
                
        else:
            # Map-Reduce実行
            run_id = run_aggregate_qa(
                args.question, 
                args.single_template,
                args.aggregate_template,
                args.parallel,
                args.subdir
            )
            
            # 結果表示
            exec_manager = ExecutionManager()
            result_path = exec_manager.get_run_dir(run_id) / "aggregate_result.txt"
            print(result_path.read_text(encoding='utf-8'))
            
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()