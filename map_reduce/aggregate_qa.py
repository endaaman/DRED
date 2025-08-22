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
import time
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
        available_templates = list((Path(__file__).parent / "prompts" / "aggregate_qa").glob("*.txt"))
        available_names = [t.stem for t in available_templates]
        raise FileNotFoundError(
            f"Aggregate template '{template_name}' not found. "
            f"Available templates: {', '.join(available_names)}"
        )
    
    template = template_path.read_text(encoding='utf-8')
    
    # 各文書の回答を整理（ドキュメント情報を完全保持）
    document_answers = []
    for i, result in enumerate(single_results):
        doc_path = Path(result['document_path'])
        # サブディレクトリ/ファイル名の形式で表示
        filename = doc_path.stem
        if len(doc_path.parts) >= 2:
            subdir = doc_path.parts[-2]
        else:
            subdir = "root"
        
        answer = result['answer']
        
        # structured形式の回答から関連度と確度を抽出
        relevance = "不明"
        confidence = "不明"
        if "**関連度**:" in answer:
            for line in answer.split('\n'):
                if "**関連度**:" in line:
                    relevance = line.split(':')[1].strip()
                elif "**確度**:" in line:
                    confidence = line.split(':')[1].strip()
        
        document_answers.append(f"""
=== ドキュメント #{i+1}: {subdir}/{filename} ===
関連度: {relevance}
確度: {confidence}

{answer}
=== ドキュメント #{i+1} 終了 ===
""")
    
    return template.format(
        question=question,
        document_answers='\n'.join(document_answers)
    )


def run_single_qa_batch(documents: List[Dict[str, Any]], question: str, 
                       template_name: str, max_workers: int = 3,
                       show_progress: bool = True, run_id: str = None) -> List[Dict[str, Any]]:
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
        # サイレントモードを設定（ログ混雑回避）
        import single_doc_qa
        single_doc_qa._SILENT_MODE = True
        
        # Single QA実行
        result = single_document_qa(doc_info['path'], question, template_name)
        
        # run_idが指定されていれば逐次保存
        if run_id:
            from execution_manager import ExecutionManager
            exec_manager = ExecutionManager()
            save_paths = exec_manager.save_single_qa_result(
                run_id, doc_info['index'], Path(doc_info['path']), result
            )
            if pbar:
                pbar.set_postfix_str(f"保存: {doc_info['filename'][:15]}...")
        
        if pbar:
            pbar.set_postfix_str(f"{doc_info['filename'][:20]}...")
            pbar.update(1)
        
        return result
    
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


def _setup_execution(question: str, single_template: str, aggregate_template: str,
                    parallel: int, subdir_filter: List[str], run_id: str = None) -> tuple[ExecutionManager, str, List[Dict[str, Any]]]:
    """
    実行環境のセットアップ
    
    Returns:
        tuple: (exec_manager, run_id, documents)
    """
    exec_manager = ExecutionManager()
    run_id = exec_manager.create_run(run_id=run_id)
    
    print(f"実行開始 - Run ID: {run_id}", file=sys.stderr)
    
    try:
        # ドキュメントスキャン
        indexer = DocumentIndexer()
        documents = indexer.scan_documents()
        
        if subdir_filter:
            documents = indexer.filter_by_subdir(documents, subdir_filter)
            print(f"サブディレクトリフィルタ適用: {subdir_filter}", file=sys.stderr)
        
        print(f"対象ドキュメント数: {len(documents)}", file=sys.stderr)
        
        # メタデータ初期化
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
        
        return exec_manager, run_id, documents
        
    except Exception as e:
        # セットアップ段階でのエラー
        exec_manager.update_metadata(run_id, {
            'status': 'setup_failed',
            'error': f"Setup error: {str(e)}"
        })
        raise RuntimeError(f"実行環境セットアップエラー: {e}") from e


def _execute_single_qa_phase(exec_manager: ExecutionManager, run_id: str, 
                            documents: List[Dict[str, Any]], question: str,
                            single_template: str, parallel: int) -> tuple[List[Dict[str, Any]], float]:
    """
    Single QA フェーズの実行
    
    Returns:
        tuple: (single_results, single_total_time)
    """
    print("Single QA実行開始...", file=sys.stderr)
    single_start_time = time.time()
    
    try:
        single_results = run_single_qa_batch(
            documents, question, single_template, parallel, True, run_id
        )
        single_total_time = time.time() - single_start_time
        return single_results, single_total_time
        
    except Exception as e:
        # Single QA段階でのエラー
        exec_manager.update_metadata(run_id, {
            'status': 'single_qa_failed',
            'error': f"Single QA error: {str(e)}"
        })
        raise RuntimeError(f"Single QA実行エラー: {e}") from e


def _execute_aggregate_phase(question: str, single_results: List[Dict[str, Any]],
                           aggregate_template: str, run_id: str = None) -> tuple[str, Dict[str, Any], float]:
    """
    Aggregate フェーズの実行
    
    Returns:
        tuple: (aggregate_answer, aggregate_metadata, aggregate_time)
    """
    print("Aggregate処理開始...", file=sys.stderr)
    aggregate_start_time = time.time()
    
    try:
        aggregate_prompt = create_aggregate_prompt(question, single_results, aggregate_template)
        
        # デバッグ: document_answersの部分のみを出力
        if run_id:
            from execution_manager import ExecutionManager
            exec_manager = ExecutionManager()
            run_dir = exec_manager.get_run_dir(run_id)
            debug_file = run_dir / "aggregate_document_answers.txt"
        else:
            debug_file = Path("aggregate_document_answers.txt")
        
        # aggregate_promptから document_answers部分のみを抽出
        start_marker = "=== DOCUMENT ANSWERS START ==="
        end_marker = "=== DOCUMENT ANSWERS END ==="
        start_idx = aggregate_prompt.find(start_marker)
        end_idx = aggregate_prompt.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            document_answers_only = aggregate_prompt[start_idx:end_idx + len(end_marker)]
            debug_file.write_text(document_answers_only, encoding='utf-8')
            print(f"Document answers saved to: {debug_file.absolute()}", file=sys.stderr)
        
        # LLMでaggregate処理
        from single_doc_qa import query_llm
        aggregate_answer, aggregate_metadata = query_llm(aggregate_prompt)
        
        aggregate_time = time.time() - aggregate_start_time
        return aggregate_answer, aggregate_metadata, aggregate_time
        
    except Exception as e:
        raise RuntimeError(f"Aggregate処理エラー: {e}") from e


def _calculate_statistics(single_results: List[Dict[str, Any]]) -> tuple[float, int]:
    """
    統計情報の計算
    
    Returns:
        tuple: (avg_single_time, total_single_tokens)
    """
    single_timings = []
    single_tokens = []
    
    for result in single_results:
        if 'metadata' in result and 'timing' in result['metadata']:
            single_timings.append(result['metadata']['timing']['total_time'])
        if 'metadata' in result and 'total_tokens' in result['metadata']:
            single_tokens.append(result['metadata']['total_tokens'])
    
    avg_single_time = sum(single_timings) / len(single_timings) if single_timings else 0
    total_single_tokens = sum(single_tokens) if single_tokens else 0
    
    return avg_single_time, total_single_tokens


def _finalize_execution(exec_manager: ExecutionManager, run_id: str, question: str,
                       single_template: str, aggregate_template: str, parallel: int,
                       documents: List[Dict[str, Any]], aggregate_answer: str,
                       single_results: List[Dict[str, Any]], aggregate_metadata: Dict[str, Any],
                       single_total_time: float, aggregate_time: float, total_time: float) -> None:
    """
    実行結果の最終化と保存
    """
    try:
        # 統計情報計算
        avg_single_time, total_single_tokens = _calculate_statistics(single_results)
        
        # Aggregate結果の作成
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

実行時間:
- Single QA合計: {single_total_time:.2f}s (平均: {avg_single_time:.2f}s/doc)
- Aggregate処理: {aggregate_time:.2f}s
- 総実行時間: {total_time:.2f}s

トークン使用量:
- Single QA合計: {total_single_tokens:,} tokens
- Aggregate: {aggregate_metadata.get('total_tokens', 0):,} tokens
- 総計: {total_single_tokens + aggregate_metadata.get('total_tokens', 0):,} tokens

処理対象文書:
"""
        
        for doc in documents:
            aggregate_result += f"  {doc['index']:2d}: {doc['subdir']}/{doc['filename']}\n"
        
        # 結果保存
        exec_manager.save_aggregate_result(run_id, aggregate_result)
        
        # メタデータ最終更新
        exec_manager.update_metadata(run_id, {
            'status': 'completed',
            'results': {
                'single_qa_count': len(single_results),
                'total_single_tokens': total_single_tokens,
                'aggregate_tokens': aggregate_metadata.get('total_tokens', 0),
                'total_tokens': total_single_tokens + aggregate_metadata.get('total_tokens', 0),
                'timing': {
                    'single_qa_total_time': single_total_time,
                    'single_qa_avg_time': avg_single_time,
                    'aggregate_time': aggregate_time,
                    'total_time': total_time
                }
            }
        })
        
    except Exception as e:
        # 最終化段階でのエラー
        exec_manager.update_metadata(run_id, {
            'status': 'finalization_failed',
            'error': f"Finalization error: {str(e)}"
        })
        raise RuntimeError(f"実行結果保存エラー: {e}") from e


def run_aggregate_only(run_id: str, aggregate_template: str = "focused") -> str:
    """
    既存のsingle QA結果からaggregate処理のみ実行
    
    Args:
        run_id: 既存の実行ID
        aggregate_template: aggregate用テンプレート
        
    Returns:
        str: 実行ID（同じrun_id）
    """
    exec_manager = ExecutionManager()
    
    # 既存メタデータから質問を取得
    metadata = exec_manager.load_metadata(run_id)
    question = metadata['parameters']['question']
    
    # 既存のsingle QA結果を読み込み
    single_results = exec_manager.load_single_qa_results(run_id)
    
    print(f"既存のsingle QA結果 {len(single_results)} 件を読み込み", file=sys.stderr)
    
    # Aggregate処理のみ実行
    aggregate_answer, aggregate_metadata, aggregate_time = _execute_aggregate_phase(
        question, single_results, aggregate_template, run_id
    )
    
    # 統計情報計算
    avg_single_time, total_single_tokens = _calculate_statistics(single_results)
    
    # Aggregate結果の作成
    aggregate_result = f"""=== MAP-REDUCE質問応答結果 ===

実行ID: {run_id}
実行日時: {datetime.now().isoformat()}

質問: {question}

パラメータ:
- Single Template: {metadata['parameters']['single_template']}
- Aggregate Template: {aggregate_template}  
- 並列数: {metadata['parameters']['parallel']}
- 対象文書数: {len(single_results)}

=== 統合回答 ===

{aggregate_answer}

=== 実行統計 ===

実行時間:
- Single QA合計: {metadata.get('results', {}).get('timing', {}).get('single_qa_total_time', 0):.2f}s (再利用)
- Aggregate処理: {aggregate_time:.2f}s
- 総実行時間: {aggregate_time:.2f}s

トークン使用量:
- Single QA合計: {total_single_tokens:,} tokens (再利用)
- Aggregate: {aggregate_metadata.get('total_tokens', 0):,} tokens
- 総計: {total_single_tokens + aggregate_metadata.get('total_tokens', 0):,} tokens

処理対象文書:
"""
    
    # ドキュメント一覧を追加
    for i, result in enumerate(single_results):
        doc_path = Path(result['document_path'])
        subdir = doc_path.parts[-2] if len(doc_path.parts) >= 2 else "root"
        filename = doc_path.stem
        aggregate_result += f"  {i:2d}: {subdir}/{filename}\n"
    
    # 結果保存
    exec_manager.save_aggregate_result(run_id, aggregate_result)
    
    # メタデータ更新
    exec_manager.update_metadata(run_id, {
        'aggregate_template': aggregate_template,
        'aggregate_time': aggregate_time,
        'aggregate_tokens': aggregate_metadata.get('total_tokens', 0)
    })
    
    print(f"Aggregate-only実行完了 - Run ID: {run_id} (実行時間: {aggregate_time:.2f}s)", file=sys.stderr)
    return run_id


def run_aggregate_qa(question: str, single_template: str = "focused",
                    aggregate_template: str = "focused", parallel: int = 3,
                    subdir_filter: List[str] = None, run_id: str = None) -> str:
    """
    Map-Reduce質問応答の完全実行
    
    Args:
        question: 質問内容
        single_template: single_qa用テンプレート
        aggregate_template: aggregate用テンプレート
        parallel: 並列実行数
        subdir_filter: 対象サブディレクトリフィルタ
        run_id: 実行ID（Noneの場合は自動生成）
        
    Returns:
        str: 実行ID
    """
    start_time = time.time()
    
    # Phase 1: 実行環境セットアップ
    exec_manager, run_id, documents = _setup_execution(
        question, single_template, aggregate_template, parallel, subdir_filter, run_id
    )
    
    try:
        # Phase 2: Single QA実行
        single_results, single_total_time = _execute_single_qa_phase(
            exec_manager, run_id, documents, question, single_template, parallel
        )
        
        # Phase 3: Aggregate実行
        aggregate_answer, aggregate_metadata, aggregate_time = _execute_aggregate_phase(
            question, single_results, aggregate_template, run_id
        )
        
        # Phase 4: 結果の最終化
        total_time = time.time() - start_time
        _finalize_execution(
            exec_manager, run_id, question, single_template, aggregate_template,
            parallel, documents, aggregate_answer, single_results, aggregate_metadata,
            single_total_time, aggregate_time, total_time
        )
        
        print(f"実行完了 - Run ID: {run_id} (総実行時間: {total_time:.2f}s)", file=sys.stderr)
        return run_id
        
    except Exception as e:
        # 全体エラーハンドリング（セットアップ以外のエラー）
        try:
            exec_manager.update_metadata(run_id, {
                'status': 'failed',
                'error': str(e)
            })
        except Exception as meta_error:
            print(f"メタデータ更新エラー: {meta_error}", file=sys.stderr)
        
        print(f"実行失敗 - Run ID: {run_id}: {e}", file=sys.stderr)
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
    
    parser.add_argument("question", nargs='?', help="質問内容")
    parser.add_argument("--single-template", default="focused",
                       help="single_qa用プロンプトテンプレート (default: focused)")
    parser.add_argument("--aggregate-template", default="focused", 
                       help="aggregate用プロンプトテンプレート (default: focused)")
    parser.add_argument("--parallel", type=int, default=3,
                       help="並列実行数 (default: 3)")
    parser.add_argument("--subdir", action="append",
                       help="対象サブディレクトリ（複数指定可）")
    parser.add_argument("--run-id", 
                       help="実行ID（存在する場合はaggregate-only、存在しない場合は新規実行）")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="run-idが指定されている場合、強制的にaggregate-only実行")
    
    args = parser.parse_args()
    
    try:
        exec_manager = ExecutionManager()
        
        if args.run_id:
            # run_id指定時の処理
            if exec_manager.run_exists(args.run_id) or args.aggregate_only:
                # 既存run_idからaggregate-only実行（--aggregate-onlyが指定されている場合は強制）
                print(f"既存run_id {args.run_id} を使用してaggregate-only実行", file=sys.stderr)
                run_id = run_aggregate_only(args.run_id, args.aggregate_template)
            else:
                # 存在しないrun_idで新規実行
                question = args.question
                if not question:
                    try:
                        question = input("質問を入力してください: ").strip()
                        if not question:
                            print("質問が入力されませんでした。", file=sys.stderr)
                            sys.exit(1)
                    except (EOFError, KeyboardInterrupt):
                        print("\n処理をキャンセルしました。", file=sys.stderr)
                        sys.exit(1)
                
                print(f"run_id {args.run_id} で新規実行", file=sys.stderr)
                run_id = run_aggregate_qa(
                    question, args.single_template, args.aggregate_template,
                    args.parallel, args.subdir, args.run_id
                )
        else:
            # run_id未指定時の新規実行
            question = args.question
            if not question:
                try:
                    question = input("質問を入力してください: ").strip()
                    if not question:
                        print("質問が入力されませんでした。", file=sys.stderr)
                        sys.exit(1)
                except (EOFError, KeyboardInterrupt):
                    print("\n処理をキャンセルしました。", file=sys.stderr)
                    sys.exit(1)
            
            # 自動incrementでrun_id生成
            auto_run_id = exec_manager.get_next_run_id()
            print(f"自動生成run_id {auto_run_id} で新規実行", file=sys.stderr)
            run_id = run_aggregate_qa(
                question, args.single_template, args.aggregate_template,
                args.parallel, args.subdir, auto_run_id
            )
        
        # 結果表示
        result_path = exec_manager.get_run_dir(run_id) / "aggregate_result.txt"
        print(result_path.read_text(encoding='utf-8'))
            
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()