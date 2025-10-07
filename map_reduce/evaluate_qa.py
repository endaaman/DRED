#!/usr/bin/env python3
"""
QA評価システム - 単一ドキュメントQAの自動評価

使用方法:
    python evaluate_qa.py --qa-file data/QA/QA.xlsx --template sandwich --output-dir run/qa

機能:
    - QA.xlsxから質問と対象ドキュメントを読み込み
    - 各質問に対して指定ドキュメントでsingle_doc_qaを実行
    - 結果をMarkdown形式で保存
    - --dry-runで実行前の確認が可能
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import pandas as pd

# single_doc_qaをインポート
from single_doc_qa import single_document_qa


def find_document_files(doc_name: str, base_path: Path = Path("data/要綱TEXT")) -> List[Path]:
    """
    ドキュメント名から実際のファイルパスを前方一致で検索

    Args:
        doc_name: ドキュメント名（例: "20-4_都市再生整備計画関連事業ハンドブック"）
        base_path: 検索ベースパス

    Returns:
        List[Path]: 見つかったファイルパスのリスト（ソート済み）
    """
    # 前方一致で検索
    pattern = f"{doc_name}*.txt"
    files = sorted(base_path.rglob(pattern))
    return files


def parse_document_column(doc_value: str) -> List[str]:
    """
    ドキュメント列の値を解析して個別のドキュメント名リストを返す

    Args:
        doc_value: ドキュメント列の値（改行区切りの可能性あり）

    Returns:
        List[str]: ドキュメント名のリスト
    """
    if pd.isna(doc_value):
        return []

    # 改行で分割してトリム、空文字を除外
    doc_names = [d.strip() for d in str(doc_value).split('\n') if d.strip()]
    return doc_names


def create_markdown_output(
    question: str,
    no: int,
    seq: int,
    category: str,
    document_name: str,
    document_path: str,
    answer: str,
    reference_answer: Optional[str] = None,
    result: Optional[Dict] = None,
    status: str = "success",
    error_message: Optional[str] = None
) -> str:
    """
    Markdown形式の出力を生成

    Args:
        question: 質問文
        no: 質問番号
        seq: シーケンス番号
        category: カテゴリ名
        document_name: ドキュメント名
        document_path: ドキュメントパス
        answer: LLMの回答
        result: single_doc_qaの結果辞書
        status: ステータス (success/not_found/error)
        error_message: エラーメッセージ（エラー時）

    Returns:
        str: Markdown形式の文字列
    """
    # Front matterを手動で構築
    fm_lines = [
        f"num: {no}",
        f"seq: {seq}",
        f"category: {category}",
        f"document_name: {document_name}",
        f"document_path: {document_path}",
        f"status: {status}",
    ]

    if result:
        metadata = result.get('metadata', {})
        fm_lines.append(f"model: {metadata.get('model', 'unknown')}")
        fm_lines.append(f"template: {result.get('template', 'unknown')}")

        # トークン情報
        if 'prompt_tokens' in metadata:
            fm_lines.append("tokens:")
            fm_lines.append(f"  prompt: {metadata['prompt_tokens']}")
            fm_lines.append(f"  completion: {metadata['completion_tokens']}")
            fm_lines.append(f"  total: {metadata['total_tokens']}")

        # タイミング情報
        if 'timing' in metadata:
            timing = metadata['timing']
            fm_lines.append("timing:")
            fm_lines.append(f"  total: {round(timing['total_time'], 2)}")
            fm_lines.append(f"  llm: {round(timing['llm_query_time'], 2)}")

    if error_message:
        fm_lines.append(f"error: {error_message}")

    yaml_str = "\n".join(fm_lines)

    # Markdown本文
    md_lines = [
        "---",
        yaml_str.rstrip(),
        "---",
        "",
        "# 質問",
        "",
        question,
        "",
    ]

    # 模範解答を追加
    if reference_answer:
        md_lines.extend([
            "---",
            "",
            "# 模範解答",
            "",
            reference_answer,
            "",
        ])

    md_lines.extend([
        "---",
        "",
        "# 回答",
        "",
    ])

    if status == "success":
        md_lines.append(answer)
    elif status == "not_found":
        md_lines.append(f"**エラー**: ドキュメントファイルが見つかりませんでした。")
        md_lines.append(f"\n検索パターン: `{document_name}*.txt`")
    elif status == "error":
        md_lines.append(f"**エラー**: LLM実行中にエラーが発生しました。")
        if error_message:
            md_lines.append(f"\n```\n{error_message}\n```")

    return "\n".join(md_lines)


def process_single_qa(
    no: int,
    question: str,
    doc_path: Path,
    output_path: Path,
    template: str,
    model: Optional[str],
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    dry_run: bool = False
) -> Tuple[str, Optional[Dict]]:
    """
    単一の質問・ドキュメントペアを処理

    Args:
        no: 質問番号
        question: 質問文
        doc_path: ドキュメントファイルパス
        output_path: 出力先パス
        template: テンプレート名
        model: モデル名
        num_ctx: コンテキスト長
        num_predict: 最大生成トークン数
        dry_run: Dry runモード

    Returns:
        Tuple[str, Optional[Dict]]: (ステータス, 結果辞書)
    """
    if dry_run:
        return "success", None

    # single_doc_qaを実行
    result = single_document_qa(
        doc_path=str(doc_path),
        question=question,
        template_name=template,
        model=model,
        num_ctx=num_ctx,
        num_predict=num_predict
    )
    return "success", result


def dry_run_display(qa_data: List[Dict], base_path: Path, template: str, model: str):
    """
    Dry runモードでの表示

    Args:
        qa_data: QAデータのリスト
        base_path: ベースパス
        template: テンプレート名
        model: モデル名
    """
    print("=" * 60)
    print("QA評価システム - Dry Run Mode")
    print("=" * 60)
    print(f"出力先: run/qa/")
    print(f"テンプレート: {template}")
    print(f"モデル: {model}")
    print(f"\n総質問数: {len(qa_data)}")

    total_doc_names = 0
    total_files = 0
    total_found = 0
    total_not_found = 0

    for qa in qa_data:
        print(f"\n{'=' * 60}")
        print(f"NO.{qa['no']}")
        print("=" * 60)

        # 質問を表示（長い場合は省略）
        question = qa['question']
        if len(question) > 100:
            question_display = question[:100] + "..."
        else:
            question_display = question
        print(f"質問: {question_display}")

        print(f"\nドキュメント指定:")
        for doc_name in qa['doc_names']:
            print(f"  - {doc_name}")

        total_doc_names += len(qa['doc_names'])

        print(f"\n検索結果: {len(qa['doc_names'])}個のドキュメント名 → {len(qa['files'])}個のファイル")
        print()

        if not qa['files']:
            for doc_name in qa['doc_names']:
                total_not_found += 1
                print(f"[{qa['no']:02d}_??] ✗ NOT FOUND")
                print(f"  ドキュメント名: {doc_name}")
                print(f"  検索パターン: data/要綱TEXT/**/{doc_name}*.txt")
                print(f"  出力先: run/qa/未分類/{qa['no']:02d}_01.md (status: not_found)")
                print()
        else:
            for idx, file_info in enumerate(qa['files'], 1):
                total_files += 1
                total_found += 1
                print(f"[{qa['no']:02d}_{idx:02d}] ✓ FOUND")
                print(f"  カテゴリ: {file_info['category']}")
                print(f"  ファイル: {file_info['path']}")
                print(f"  出力先: {file_info['output_path']}")
                print()

    print("=" * 60)
    print("サマリー")
    print("=" * 60)
    print(f"総質問数: {len(qa_data)}")
    print(f"総ドキュメント指定数: {total_doc_names}")
    print(f"総ファイル数: {total_files}")
    print(f"  ✓ 検出成功: {total_found}")
    print(f"  ✗ 未検出: {total_not_found}")
    print()
    print("--dry-run モードのため、LLMは実行されませんでした。")
    print("実行する場合は --dry-run オプションを外してください。")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="QA評価システム - 単一ドキュメントQAの自動評価",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--qa-file", default="data/QA/QA.xlsx",
                       help="QAファイルのパス (default: data/QA/QA.xlsx)")
    parser.add_argument("--template", default="sandwich",
                       help="プロンプトテンプレート名 (default: sandwich)")
    parser.add_argument("--output-dir", default="run/qa",
                       help="出力ディレクトリ (default: run/qa)")
    parser.add_argument("--model", default=None,
                       help="Ollamaモデル名 (default: 環境変数OLLAMA_MODELまたはデフォルト)")
    parser.add_argument("--num-ctx", type=int, default=None,
                       help="コンテキスト長 (default: 環境変数OLLAMA_NUM_CTXまたはモデルから自動取得)")
    parser.add_argument("--num-predict", type=int, default=None,
                       help="最大生成トークン数 (default: 環境変数OLLAMA_NUM_PREDICTまたは4096)")
    parser.add_argument("--sheet", default=None,
                       help="処理対象のシート名 (default: 全シート)")
    parser.add_argument("--dry-run", action="store_true",
                       help="実行せずに対象ファイルと質問を表示")

    args = parser.parse_args()

    # 環境変数からモデル名を取得
    model = args.model or os.environ.get('OLLAMA_MODEL', 'gpt-oss:20b')

    # コンテキスト長と生成トークン数を取得
    num_ctx = args.num_ctx
    if num_ctx is None:
        env_num_ctx = os.environ.get('OLLAMA_NUM_CTX')
        if env_num_ctx:
            num_ctx = int(env_num_ctx)
        else:
            num_ctx = 128000  # デフォルト: 128k

    num_predict = args.num_predict
    if num_predict is None:
        env_num_predict = os.environ.get('OLLAMA_NUM_PREDICT')
        if env_num_predict:
            num_predict = int(env_num_predict)

    # QA.xlsxを読み込み
    print(f"QAファイル読み込み中: {args.qa_file}")

    # シート名を決定
    xls = pd.ExcelFile(args.qa_file)
    if args.sheet:
        if args.sheet not in xls.sheet_names:
            print(f"エラー: シート '{args.sheet}' が見つかりません", file=sys.stderr)
            print(f"利用可能なシート: {', '.join(xls.sheet_names)}", file=sys.stderr)
            sys.exit(1)
        sheet_name = args.sheet
    else:
        # 指定がなければ先頭のシート
        sheet_name = xls.sheet_names[0]

    print(f"処理対象シート: {sheet_name}")

    # シートを読み込み
    df = pd.read_excel(args.qa_file, sheet_name=sheet_name, header=2)
    print(f"質問数: {len(df)}")

    # データを準備
    qa_data = []
    base_path = Path("data/要綱TEXT")

    for idx, row in df.iterrows():
        no = int(row['NO'])
        question = str(row['質問'])
        doc_column = row['ドキュメント']
        reference_answer = str(row['回答']) if pd.notna(row['回答']) else None

        # ドキュメント列を解析
        doc_names = parse_document_column(doc_column)

        # 各ドキュメント名でファイルを検索
        all_files = []
        for doc_name in doc_names:
            files = find_document_files(doc_name, base_path)
            for file_path in files:
                category = file_path.parent.name
                all_files.append({
                    'doc_name': doc_name,
                    'path': file_path,
                    'category': category,
                    'output_path': None  # 後で設定
                })

        # 出力パスを設定
        for seq, file_info in enumerate(all_files, 1):
            category = file_info['category']
            output_path = Path(args.output_dir) / category / f"{no:02d}_{seq:02d}.md"
            file_info['output_path'] = output_path

        qa_data.append({
            'no': no,
            'question': question,
            'reference_answer': reference_answer,
            'doc_names': doc_names,
            'files': all_files
        })

    # Dry runモード
    if args.dry_run:
        dry_run_display(qa_data, base_path, args.template, model)
        return

    # 実際の処理
    print(f"\n{'=' * 60}")
    print("QA評価実行開始")
    print("=" * 60)

    total_processed = 0
    total_success = 0
    total_error = 0

    for qa in qa_data:
        no = qa['no']
        question = qa['question']

        print(f"\n{'=' * 60}")
        print(f"NO.{no}: 処理中...")
        print("=" * 60)

        if not qa['files']:
            # ドキュメントが見つからなかった場合
            print(f"  ⚠️  ドキュメントが見つかりませんでした")

            # 未分類ディレクトリに not_found ステータスで保存
            output_dir = Path(args.output_dir) / "未分類"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{no:02d}_01.md"

            md_content = create_markdown_output(
                question=question,
                no=no,
                seq=1,
                category="未分類",
                document_name=", ".join(qa['doc_names']),
                document_path="N/A",
                answer="",
                reference_answer=qa['reference_answer'],
                status="not_found"
            )

            output_path.write_text(md_content, encoding='utf-8')
            print(f"  出力: {output_path}")
            total_processed += 1
            total_error += 1
            continue

        # 各ファイルを処理
        for seq, file_info in enumerate(qa['files'], 1):
            doc_path = file_info['path']
            output_path = file_info['output_path']
            category = file_info['category']

            print(f"\n  [{no:02d}_{seq:02d}] {doc_path.name}")

            # 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # single_doc_qaを実行
            status, result = process_single_qa(
                no=no,
                question=question,
                doc_path=doc_path,
                output_path=output_path,
                template=args.template,
                model=model,
                num_ctx=num_ctx,
                num_predict=num_predict,
                dry_run=False
            )

            total_processed += 1

            if status == "success":
                total_success += 1
                answer = result['answer']
                print(f"  ✓ 成功")
            else:
                total_error += 1
                answer = ""
                result = None

            # Markdownファイルを生成
            md_content = create_markdown_output(
                question=question,
                no=no,
                seq=seq,
                category=category,
                document_name=doc_path.name,
                document_path=str(doc_path),
                answer=answer,
                reference_answer=qa['reference_answer'],
                result=result,
                status=status,
                error_message=result.get('error') if result else None
            )

            output_path.write_text(md_content, encoding='utf-8')
            print(f"  出力: {output_path}")

    # 最終サマリー
    print(f"\n{'=' * 60}")
    print("処理完了")
    print("=" * 60)
    print(f"総処理数: {total_processed}")
    print(f"  ✓ 成功: {total_success}")
    print(f"  ✗ エラー: {total_error}")


if __name__ == "__main__":
    main()
