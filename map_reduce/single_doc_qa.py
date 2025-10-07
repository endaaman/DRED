#!/usr/bin/env python3
"""
単発ドキュメント質問応答システム

使用方法:
    python single_doc_qa.py <document_path> <question>

機能:
    - 指定されたドキュメント全体をLLMに渡して質問に回答
    - 出典情報を含む回答を生成
    - コマンドライン実行とモジュール使用の両方をサポート
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import ollama
import chardet


def read_document(doc_path: str) -> str:
    """
    ドキュメントファイルを読み込む（エンコーディング自動検出）

    Args:
        doc_path: ドキュメントファイルのパス

    Returns:
        str: ドキュメントの内容

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    # ファイル内容をバイナリで読み込んでエンコーディングを検出
    raw_data = path.read_bytes()
    detected = chardet.detect(raw_data)
    encoding = detected['encoding']

    # 検出されたエンコーディングで読み込み
    return raw_data.decode(encoding)


def load_prompt_template(template_name: str) -> str:
    """
    プロンプトテンプレートファイルを読み込む

    Args:
        template_name: テンプレート名 (例: "baseline", "structured")

    Returns:
        str: プロンプトテンプレート

    Raises:
        FileNotFoundError: テンプレートファイルが見つからない場合
    """
    template_path = Path(__file__).parent / "prompts" / "single_qa" / f"{template_name}.txt"

    if not template_path.exists():
        available_templates = list((Path(__file__).parent / "prompts" / "single_qa").glob("*.txt"))
        available_names = [t.stem for t in available_templates]
        raise FileNotFoundError(
            f"Template '{template_name}' not found. "
            f"Available templates: {', '.join(available_names)}"
        )

    return template_path.read_text(encoding='utf-8')


def create_prompt(document: str, question: str, document_path: str,
                  template_name: str = "baseline",
                  conversation_history: List[Dict[str, str]] = None) -> str:
    """
    テンプレートを使用して質問応答用のプロンプトを作成

    Args:
        document: ドキュメント内容
        question: 質問内容
        template_name: 使用するテンプレート名
        conversation_history: 対話履歴のリスト

    Returns:
        str: LLMに送信するプロンプト
    """
    template = load_prompt_template(template_name)

    p = Path(document_path)

    params = {
        'document':document,
        'category': p.parent.stem,
        'document_name': p.stem,
        'question': question,
    }


    # 対話履歴がある場合は追加
    if conversation_history:
        history_text = "\n\n## 過去の質問と回答\n"
        for i, exchange in enumerate(conversation_history, 1):
            history_text += f"\n**質問{i}**: {exchange['question']}\n"
            history_text += f"**回答{i}**: {exchange['answer']}\n"

        # テンプレートに履歴を挿入
        base_prompt = template.format(**params)
        # ドキュメント部分の後に履歴を挿入
        if "---" in base_prompt:
            parts = base_prompt.split("---", 1)
            return parts[0] + history_text + "\n\n---" + parts[1]
        else:
            return base_prompt + history_text

    return template.format(**params)


def get_model_context_length(model: str) -> int:
    """
    Ollamaからモデル情報を取得してコンテキスト長を返す

    Args:
        model: Ollamaモデル名

    Returns:
        int: コンテキスト長（取得失敗時はNone）
    """
    try:
        model_info = ollama.show(model)
        # モデル情報からnum_ctxを取得
        if 'modelfile' in model_info:
            # モデルファイルからPARAMETER num_ctxを探す
            lines = model_info['modelfile'].split('\n')
            for line in lines:
                if line.strip().startswith('PARAMETER num_ctx'):
                    parts = line.split()
                    if len(parts) >= 3:
                        return int(parts[2])

        # modelinfoから直接取得を試みる
        if 'details' in model_info:
            if 'parameter_size' in model_info['details']:
                # パラメータサイズから推定（これは正確ではないため、別の方法を優先）
                pass

        # デフォルト値を返す
        return None
    except Exception as e:
        if not globals().get('_SILENT_MODE', False):
            print(f"モデル情報取得エラー: {e}", file=sys.stderr)
        return None


def query_llm(prompt: str, model: str = None, num_ctx: int = None, num_predict: int = None):
    """
    Ollama公式ライブラリを使ってLLMに質問を投げて回答を取得

    Args:
        prompt: LLMに送信するプロンプト
        model: 使用するOllamaモデル名
        num_ctx: コンテキスト長（手動指定する場合）
        num_predict: 最大生成トークン数（手動指定する場合）

    Returns:
        tuple[str, dict]: (回答テキスト, メタデータ辞書)

    Raises:
        Exception: Ollamaとの通信でエラーが発生した場合
    """

    # 環境変数からモデル名を取得、デフォルトはGPT-OSS 20B
    if model is None:
        model = os.environ.get('OLLAMA_MODEL', 'gpt-oss:20b')

    # 環境変数からnum_ctxを取得（引数で指定されていない場合）
    if num_ctx is None:
        env_num_ctx = os.environ.get('OLLAMA_NUM_CTX')
        if env_num_ctx:
            try:
                num_ctx = int(env_num_ctx)
                if not globals().get('_SILENT_MODE', False):
                    print(f"環境変数OLLAMA_NUM_CTXから設定: {num_ctx} tokens", file=sys.stderr)
            except ValueError:
                if not globals().get('_SILENT_MODE', False):
                    print(f"警告: OLLAMA_NUM_CTXの値が無効です: {env_num_ctx}", file=sys.stderr)
        else:
            # デフォルト: 128k
            num_ctx = 131072

    # 環境変数からnum_predictを取得（引数で指定されていない場合）、デフォルトは4096
    if num_predict is None:
        env_num_predict = os.environ.get('OLLAMA_NUM_PREDICT')
        if env_num_predict:
            try:
                num_predict = int(env_num_predict)
                if not globals().get('_SILENT_MODE', False):
                    print(f"環境変数OLLAMA_NUM_PREDICTから設定: {num_predict} tokens", file=sys.stderr)
            except ValueError:
                if not globals().get('_SILENT_MODE', False):
                    print(f"警告: OLLAMA_NUM_PREDICTの値が無効です: {env_num_predict}", file=sys.stderr)
                num_predict = 4096
        else:
            num_predict = 4096

    try:
        # 並列実行時のログ混雑を避けるため、条件付きでログ出力
        if not globals().get('_SILENT_MODE', False):
            print(f"LLMクエリ開始 (モデル: {model})", file=sys.stderr)

        # コンテキスト長の決定（優先順位: 引数 > 環境変数 > モデル情報から自動取得）
        if num_ctx:  # 0でも有効な値として扱う
            context_length = num_ctx
            if not globals().get('_SILENT_MODE', False):
                print(f"コンテキスト長を手動設定: {context_length} tokens", file=sys.stderr)
        else:
            # モデルのコンテキスト長を自動取得
            context_length = get_model_context_length(model)
            if context_length and not globals().get('_SILENT_MODE', False):
                print(f"コンテキスト長を自動取得: {context_length} tokens", file=sys.stderr)

        # generateオプションを準備
        options = {
            "temperature": 0.4,        # 反復ループ防止のため適度なランダム性を確保
            "top_p": 0.9,
            "repeat_penalty": 1.1,     # 反復抑制: 同じ語句の繰り返しにペナルティ
            "frequency_penalty": 0.3,  # 頻出語抑制: 頻繁に使われる語句にペナルティ
            "num_predict": num_predict,  # 最大生成トークン数（デフォルト4096）
        }

        if not globals().get('_SILENT_MODE', False):
            print(f"最大生成トークン数: {num_predict} tokens", file=sys.stderr)

        # コンテキスト長が取得できた場合はnum_ctxを設定
        if context_length:
            options["num_ctx"] = context_length

        response = ollama.generate(
            model=model,
            prompt=prompt,
            options=options
        )

        # メタデータを構築
        metadata = {
            "model": model,
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_duration": response.get("prompt_eval_duration"),
            "eval_duration": response.get("eval_duration"),
        }

        # トークン使用量を記録・表示
        if "prompt_eval_count" in response and "eval_count" in response:
            prompt_tokens = response["prompt_eval_count"]
            completion_tokens = response["eval_count"]
            total_tokens = prompt_tokens + completion_tokens

            metadata.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            })

            if not globals().get('_SILENT_MODE', False):
                print(f"トークン使用量:", file=sys.stderr)
                print(f"  プロンプト: {prompt_tokens} tokens", file=sys.stderr)
                print(f"  回答生成: {completion_tokens} tokens", file=sys.stderr)
                print(f"  合計: {total_tokens} tokens", file=sys.stderr)

            # コンテキスト長が指定されている場合は残りトークンを推定
            if context_length:
                max_tokens = context_length
                remaining_tokens = max_tokens - prompt_tokens
                metadata["remaining_tokens"] = remaining_tokens
                metadata["context_usage_percent"] = (prompt_tokens / max_tokens) * 100
                if not globals().get('_SILENT_MODE', False):
                    print(f"  残りコンテキスト: {remaining_tokens} tokens ({remaining_tokens/max_tokens*100:.1f}%)", file=sys.stderr)

        if "response" in response:
            return response["response"].strip(), metadata
        else:
            raise Exception(f"予期しないレスポンス形式: {response}")

    except Exception as e:
        if "connection" in str(e).lower():
            raise Exception("Ollamaサーバーに接続できません。Ollamaが起動していることを確認してください。")
        else:
            raise Exception(f"LLMクエリ中にエラーが発生しました: {e}")


def single_document_qa(doc_path: str, question: str, template_name: str = "baseline",
                      conversation_history: List[Dict[str, str]] = None, model: str = None,
                      num_ctx: int = None, num_predict: int = None) -> dict:
    """
    単一ドキュメントに対する質問応答を実行

    Args:
        doc_path: ドキュメントファイルのパス
        question: 質問内容
        template_name: 使用するプロンプトテンプレート名
        conversation_history: 対話履歴
        model: 使用するOllamaモデル名
        num_ctx: コンテキスト長（手動指定する場合）
        num_predict: 最大生成トークン数（手動指定する場合）

    Returns:
        dict: 結果情報を含む辞書
            - document_path: ドキュメントパス
            - question: 質問内容
            - template: 使用テンプレート
            - answer: LLMからの回答
            - metadata: 実行情報（トークン使用量等）

    Raises:
        FileNotFoundError: ドキュメントファイルが見つからない場合
        Exception: その他のエラー
    """
    start_time = time.time()

    # ドキュメント読み込み
    doc_start = time.time()
    document = read_document(doc_path)
    doc_time = time.time() - doc_start

    if not globals().get('_SILENT_MODE', False):
        print(f"ドキュメント読み込み完了: {len(document)} 文字 ({doc_time:.2f}s)", file=sys.stderr)

    # プロンプト作成
    prompt_start = time.time()
    prompt = create_prompt(document, question, doc_path, template_name, conversation_history)
    prompt_time = time.time() - prompt_start

    if not globals().get('_SILENT_MODE', False):
        print(f"プロンプト作成完了: {len(prompt)} 文字 (テンプレート: {template_name}, {prompt_time:.2f}s)", file=sys.stderr)

    # LLMクエリ実行（空の回答の場合は再試行）
    max_retries = 3
    retry_count = 0
    answer = ""
    llm_metadata = {}
    llm_time = 0

    while retry_count < max_retries:
        llm_start = time.time()
        answer, llm_metadata = query_llm(prompt, model, num_ctx, num_predict)
        llm_time = time.time() - llm_start

        # 回答が十分な長さがあればOK
        if answer and len(answer.strip()) >= 10:
            break

        # 再試行
        retry_count += 1
        if retry_count < max_retries:
            if not globals().get('_SILENT_MODE', False):
                print(f"⚠️  警告: 回答が空または短すぎます（{len(answer.strip())}文字）。再試行します ({retry_count}/{max_retries})...", file=sys.stderr)

    # 最後まで空だった場合の警告
    if not answer or len(answer.strip()) < 10:
        if not globals().get('_SILENT_MODE', False):
            print(f"⚠️  警告: {max_retries}回試行しましたが、十分な回答が得られませんでした", file=sys.stderr)

    # 総実行時間計算
    total_time = time.time() - start_time

    # 結果を辞書として構築
    result = {
        "document_path": str(doc_path),
        "question": question,
        "template": template_name,
        "answer": answer,
        "metadata": {
            "document_length": len(document),
            "prompt_length": len(prompt),
            "timing": {
                "document_load_time": doc_time,
                "prompt_creation_time": prompt_time,
                "llm_query_time": llm_time,
                "total_time": total_time
            },
            **llm_metadata
        }
    }

    if not globals().get('_SILENT_MODE', False):
        print(f"処理完了: 総実行時間 {total_time:.2f}s (LLM: {llm_time:.2f}s)", file=sys.stderr)

    return result


def interactive_mode(doc_path: str, template_name: str = "baseline", model: str = None,
                    num_ctx: int = None, num_predict: int = None):
    """
    対話継続モード

    Args:
        doc_path: ドキュメントファイルのパス
        template_name: 使用するプロンプトテンプレート名
        model: 使用するOllamaモデル名
        num_ctx: コンテキスト長（手動指定する場合）
        num_predict: 最大生成トークン数（手動指定する場合）
    """
    print("=" * 60)
    print(f"対話モード開始")
    print(f"ドキュメント: {doc_path}")
    print(f"テンプレート: {template_name}")
    print("=" * 60)
    print("質問を入力してください。終了するには 'quit', 'exit', 'q' を入力してください。")
    print()

    conversation_history = []

    try:
        while True:
            # 質問入力
            try:
                question = input("質問> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n対話を終了します。")
                break

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("対話を終了します。")
                break

            # 質問応答実行
            try:
                result = single_document_qa(doc_path, question, template_name, conversation_history,
                                          model, num_ctx, num_predict)
                answer = result['answer']

                # 回答表示
                print(f"\n回答> {answer}\n")

                # 履歴に追加
                conversation_history.append({
                    'question': question,
                    'answer': answer
                })

                # 履歴が長くなりすぎたら古いものを削除（最新5件まで保持）
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-5:]

            except Exception as e:
                print(f"\nエラーが発生しました: {e}\n")

    except Exception as e:
        print(f"\n予期しないエラー: {e}")


def main():
    """コマンドライン実行時のメイン処理"""
    parser = argparse.ArgumentParser(
        description="単一ドキュメントに対する質問応答システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python single_doc_qa.py data/要綱TEXT/空き家/空き家ガイドライン●.txt "管理不全空家等の定義は何ですか？"
        """
    )

    parser.add_argument("document", nargs='?', help="質問対象のドキュメントファイルパス")
    parser.add_argument("question", nargs='?', help="質問内容")
    parser.add_argument("-t", "--template", default="sandwich",
                       help="使用するプロンプトテンプレート (default: sandwich)")
    parser.add_argument("-m", "--model", default=None,
                       help="使用するOllamaモデル名 (default: 環境変数OLLAMA_MODEL or gpt-oss:20b)")
    parser.add_argument("-c", "--num-ctx", type=int, default=None,
                       help="コンテキスト長を手動指定 (default: 環境変数OLLAMA_NUM_CTX or 131072)")
    parser.add_argument("-p", "--num-predict", type=int, default=None,
                       help="最大生成トークン数を指定 (default: 環境変数OLLAMA_NUM_PREDICT or 4096)")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細な出力を表示")
    parser.add_argument("--list-templates", action="store_true",
                       help="利用可能なテンプレート一覧を表示")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="出力形式 (default: text)")
    parser.add_argument("-i", "--interactive", action="store_true",
                       help="対話継続モード")

    args = parser.parse_args()

    # テンプレート一覧表示
    if args.list_templates:
        template_dir = Path(__file__).parent / "prompts" / "single_qa"
        templates = [t.stem for t in template_dir.glob("*.txt")]
        print("利用可能なプロンプトテンプレート:")
        for template in sorted(templates):
            print(f"  - {template}")
        return

    # 対話モードの場合
    if args.interactive:
        if not args.document:
            parser.error("document is required for interactive mode")
        interactive_mode(args.document, args.template, args.model, args.num_ctx, args.num_predict)
        return

    # 引数の確認と input() による補完
    if not args.document:
        if not args.list_templates:
            parser.error("document is required unless --list-templates is used")

    if not args.question and not args.interactive:
        if args.document:
            # documentはあるがquestionがない場合はinput()で取得
            try:
                args.question = input("質問を入力してください: ").strip()
                if not args.question:
                    print("質問が入力されませんでした。", file=sys.stderr)
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\n処理をキャンセルしました。", file=sys.stderr)
                sys.exit(1)
        else:
            parser.error("question is required unless --interactive mode is used")

    # 質問応答実行
    result = single_document_qa(args.document, args.question, args.template,
                               model=args.model, num_ctx=args.num_ctx, num_predict=args.num_predict)

    # 結果出力
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # テキスト形式出力
        print("=" * 60)
        print(f"ドキュメント: {result['document_path']}")
        print(f"質問: {result['question']}")
        print(f"テンプレート: {result['template']}")
        print("=" * 60)
        print(result['answer'])

        if args.verbose:
            print("\n" + "=" * 60)
            print("実行情報:")
            metadata = result['metadata']
            print(f"  ドキュメント長: {metadata.get('document_length', 'N/A')} 文字")
            print(f"  プロンプト長: {metadata.get('prompt_length', 'N/A')} 文字")
            if 'total_tokens' in metadata:
                print(f"  使用トークン: {metadata['total_tokens']} tokens")
                print(f"  残りコンテキスト: {metadata.get('remaining_tokens', 'N/A')} tokens")
            if 'timing' in metadata:
                timing = metadata['timing']
                print(f"  実行時間:")
                print(f"    ドキュメント読み込み: {timing['document_load_time']:.2f}s")
                print(f"    プロンプト作成: {timing['prompt_creation_time']:.2f}s")
                print(f"    LLM処理: {timing['llm_query_time']:.2f}s")
                print(f"    総実行時間: {timing['total_time']:.2f}s")



if __name__ == "__main__":
    main()
