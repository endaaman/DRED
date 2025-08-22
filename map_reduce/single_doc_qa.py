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

import sys
import argparse
import json
from pathlib import Path


def read_document(doc_path: str) -> str:
    """
    ドキュメントファイルを読み込む
    
    Args:
        doc_path: ドキュメントファイルのパス
        
    Returns:
        str: ドキュメントの内容
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        UnicodeDecodeError: ファイルの読み込みに失敗した場合
    """
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")
    
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # UTF-8で読み込めない場合はShift_JISを試す
        try:
            return path.read_text(encoding='shift_jis')
        except UnicodeDecodeError:
            raise UnicodeDecodeError(f"Could not decode file: {doc_path}")


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


def create_prompt(document: str, question: str, template_name: str = "baseline") -> str:
    """
    テンプレートを使用して質問応答用のプロンプトを作成
    
    Args:
        document: ドキュメント内容
        question: 質問内容
        template_name: 使用するテンプレート名
        
    Returns:
        str: LLMに送信するプロンプト
    """
    template = load_prompt_template(template_name)
    return template.format(document=document, question=question)


def query_llm(prompt: str, model: str = "mistral-nemo-jp") -> tuple[str, dict]:
    """
    Ollama公式ライブラリを使ってLLMに質問を投げて回答を取得
    
    Args:
        prompt: LLMに送信するプロンプト
        model: 使用するOllamaモデル名
        
    Returns:
        tuple[str, dict]: (回答テキスト, メタデータ辞書)
        
    Raises:
        Exception: Ollamaとの通信でエラーが発生した場合
    """
    import ollama
    
    try:
        # 並列実行時のログ混雑を避けるため、条件付きでログ出力
        if not globals().get('_SILENT_MODE', False):
            print(f"LLMクエリ開始 (モデル: {model})", file=sys.stderr)
        
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.1,  # 一貫性のある回答のため低めに設定
                "top_p": 0.9
            }
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
                print(f"  プロンプト: {prompt_tokens:,} tokens", file=sys.stderr)
                print(f"  回答生成: {completion_tokens:,} tokens", file=sys.stderr)
                print(f"  合計: {total_tokens:,} tokens", file=sys.stderr)
            
            # モデルのコンテキスト長から残りトークンを推定
            context_lengths = {
                "mistral-nemo-jp": 128000,  # Mistral Nemo は128k context
                "llama-elyzsa-jp-8b": 8192,
                "hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF": 8192,
            }
            
            if model in context_lengths:
                max_tokens = context_lengths[model]
                remaining_tokens = max_tokens - prompt_tokens
                metadata["remaining_tokens"] = remaining_tokens
                metadata["context_usage_percent"] = (prompt_tokens / max_tokens) * 100
                if not globals().get('_SILENT_MODE', False):
                    print(f"  残りコンテキスト: {remaining_tokens:,} tokens ({remaining_tokens/max_tokens*100:.1f}%)", file=sys.stderr)
        
        if "response" in response:
            return response["response"].strip(), metadata
        else:
            raise Exception(f"予期しないレスポンス形式: {response}")
            
    except Exception as e:
        if "connection" in str(e).lower():
            raise Exception("Ollamaサーバーに接続できません。Ollamaが起動していることを確認してください。")
        else:
            raise Exception(f"LLMクエリ中にエラーが発生しました: {e}")


def single_document_qa(doc_path: str, question: str, template_name: str = "baseline") -> dict:
    """
    単一ドキュメントに対する質問応答を実行
    
    Args:
        doc_path: ドキュメントファイルのパス
        question: 質問内容
        template_name: 使用するプロンプトテンプレート名
        
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
    try:
        # ドキュメント読み込み
        document = read_document(doc_path)
        if not globals().get('_SILENT_MODE', False):
            print(f"ドキュメント読み込み完了: {len(document)} 文字", file=sys.stderr)
        
        # プロンプト作成
        prompt = create_prompt(document, question, template_name)
        if not globals().get('_SILENT_MODE', False):
            print(f"プロンプト作成完了: {len(prompt)} 文字 (テンプレート: {template_name})", file=sys.stderr)
        
        # LLMクエリ実行
        answer, llm_metadata = query_llm(prompt)
        
        # 結果を辞書として構築
        result = {
            "document_path": str(doc_path),
            "question": question,
            "template": template_name,
            "answer": answer,
            "metadata": {
                "document_length": len(document),
                "prompt_length": len(prompt),
                **llm_metadata
            }
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"処理中にエラーが発生しました: {e}")


def main():
    """コマンドライン実行時のメイン処理"""
    parser = argparse.ArgumentParser(
        description="単一ドキュメントに対する質問応答システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python single_doc_qa.py data/要綱/空き家/空き家ガイドライン●.txt "管理不全空家等の定義は何ですか？"
        """
    )
    
    parser.add_argument("document", nargs='?', help="質問対象のドキュメントファイルパス")
    parser.add_argument("question", nargs='?', help="質問内容")
    parser.add_argument("-t", "--template", default="baseline", 
                       help="使用するプロンプトテンプレート (default: baseline)")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細な出力を表示")
    parser.add_argument("--list-templates", action="store_true", 
                       help="利用可能なテンプレート一覧を表示")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="出力形式 (default: text)")
    
    args = parser.parse_args()
    
    # テンプレート一覧表示
    if args.list_templates:
        template_dir = Path(__file__).parent / "prompts" / "single_qa"
        templates = [t.stem for t in template_dir.glob("*.txt")]
        print("利用可能なプロンプトテンプレート:")
        for template in sorted(templates):
            print(f"  - {template}")
        return
    
    # 必須引数の確認
    if not args.document or not args.question:
        parser.error("document and question are required unless --list-templates is used")
    
    try:
        # 質問応答実行
        result = single_document_qa(args.document, args.question, args.template)
        
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
                print(f"  ドキュメント長: {metadata.get('document_length', 'N/A'):,} 文字")
                print(f"  プロンプト長: {metadata.get('prompt_length', 'N/A'):,} 文字")
                if 'total_tokens' in metadata:
                    print(f"  使用トークン: {metadata['total_tokens']:,} tokens")
                    print(f"  残りコンテキスト: {metadata.get('remaining_tokens', 'N/A'):,} tokens")
        
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()