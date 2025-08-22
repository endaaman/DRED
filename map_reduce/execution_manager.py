#!/usr/bin/env python3
"""
実行管理システム

機能:
- 実行ID生成とディレクトリ管理
- 実行メタデータの保存・読み込み
- 結果ファイルの統一的管理
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil


class ExecutionManager:
    """実行管理クラス"""
    
    def __init__(self, base_dir: str = "run"):
        """
        Args:
            base_dir: 実行結果を保存するベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_run(self, run_id: Optional[str] = None) -> str:
        """
        新しい実行IDを生成してディレクトリを作成
        
        Args:
            run_id: 指定する実行ID。Noneの場合は自動生成
            
        Returns:
            str: 作成された実行ID
        """
        if run_id is None:
            run_id = self._generate_run_id()
        
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # single_qa用ディレクトリも作成
        (run_dir / "single_qa").mkdir(exist_ok=True)
        
        # メタデータファイル初期化
        metadata = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "parameters": {},
            "documents": [],
            "results": {}
        }
        self.save_metadata(run_id, metadata)
        
        return run_id
    
    def _generate_run_id(self) -> str:
        """日付ベースの実行ID生成 (YYYY-MM-DD_NNNN)"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 同日の実行ID数をカウント
        existing_runs = [
            run for run in self.list_runs() 
            if run.startswith(today)
        ]
        
        # 次の順序番号を決定
        next_num = len(existing_runs) + 1
        run_id = f"{today}_{next_num:04d}"
        
        # 念のため重複チェック
        while (self.base_dir / run_id).exists():
            next_num += 1
            run_id = f"{today}_{next_num:04d}"
            
        return run_id
    
    def get_run_dir(self, run_id: str) -> Path:
        """実行IDに対応するディレクトリパスを取得"""
        return self.base_dir / run_id
    
    def get_single_qa_dir(self, run_id: str) -> Path:
        """single_qa結果ディレクトリパスを取得"""
        return self.get_run_dir(run_id) / "single_qa"
    
    def list_runs(self) -> List[str]:
        """既存の実行ID一覧を取得"""
        runs = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                runs.append(path.name)
        return sorted(runs)
    
    def save_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """メタデータをファイルに保存"""
        metadata_path = self.get_run_dir(run_id) / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_metadata(self, run_id: str) -> Dict[str, Any]:
        """メタデータをファイルから読み込み"""
        metadata_path = self.get_run_dir(run_id) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for run_id: {run_id}")
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_metadata(self, run_id: str, updates: Dict[str, Any]) -> None:
        """メタデータを部分更新"""
        metadata = self.load_metadata(run_id)
        metadata.update(updates)
        metadata["updated_at"] = datetime.now().isoformat()
        self.save_metadata(run_id, metadata)
    
    def generate_single_qa_filename(self, doc_index: int, doc_path: Path) -> str:
        """
        single_qa結果ファイル名を生成
        
        Args:
            doc_index: ドキュメントのインデックス番号
            doc_path: ドキュメントファイルのパス
            
        Returns:
            str: 生成されたファイル名
        """
        # パスを解析してサブディレクトリとファイル名を取得
        if len(doc_path.parts) >= 2 and doc_path.parts[-2] != "data":
            subdir = doc_path.parts[-2]  # サブディレクトリ名
        else:
            subdir = "misc"
        
        # ファイル名から拡張子を除去
        filename = doc_path.stem
        
        return f"{doc_index:03d}_{subdir}_{filename}.json"
    
    def save_single_qa_result(self, run_id: str, doc_index: int, doc_path: Path, 
                            result: Dict[str, Any]) -> Dict[str, str]:
        """
        single_qa結果をJSONとTXT両方で保存
        
        Args:
            run_id: 実行ID
            doc_index: ドキュメントインデックス
            doc_path: ドキュメントパス
            result: 結果データ
            
        Returns:
            Dict[str, str]: 保存されたファイルパス（json, txt）
        """
        base_filename = self.generate_single_qa_filename(doc_index, doc_path)
        base_name = base_filename.replace('.json', '')
        
        json_path = self.get_single_qa_dir(run_id) / f"{base_name}.json"
        txt_path = self.get_single_qa_dir(run_id) / f"{base_name}.txt"
        
        # JSON保存（機械読み取り用）
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # TXT保存（人間読み取り用）
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Single QA結果 ===\n\n")
            f.write(f"ドキュメント: {result['document_path']}\n")
            f.write(f"質問: {result['question']}\n")
            f.write(f"テンプレート: {result['template']}\n\n")
            f.write(f"=== 回答 ===\n\n")
            f.write(f"{result['answer']}\n\n")
            
            if 'metadata' in result:
                metadata = result['metadata']
                f.write(f"=== 実行情報 ===\n\n")
                f.write(f"ドキュメント長: {metadata.get('document_length', 'N/A'):,} 文字\n")
                f.write(f"プロンプト長: {metadata.get('prompt_length', 'N/A'):,} 文字\n")
                
                if 'total_tokens' in metadata:
                    f.write(f"使用トークン: {metadata['total_tokens']:,} tokens\n")
                    f.write(f"残りコンテキスト: {metadata.get('remaining_tokens', 'N/A'):,} tokens\n")
                
                if 'timing' in metadata:
                    timing = metadata['timing']
                    f.write(f"実行時間:\n")
                    f.write(f"  ドキュメント読み込み: {timing['document_load_time']:.2f}s\n")
                    f.write(f"  プロンプト作成: {timing['prompt_creation_time']:.2f}s\n")
                    f.write(f"  LLM処理: {timing['llm_query_time']:.2f}s\n")
                    f.write(f"  総実行時間: {timing['total_time']:.2f}s\n")
            
        return {
            'json': str(json_path),
            'txt': str(txt_path)
        }
    
    def load_single_qa_results(self, run_id: str) -> List[Dict[str, Any]]:
        """single_qa結果を全て読み込み"""
        single_qa_dir = self.get_single_qa_dir(run_id)
        results = []
        
        for result_file in sorted(single_qa_dir.glob("*.json")):
            with open(result_file, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
                
        return results
    
    def save_aggregate_result(self, run_id: str, result: str) -> str:
        """
        aggregate結果を保存
        
        Args:
            run_id: 実行ID
            result: 統合結果テキスト
            
        Returns:
            str: 保存されたファイルパス
        """
        output_path = self.get_run_dir(run_id) / "aggregate_result.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
            
        return str(output_path)
    
    def cleanup_run(self, run_id: str) -> None:
        """実行結果を削除"""
        run_dir = self.get_run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """実行結果のサマリーを取得"""
        metadata = self.load_metadata(run_id)
        run_dir = self.get_run_dir(run_id)
        
        # single_qa結果数をカウント
        single_qa_count = len(list(self.get_single_qa_dir(run_id).glob("*.json")))
        
        # aggregate結果の存在確認
        has_aggregate = (run_dir / "aggregate_result.txt").exists()
        
        return {
            "run_id": run_id,
            "created_at": metadata.get("created_at"),
            "status": metadata.get("status"),
            "single_qa_count": single_qa_count,
            "has_aggregate": has_aggregate,
            "parameters": metadata.get("parameters", {})
        }


def main():
    """コマンドライン実行時のテスト処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="実行管理システムのテスト")
    parser.add_argument("--list", action="store_true", help="実行一覧を表示")
    parser.add_argument("--create", action="store_true", help="新しい実行を作成")
    parser.add_argument("--summary", help="指定された実行のサマリーを表示")
    
    args = parser.parse_args()
    
    manager = ExecutionManager()
    
    if args.list:
        runs = manager.list_runs()
        print("実行一覧:")
        for run_id in runs:
            summary = manager.get_run_summary(run_id)
            print(f"  {run_id}: {summary['status']} ({summary['single_qa_count']} docs)")
    
    elif args.create:
        run_id = manager.create_run()
        print(f"新しい実行を作成しました: {run_id}")
    
    elif args.summary:
        summary = manager.get_run_summary(args.summary)
        print(f"実行サマリー: {args.summary}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()