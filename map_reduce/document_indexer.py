#!/usr/bin/env python3
"""
ドキュメントインデックス作成システム

機能:
- data/ディレクトリ以下のテキストファイル自動検出
- ファイル名の正規化とインデックス付与
- メタデータ付きドキュメントリスト生成
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib


class DocumentIndexer:
    """ドキュメントインデックス作成クラス"""
    
    def __init__(self, base_dir: str = "data/要綱"):
        """
        Args:
            base_dir: ドキュメントベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    def scan_documents(self, pattern: str = "*.txt") -> List[Dict[str, Any]]:
        """
        ドキュメントファイルをスキャンしてインデックス作成
        
        Args:
            pattern: ファイル検索パターン
            
        Returns:
            List[Dict]: ドキュメント情報のリスト
                - index: インデックス番号 (0-based)
                - path: ファイルの絶対パス
                - relative_path: base_dirからの相対パス
                - filename: ファイル名（拡張子なし）
                - subdir: サブディレクトリ名
                - size: ファイルサイズ（バイト）
                - hash: ファイルの内容ハッシュ（MD5）
        """
        documents = []
        
        # 再帰的にtxtファイルを検索
        for txt_file in self.base_dir.rglob(pattern):
            if txt_file.is_file():
                doc_info = self._create_document_info(len(documents), txt_file)
                documents.append(doc_info)
        
        # パスでソートして一貫した順序を保証
        documents.sort(key=lambda x: x['relative_path'])
        
        # インデックスを再割り当て
        for i, doc in enumerate(documents):
            doc['index'] = i
            
        return documents
    
    def _create_document_info(self, index: int, file_path: Path) -> Dict[str, Any]:
        """
        単一ドキュメントの情報を作成
        
        Args:
            index: インデックス番号
            file_path: ファイルパス
            
        Returns:
            Dict: ドキュメント情報
        """
        relative_path = file_path.relative_to(self.base_dir)
        
        # サブディレクトリの取得
        if len(relative_path.parts) > 1:
            # ファイルの親ディレクトリ名を取得（例: 空き家）
            subdir = relative_path.parts[-2]
        else:
            subdir = "root"
        
        # ファイルサイズ取得
        file_size = file_path.stat().st_size
        
        # ファイル内容のハッシュ計算
        file_hash = self._calculate_file_hash(file_path)
        
        return {
            'index': index,
            'path': str(file_path.absolute()),
            'relative_path': str(relative_path),
            'filename': file_path.stem,
            'subdir': subdir,
            'size': file_size,
            'hash': file_hash
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        ファイル内容のMD5ハッシュを計算
        
        Args:
            file_path: ファイルパス
            
        Returns:
            str: MD5ハッシュ値
        """
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            # ハッシュ計算に失敗した場合はファイルパスベースのハッシュを使用
            hash_md5.update(str(file_path).encode('utf-8'))
            
        return hash_md5.hexdigest()[:8]  # 短縮版
    
    def filter_by_subdir(self, documents: List[Dict[str, Any]], 
                        subdirs: List[str]) -> List[Dict[str, Any]]:
        """
        指定サブディレクトリのドキュメントのみをフィルタリング
        
        Args:
            documents: ドキュメントリスト
            subdirs: 対象サブディレクトリ名のリスト
            
        Returns:
            List[Dict]: フィルタリング後のドキュメントリスト
        """
        filtered = [doc for doc in documents if doc['subdir'] in subdirs]
        
        # インデックスを振り直し
        for i, doc in enumerate(filtered):
            doc['index'] = i
            
        return filtered
    
    def get_document_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ドキュメント統計情報を取得
        
        Args:
            documents: ドキュメントリスト
            
        Returns:
            Dict: 統計情報
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_size': 0,
                'subdirs': [],
                'avg_size': 0
            }
        
        total_size = sum(doc['size'] for doc in documents)
        subdirs = list(set(doc['subdir'] for doc in documents))
        
        return {
            'total_documents': len(documents),
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'subdirs': sorted(subdirs),
            'subdir_counts': {subdir: sum(1 for doc in documents if doc['subdir'] == subdir) 
                            for subdir in subdirs},
            'avg_size': round(total_size / len(documents)),
            'avg_size_kb': round(total_size / len(documents) / 1024, 1)
        }
    
    def print_document_list(self, documents: List[Dict[str, Any]], 
                          show_hash: bool = False) -> None:
        """
        ドキュメントリストを表示
        
        Args:
            documents: ドキュメントリスト
            show_hash: ハッシュ値を表示するか
        """
        print("ドキュメント一覧:")
        print("-" * 80)
        
        for doc in documents:
            size_kb = doc['size'] / 1024
            hash_info = f" [{doc['hash']}]" if show_hash else ""
            
            print(f"{doc['index']:3d}: {doc['subdir']}/{doc['filename']}"
                  f" ({size_kb:.1f}KB){hash_info}")
            print(f"     {doc['relative_path']}")
    
    def save_index(self, documents: List[Dict[str, Any]], 
                  output_path: str = "document_index.json") -> None:
        """
        ドキュメントインデックスをJSONファイルに保存
        
        Args:
            documents: ドキュメントリスト
            output_path: 出力ファイルパス
        """
        import json
        
        index_data = {
            'created_at': __import__('datetime').datetime.now().isoformat(),
            'base_dir': str(self.base_dir),
            'stats': self.get_document_stats(documents),
            'documents': documents
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        print(f"ドキュメントインデックスを保存しました: {output_path}")


def main():
    """コマンドライン実行時のメイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ドキュメントインデックス作成ツール")
    parser.add_argument("--base-dir", default="data", 
                       help="ドキュメントベースディレクトリ (default: data)")
    parser.add_argument("--pattern", default="*.txt", 
                       help="ファイル検索パターン (default: *.txt)")
    parser.add_argument("--subdir", action="append", 
                       help="特定サブディレクトリのみを対象にする")
    parser.add_argument("--output", 
                       help="インデックスをJSONファイルに保存")
    parser.add_argument("--stats", action="store_true", 
                       help="統計情報を表示")
    parser.add_argument("--show-hash", action="store_true", 
                       help="ファイルハッシュを表示")
    
    args = parser.parse_args()
    
    try:
        indexer = DocumentIndexer(args.base_dir)
        documents = indexer.scan_documents(args.pattern)
        
        # サブディレクトリフィルタリング
        if args.subdir:
            documents = indexer.filter_by_subdir(documents, args.subdir)
            print(f"サブディレクトリフィルタ適用: {', '.join(args.subdir)}")
        
        # ドキュメント一覧表示
        indexer.print_document_list(documents, args.show_hash)
        
        # 統計情報表示
        if args.stats:
            stats = indexer.get_document_stats(documents)
            print("\n統計情報:")
            print(f"  総ドキュメント数: {stats['total_documents']}")
            print(f"  総サイズ: {stats['total_size_mb']}MB")
            print(f"  平均サイズ: {stats['avg_size_kb']}KB")
            print(f"  サブディレクトリ: {', '.join(stats['subdirs'])}")
            for subdir, count in stats['subdir_counts'].items():
                print(f"    {subdir}: {count}件")
        
        # インデックス保存
        if args.output:
            indexer.save_index(documents, args.output)
            
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()