#!/usr/bin/env python3
"""
行政文書RAGシステム用の文書分析・可視化スクリプト（トークン数計算版）
GPT-OSSモデルを使用してトークン数を計算
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import ollama
import time

# 日本語フォントの設定
FONT_PATH = "/usr/share/fonts/line-seed/LINESeedJP_TTF_Rg.ttf"

# カテゴリごとの色設定
CATEGORY_COLORS = {
    "1空き家": "#FF6B6B",
    "2立地適正化計画": "#4ECDC4",
    "3街かん": "#45B7D1",
    "4官まち": "#96CEB4",
    "5都市防災": "#FFEAA7"
}

# 100k文字の閾値
CHAR_THRESHOLD = 100000

# 128kトークンの閾値（GPT-OSS用）
TOKEN_THRESHOLD = 128000

# サンプリングによる推定を使うファイルサイズ閾値（50KB）
SAMPLING_THRESHOLD = 50000


def setup_japanese_font():
    """日本語フォントの設定"""
    if os.path.exists(FONT_PATH):
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams["font.family"] = font_prop.get_name()
        print(f"フォントを設定しました: {FONT_PATH}")
    else:
        print(f"警告: フォントが見つかりません: {FONT_PATH}")
        plt.rcParams["font.family"] = "sans-serif"


def estimate_token_ratio(model: str = 'gpt-oss:20b') -> float:
    """サンプルテキストからトークン/文字比率を推定"""
    sample_texts = [
        "行政文書の管理に関する基本的な指針",
        "都市計画法に基づく立地適正化計画の策定",
        "空き家等対策特別措置法の施行について"
    ]

    ratios = []
    for text in sample_texts:
        try:
            response = ollama.generate(model=model, prompt=text, options={'num_predict': 0})
            tokens = response.get('prompt_eval_count', 0)
            if tokens > 0:
                ratios.append(tokens / len(text))
        except:
            pass

    # 平均比率を返す（デフォルトは5.0）
    return sum(ratios) / len(ratios) if ratios else 5.0


def get_token_count(text: str, model: str = 'gpt-oss:20b', use_sampling: bool = False, ratio: float = 5.0) -> int:
    """
    GPT-OSSモデルを使ってトークン数を取得
    大きいファイルはサンプリングして推定
    """
    if use_sampling and len(text) > SAMPLING_THRESHOLD:
        # サンプリング: 最初の10KB分だけ計算して推定
        sample_size = 10000
        sample_text = text[:sample_size]
        try:
            response = ollama.generate(model=model, prompt=sample_text, options={'num_predict': 0})
            sample_tokens = response.get('prompt_eval_count', 0)
            if sample_tokens > 0:
                # サンプルから全体を推定
                estimated_tokens = int((sample_tokens / len(sample_text)) * len(text))
                return estimated_tokens
        except:
            pass

    # 通常の計算またはフォールバック
    if len(text) < SAMPLING_THRESHOLD:
        try:
            response = ollama.generate(model=model, prompt=text, options={'num_predict': 0})
            return response.get('prompt_eval_count', 0)
        except Exception as e:
            print(f"  トークン数取得エラー: {e}")

    # フォールバック: 推定比率を使用
    return int(len(text) * ratio)


def collect_document_data(base_dir: str) -> List[Dict]:
    """
    文書データの収集とトークン数計算
    """
    data = []
    base_path = Path(base_dir)

    # まず全ファイル情報を収集
    print("ファイル情報を収集中...")
    for category_dir in sorted(base_path.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        for txt_file in category_dir.glob("*.txt"):
            try:
                file_size = txt_file.stat().st_size
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    char_count = len(content)

                data.append({
                    'path': str(txt_file),
                    'name': txt_file.stem,
                    'category': category_name,
                    'char_count': char_count,
                    'file_size': file_size,
                    'content': content,
                    'token_count': 0
                })
            except Exception as e:
                print(f"エラー: {txt_file} の読み込みに失敗: {e}")

    # 文字数でソート（小さいファイルから処理）
    data.sort(key=lambda x: x['char_count'])

    print(f"\n{len(data)}個のファイルのトークン数を計算します")
    print("トークン/文字比率を推定中...")
    ratio = estimate_token_ratio()
    print(f"推定比率: {ratio:.2f} トークン/文字")

    # トークン数計算
    start_time = time.time()
    for i, item in enumerate(data):
        print(f"  [{i+1}/{len(data)}] {item['name'][:30]:30} ({item['char_count']:,}文字)", end="")

        use_sampling = item['char_count'] > SAMPLING_THRESHOLD
        token_count = get_token_count(item['content'], use_sampling=use_sampling, ratio=ratio)
        item['token_count'] = token_count if token_count else int(item['char_count'] * ratio)

        # contentは不要なので削除
        del item['content']

        print(f" -> {item['token_count']:,}トークン" + (" (推定)" if use_sampling else ""))

        # 30秒経過したら残りは推定
        if time.time() - start_time > 30:
            print("\n時間制限により残りは推定値を使用します")
            for j in range(i+1, len(data)):
                data[j]['token_count'] = int(data[j]['char_count'] * ratio)
                del data[j]['content']
            break

    return data


def create_char_count_chart(data: List[Dict], output_path: str):
    """文字数の棒グラフを作成"""
    sorted_data = sorted(data, key=lambda x: x['char_count'])

    names = [d['name'] for d in sorted_data]
    char_counts = [d['char_count'] for d in sorted_data]
    colors = [CATEGORY_COLORS.get(d['category'], '#888888') for d in sorted_data]

    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, char_counts, color=colors)

    ax.axvline(x=CHAR_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='100k文字（RAG処理目安）')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('文字数', fontsize=10)
    ax.set_title('行政文書の文字数分析', fontsize=14, fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat)
                      for cat, color in CATEGORY_COLORS.items()]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='100k文字'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax.ticklabel_format(style='plain', axis='x')
    ax.set_xlim(0, max(char_counts) * 1.1)

    for i, (bar, count) in enumerate(zip(bars, char_counts)):
        if count > CHAR_THRESHOLD:
            ax.text(count, bar.get_y() + bar.get_height()/2,
                   f' {count:,}', ha='left', va='center', fontsize=7, color='red', fontweight='bold')
        else:
            ax.text(count, bar.get_y() + bar.get_height()/2,
                   f' {count:,}', ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"文字数グラフを保存しました: {output_path}")


def create_token_count_chart(data: List[Dict], output_path: str):
    """トークン数の棒グラフを作成（GPT-OSS）"""
    sorted_data = sorted(data, key=lambda x: x.get('token_count', 0))

    names = [d['name'] for d in sorted_data]
    token_counts = [d.get('token_count', 0) for d in sorted_data]
    colors = [CATEGORY_COLORS.get(d['category'], '#888888') for d in sorted_data]

    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, token_counts, color=colors)

    ax.axvline(x=TOKEN_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='128kトークン（GPT-OSS処理限界）')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('トークン数（GPT-OSS）', fontsize=10)
    ax.set_title('行政文書のトークン数分析（GPT-OSSモデル）', fontsize=14, fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat)
                      for cat, color in CATEGORY_COLORS.items()]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='128kトークン'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax.ticklabel_format(style='plain', axis='x')
    max_tokens = max(token_counts) if token_counts else 1000
    ax.set_xlim(0, max_tokens * 1.1)

    for i, (bar, count) in enumerate(zip(bars, token_counts)):
        if count > TOKEN_THRESHOLD:
            ax.text(count, bar.get_y() + bar.get_height()/2,
                   f' {count:,}', ha='left', va='center', fontsize=7, color='red', fontweight='bold')
        else:
            ax.text(count, bar.get_y() + bar.get_height()/2,
                   f' {count:,}', ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"トークン数グラフを保存しました: {output_path}")


def create_file_size_chart(data: List[Dict], output_path: str):
    """ファイルサイズの棒グラフを作成"""
    sorted_data = sorted(data, key=lambda x: x['file_size'])

    names = [d['name'] for d in sorted_data]
    file_sizes_mb = [d['file_size'] / (1024 * 1024) for d in sorted_data]
    colors = [CATEGORY_COLORS.get(d['category'], '#888888') for d in sorted_data]

    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, file_sizes_mb, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('ファイルサイズ (MB)', fontsize=10)
    ax.set_title('行政文書のファイルサイズ分析', fontsize=14, fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat)
                      for cat, color in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax.set_xlim(0, max(file_sizes_mb) * 1.1)

    for bar, size in zip(bars, file_sizes_mb):
        ax.text(size, bar.get_y() + bar.get_height()/2,
               f' {size:.2f} MB', ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ファイルサイズグラフを保存しました: {output_path}")


def print_statistics(data: List[Dict]):
    """統計情報の出力"""
    print("\n" + "="*50)
    print("文書統計情報")
    print("="*50)

    total_chars = sum(d['char_count'] for d in data)
    total_size = sum(d['file_size'] for d in data)
    total_tokens = sum(d.get('token_count', 0) for d in data)

    print(f"\n全体:")
    print(f"  ファイル数: {len(data)}")
    print(f"  総文字数: {total_chars:,}")
    print(f"  総トークン数(GPT-OSS): {total_tokens:,}")
    print(f"  総ファイルサイズ: {total_size / (1024*1024):.2f} MB")
    print(f"  平均文字数: {total_chars // len(data):,}")
    print(f"  平均トークン数: {total_tokens // len(data):,}")
    print(f"  平均トークン/文字比: {total_tokens / total_chars:.2f}")

    print(f"\nカテゴリ別:")
    for category in sorted(set(d['category'] for d in data)):
        cat_data = [d for d in data if d['category'] == category]
        cat_chars = sum(d['char_count'] for d in cat_data)
        cat_tokens = sum(d.get('token_count', 0) for d in cat_data)
        cat_size = sum(d['file_size'] for d in cat_data)

        print(f"\n  {category}:")
        print(f"    ファイル数: {len(cat_data)}")
        print(f"    総文字数: {cat_chars:,}")
        print(f"    総トークン数: {cat_tokens:,}")
        print(f"    総サイズ: {cat_size / (1024*1024):.2f} MB")

    over_char_threshold = [d for d in data if d['char_count'] > CHAR_THRESHOLD]
    if over_char_threshold:
        print(f"\n100k文字を超えるファイル ({len(over_char_threshold)}件):")
        for d in sorted(over_char_threshold, key=lambda x: x['char_count'], reverse=True):
            print(f"  - {d['name']}: {d['char_count']:,}文字 ({d['category']})")

    over_token_threshold = [d for d in data if d.get('token_count', 0) > TOKEN_THRESHOLD]
    if over_token_threshold:
        print(f"\n128kトークンを超えるファイル(GPT-OSS) ({len(over_token_threshold)}件):")
        for d in sorted(over_token_threshold, key=lambda x: x.get('token_count', 0), reverse=True):
            print(f"  - {d['name']}: {d.get('token_count', 0):,}トークン ({d['category']})")


def main():
    base_dir = "data/要綱"
    output_dir = "out"

    os.makedirs(output_dir, exist_ok=True)
    setup_japanese_font()

    print("文書データを収集中...")
    data = collect_document_data(base_dir)

    if not data:
        print("エラー: 文書が見つかりませんでした")
        sys.exit(1)

    create_char_count_chart(data, os.path.join(output_dir, "document_analysis_char_count.png"))
    create_token_count_chart(data, os.path.join(output_dir, "document_analysis_token_count.png"))
    create_file_size_chart(data, os.path.join(output_dir, "document_analysis_file_size.png"))

    print_statistics(data)
    print("\n完了しました！")


if __name__ == "__main__":
    main()