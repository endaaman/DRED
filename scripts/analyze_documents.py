#!/usr/bin/env python3

"""
行政文書RAGシステム用の文書分析・可視化スクリプト
カテゴリ別にtxtファイルの文字数とファイルサイズを可視化
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import numpy as np
import ollama

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


def setup_japanese_font():
    """日本語フォントの設定"""
    if os.path.exists(FONT_PATH):
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams["font.family"] = font_prop.get_name()
        print(f"フォントを設定しました: {FONT_PATH}")
    else:
        print(f"警告: フォントが見つかりません: {FONT_PATH}")
        plt.rcParams["font.family"] = "sans-serif"


def get_token_count(text: str, model: str = 'gpt-oss:20b') -> int:
    """
    GPT-OSSモデルを使ってトークン数を取得
    """
    try:
        response = ollama.generate(model=model, prompt=text, options={'num_predict': 0})
        return response.get('prompt_eval_count', 0)
    except Exception as e:
        print(f"トークン数取得エラー: {e}")
        # フォールバック: 日本語は約5トークン/文字として推定
        return len(text) * 5


def collect_document_data(base_dir: str, calculate_tokens: bool = True) -> List[Dict]:
    """
    文書データの収集

    Returns:
        List[Dict]: ファイル情報のリスト
            - path: ファイルパス
            - name: ファイル名（拡張子なし）
            - category: カテゴリ名
            - char_count: 文字数
            - file_size: ファイルサイズ（バイト）
            - token_count: トークン数（GPT-OSS）
    """
    data = []
    base_path = Path(base_dir)

    for category_dir in sorted(base_path.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        for txt_file in category_dir.glob("*.txt"):
            try:
                # ファイルサイズ取得
                file_size = txt_file.stat().st_size

                # 文字数カウント
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    char_count = len(content)

                file_info = {
                    'path': str(txt_file),
                    'name': txt_file.stem,
                    'category': category_name,
                    'char_count': char_count,
                    'file_size': file_size
                }

                # トークン数計算（オプション）
                if calculate_tokens:
                    print(f"  トークン数計算中: {txt_file.name}")
                    file_info['token_count'] = get_token_count(content)
                else:
                    file_info['token_count'] = 0

                data.append(file_info)

            except Exception as e:
                print(f"エラー: {txt_file} の読み込みに失敗: {e}")

    return data


def create_char_count_chart(data: List[Dict], output_path: str):
    """文字数の棒グラフを作成"""
    # 文字数でソート（昇順）
    sorted_data = sorted(data, key=lambda x: x['char_count'])

    # データ準備
    names = [d['name'] for d in sorted_data]
    char_counts = [d['char_count'] for d in sorted_data]
    colors = [CATEGORY_COLORS.get(d['category'], '#888888') for d in sorted_data]

    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    # 横棒グラフ
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, char_counts, color=colors)

    # 100k文字の閾値線
    ax.axvline(x=CHAR_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='100k文字（RAG処理目安）')

    # ラベル設定
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('文字数', fontsize=10)
    ax.set_title('行政文書の文字数分析', fontsize=14, fontweight='bold')

    # グリッド
    ax.grid(True, axis='x', alpha=0.3)

    # 凡例の作成
    legend_elements = [Patch(facecolor=color, label=cat)
                      for cat, color in CATEGORY_COLORS.items()]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='100k文字'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # X軸のフォーマット
    ax.ticklabel_format(style='plain', axis='x')
    print('MAX', max(char_counts + [120000]) * 1.1)

    # 値の表示
    for i, (bar, count) in enumerate(zip(bars, char_counts)):
        if count > CHAR_THRESHOLD:
            ax.text(count, bar.get_y() + bar.get_height()/2,
                   f' {count:,}', ha='left', va='center', fontsize=7, color='red', fontweight='bold')
        else:
            ax.text(count, bar.get_y() + bar.get_height()/2,
                   f' {count:,}', ha='left', va='center', fontsize=7)

    plt.tight_layout()
    ax.set_xlim(0, max(char_counts + [100000]) * 1.1)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"文字数グラフを保存しました: {output_path}")


def create_token_count_chart(data: List[Dict], output_path: str):
    """トークン数の棒グラフを作成（GPT-OSS）"""
    # トークン数でソート（昇順）
    sorted_data = sorted(data, key=lambda x: x.get('token_count', 0))

    # データ準備
    names = [d['name'] for d in sorted_data]
    token_counts = [d.get('token_count', 0) for d in sorted_data]
    colors = [CATEGORY_COLORS.get(d['category'], '#888888') for d in sorted_data]

    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    # 横棒グラフ
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, token_counts, color=colors)

    # 128kトークンの閾値線
    ax.axvline(x=TOKEN_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='128kトークン（GPT-OSS処理限界）')

    # ラベル設定
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('トークン数（GPT-OSS）', fontsize=10)
    ax.set_title('行政文書のトークン数分析（GPT-OSSモデル）', fontsize=14, fontweight='bold')

    # グリッド
    ax.grid(True, axis='x', alpha=0.3)

    # 凡例の作成
    legend_elements = [Patch(facecolor=color, label=cat)
                      for cat, color in CATEGORY_COLORS.items()]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='128kトークン'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # X軸のフォーマット
    ax.ticklabel_format(style='plain', axis='x')
    ax.set_xlim(0, max(token_counts) * 1.1 if token_counts else 1000)

    # 値の表示
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
    # ファイルサイズでソート（昇順）
    sorted_data = sorted(data, key=lambda x: x['file_size'])

    # データ準備
    names = [d['name'] for d in sorted_data]
    file_sizes_mb = [d['file_size'] / (1024 * 1024) for d in sorted_data]  # MB単位
    colors = [CATEGORY_COLORS.get(d['category'], '#888888') for d in sorted_data]

    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    # 横棒グラフ
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, file_sizes_mb, color=colors)

    # ラベル設定
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('ファイルサイズ (MB)', fontsize=10)
    ax.set_title('行政文書のファイルサイズ分析', fontsize=14, fontweight='bold')

    # グリッド
    ax.grid(True, axis='x', alpha=0.3)

    # 凡例の作成
    legend_elements = [Patch(facecolor=color, label=cat)
                      for cat, color in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # X軸のフォーマット
    ax.set_xlim(0, max(file_sizes_mb) * 1.1)

    # 値の表示
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

    # 全体統計
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
    print(f"  平均ファイルサイズ: {total_size / len(data) / 1024:.2f} KB")

    # カテゴリ別統計
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

    # 100k文字超過ファイル
    over_char_threshold = [d for d in data if d['char_count'] > CHAR_THRESHOLD]
    if over_char_threshold:
        print(f"\n100k文字を超えるファイル ({len(over_char_threshold)}件):")
        for d in sorted(over_char_threshold, key=lambda x: x['char_count'], reverse=True):
            print(f"  - {d['name']}: {d['char_count']:,}文字 ({d['category']})")

    # 128kトークン超過ファイル
    over_token_threshold = [d for d in data if d.get('token_count', 0) > TOKEN_THRESHOLD]
    if over_token_threshold:
        print(f"\n128kトークンを超えるファイル(GPT-OSS) ({len(over_token_threshold)}件):")
        for d in sorted(over_token_threshold, key=lambda x: x.get('token_count', 0), reverse=True):
            print(f"  - {d['name']}: {d.get('token_count', 0):,}トークン ({d['category']})")


def main():
    # 基本設定
    base_dir = "data/要綱"
    output_dir = "out"

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # フォント設定
    setup_japanese_font()

    # データ収集
    print("文書データを収集中...")
    data = collect_document_data(base_dir, calculate_tokens=False)  # まずトークン計算なしで実行

    if not data:
        print("エラー: 文書が見つかりませんでした")
        sys.exit(1)

    print(f"{len(data)}個のファイルを分析します")

    # グラフ作成
    create_char_count_chart(data, os.path.join(output_dir, "document_analysis_char_count.png"))
    # create_token_count_chart(data, os.path.join(output_dir, "document_analysis_token_count.png"))  # トークン計算は重いため無効化
    create_file_size_chart(data, os.path.join(output_dir, "document_analysis_file_size.png"))

    # 統計情報出力
    print_statistics(data)

    print("\n完了しました！")


if __name__ == "__main__":
    main()
