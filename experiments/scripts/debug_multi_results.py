#!/usr/bin/env python3
"""
複数点群評価結果のJSONファイルをデバッグするためのスクリプト
"""

import json
import sys
import os

def debug_multi_results(json_file):
    """複数点群評価結果をデバッグ表示"""
    print(f"ファイルを分析中: {json_file}")
    
    if not os.path.exists(json_file):
        print(f"エラー: ファイルが見つかりません: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n=== ファイル構造 ===")
    print(f"トップレベルキー: {list(data.keys())}")
    
    if "prompt" in data:
        print(f"プロンプト: {data['prompt']}")
    
    if "num_objects" in data:
        print(f"オブジェクト数: {data['num_objects']}")
    
    if "results" in data:
        results = data["results"]
        print(f"結果数: {len(results)}")
        
        if len(results) > 0:
            print(f"\n=== 最初のサンプル ===")
            sample = results[0]
            print(f"キー: {list(sample.keys())}")
            
            for key, value in sample.items():
                if key == "model_output":
                    print(f"  {key}: '{value[:100]}...' (長さ: {len(value)})")
                elif key == "object_details":
                    print(f"  {key}: {len(value)}個のオブジェクト")
                    for i, obj in enumerate(value):
                        print(f"    オブジェクト{i+1}: {obj}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n=== モデル出力の例 ===")
        for i in range(min(5, len(results))):
            output = results[i]["model_output"]
            if "object_details" in results[i]:
                true_categories = [obj["category"] for obj in results[i]["object_details"]]
                print(f"サンプル{i+1}: 真値={true_categories}, 出力='{output[:50]}...'")
            else:
                print(f"サンプル{i+1}: 出力='{output[:50]}...'")
        
        # モデル出力の統計
        outputs = [result["model_output"] for result in results]
        lengths = [len(output) for output in outputs]
        
        print(f"\n=== 出力統計 ===")
        print(f"平均長さ: {sum(lengths) / len(lengths):.1f}")
        print(f"最小長さ: {min(lengths)}")
        print(f"最大長さ: {max(lengths)}")
        
        # 出力の始まりの傾向を調査
        first_words = []
        first_chars = []
        for output in outputs[:20]:  # 最初の20個
            words = output.strip().split()
            if words:
                first_words.append(words[0])
            if output.strip():
                first_chars.append(output.strip()[0])
        
        print(f"\n=== 出力パターン (最初の20サンプル) ===")
        print(f"最初の単語: {set(first_words)}")
        print(f"最初の文字: {set(first_chars)}")
        
        # 単一文字出力の検出
        single_char_outputs = [output for output in outputs if len(output.strip()) == 1]
        if single_char_outputs:
            print(f"\n=== 単一文字出力 ===")
            print(f"数: {len(single_char_outputs)}")
            print(f"文字: {set(single_char_outputs)}")
            
        # 短い出力の検出 (10文字以下)
        short_outputs = [output for output in outputs if len(output.strip()) <= 10]
        if short_outputs:
            print(f"\n=== 短い出力 (10文字以下) ===")
            print(f"数: {len(short_outputs)}")
            print(f"例: {short_outputs[:10]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python debug_multi_results.py <json_file>")
        print("例: python debug_multi_results.py /path/to/evaluation_multi/ModelNet_multi_classification_prompt0_obj3_batch30.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    debug_multi_results(json_file) 