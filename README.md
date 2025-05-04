# RAG-Practice

RAGを用いたLLMシステムのプロトタイプです。

## 実行方法

1. 依存関係をインストールします。

```bash
uv sync
```

2. `main.py`を実行します。

```bash
uv run python main.py # RAGなし
uv run python main.py --with-reference # RAGあり
```

## 利用しているAIモデルについて

本プロジェクトでは、以下の大規模言語モデル（LLM）を使用しています：

- **モデル名**: [SakanaAI/TinySwallow-1.5B-Instruct](https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct)
- **ライセンス**:  
  - モデル本体は **Apache License 2.0** に準拠（Qwen由来）  
  - 学習データは **Gemma Terms** に従って利用（[Gemma 利用規約](https://ai.google.dev/gemma/terms)）

このモデルは研究・開発用途向けに提供されており、以下の条件を遵守して利用しています：

- 本プロジェクトは **非商用** かつ **教育目的** の開発です
- モデル出力の精度や内容は保証されません（研究プロトタイプとして利用）
- Gemmaの **Prohibited Use（禁止用途）** に該当する使い方は行っていません

> This model is provided for research and development purposes only. Commercial use is permitted only if you comply with both the Apache 2.0 License and the Gemma Terms, including the Prohibited Use section.

## ライセンス

このプロジェクトのソースコードは [MITライセンス](./LICENSE) のもとで公開されています。  
自由にご利用いただけますが、詳細は LICENSE ファイルをご確認ください。

使用モデルのライセンス情報：

- SakanaAI/TinySwallow-1.5B-Instruct  
  - モデル: Apache 2.0  
  - データ: Google Gemma Terms（[詳細はこちら](https://ai.google.dev/gemma/terms)）
