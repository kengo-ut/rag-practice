from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter

from clients import LLMClient, RAGClient
from settings import PipeSettings


def parse_args():
    parser = ArgumentParser(description="RAG with LangChain")
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="Use reference documents for answering questions.",
    )
    return parser.parse_args()


# 改行と句点「。」を優先的に使う分割ロジック (日本語向け)
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。"],
    chunk_size=40,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)


def main(with_reference: bool = False):
    pipe_settings = PipeSettings()
    rag_client = RAGClient(pipe_settings)
    llm_client = LLMClient(pipe_settings)

    docs_dir = Path(__file__).parent / "documents"  # ドキュメントフォルダのパス
    all_chunks = {}  # ファイルごとのチャンク格納

    for file_path in docs_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        chunks = text_splitter.split_text(text)
        all_chunks[file_path.name] = chunks  # ファイル名をキーにしてチャンクを格納

    queries = [
        {
            "filename": "boiling_point.txt",
            "query": "水の沸点は?",
        },
        {
            "filename": "tokyo_weather.txt",
            "query": "2025年5月5日の東京の天気は?",
        },
        {
            "filename": "faithful_dog_histories.txt",
            "query": "ハチ公はどこで待ち続けた?",
        },
        {
            "filename": "high_voltage_transmission.txt",
            "query": "電気はなぜ高圧で送るの?",
        },
        {
            "filename": "exhibitions_may_2025.txt",
            "query": "2025年5月、東京都内で開催中の展覧会は?",
        },
    ]

    for item in queries:
        filename = item["filename"]
        query = item["query"]
        chunks = all_chunks[filename]
        references = rag_client.get_contextual_chunks(query, chunks, top_k=3)

        references_formatted = "\n".join([doc for doc, _ in references])

        system_prompt = "質問に回答してください。必ず「日本語で回答」すること。"
        question = f"[質問]\n{query}\n\n[回答]"

        if with_reference:
            system_prompt = "質問に回答してください。必ず「日本語で回答」すること。また、与えられる資料を参考にして回答すること。"
            question = (
                f"[参考資料]\n{references_formatted}\n\n[質問]\n{query}\n\n[回答]"
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # LLMに質問を投げる
        response = llm_client.generate_response_from_messages(
            messages=messages,
        )

        print(f"{question}")
        print(f"{response}")

        # 質問と回答をファイルに保存
        output_file = (
            Path(__file__).parent
            / f"output{datetime.now().strftime('%Y%m%d%H%M%S')}"
            / f"{query}_output.txt"
        )
        if with_reference:
            output_file = (
                Path(__file__).parent
                / f"output{datetime.now().strftime('%Y%m%d%H%M%S')}"
                / f"{query}_with_reference_output.txt"
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"{question}\n")
            f.write(f"{response}\n")
        print(f"出力ファイル: {output_file}\n")


if __name__ == "__main__":
    args = parse_args()
    main(with_reference=args.with_reference)
