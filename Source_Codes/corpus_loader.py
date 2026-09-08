import csv
import json
import os
import pandas as pd


class CorpusLoader:

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".json":
            docs = self._load_json(filepath)
        elif ext == ".csv":
            docs = self._load_csv(filepath, delimiter=",")
        elif ext == ".tsv":
            docs = self._load_csv(filepath, delimiter="\t")
        elif ext == ".txt":
            docs = self._load_txt(filepath)
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. "f"Supported: .json, .csv, .tsv, .txt")

        docs = self._normalize(docs)

        print(f"[CorpusLoader] Loaded {len(docs)} documents from '{filepath}'")
        return docs

    def _load_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            raw_docs = data
        elif isinstance(data, dict):
            for key in ("docs", "documents", "data", "items", "results"):
                if key in data:
                    raw_docs = data[key]
                    break
            else:
                raw_docs = [data]
        else:
            raise ValueError("JSON file must contain a list or an object with a 'docs' key")

        return raw_docs

    def _load_csv(self, filepath, delimiter=","):
        raw_docs = []
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                raw_docs.append(dict(row))
        return raw_docs

    def _load_txt(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.splitlines()
        lines = [l for l in lines if l.strip()]

        if not lines:
            return []

        avg_len = sum(len(l) for l in lines) / len(lines)

        if avg_len < 200:
            raw_docs = [{"text": line.strip()} for line in lines if line.strip()]
        else:
            paragraphs = content.split("\n\n")
            raw_docs = [{"text": p.strip()} for p in paragraphs if p.strip()]

        return raw_docs

    def _normalize(self, raw_docs):
        TEXT_KEYS     = ("text", "body", "content", "abstract", "description", "passage")
        TITLE_KEYS    = ("title", "headline", "name", "subject", "header")
        CATEGORY_KEYS = ("category", "label", "topic", "type", "class", "tag")
        ID_KEYS       = ("id", "index", "doc_id", "docid")

        normalized = []

        for i, raw in enumerate(raw_docs):
            lowered = {k.lower(): v for k, v in raw.items()}

            text = ""
            for k in TEXT_KEYS:
                if k in lowered and lowered[k]:
                    text = str(lowered[k]).strip()
                    break
            if not text:
                text = " ".join(str(v) for v in raw.values() if v)

            title = ""
            for k in TITLE_KEYS:
                if k in lowered and lowered[k]:
                    title = str(lowered[k]).strip()
                    break
            if not title:
                title = text[:60] + ("..." if len(text) > 60 else "")

            category = ""
            for k in CATEGORY_KEYS:
                if k in lowered and lowered[k]:
                    category = str(lowered[k]).strip()
                    break

            doc_id = i + 1
            for k in ID_KEYS:
                if k in lowered and lowered[k]:
                    try:
                        doc_id = int(lowered[k])
                    except (ValueError, TypeError):
                        doc_id = i + 1
                    break

            normalized.append({
                "index"   : doc_id,
                "title"   : title,
                "category": category,
                "text"    : text,
            })

        return normalized

    def load_demo(self):
        demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbc-text.csv")
        demo_docs = self.load(demo_path)
        print(f"[CorpusLoader] Loaded demo corpus: {len(demo_docs)} documents")
        return demo_docs

    def save_as_json(self, docs, output_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        print(f"[CorpusLoader] Saved {len(docs)} documents to: {output_path}")
