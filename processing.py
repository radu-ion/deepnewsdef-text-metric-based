import os
import argparse
from typing import List, Union, Optional
import multiprocessing as mp
from pathlib import Path
import time
from conllu.models import TokenList, Token, SentenceList, Metadata
from conllu import parse
import stanza
import stanza.models.common.doc as stanza_doc
from stanza import DownloadMethod
import spacy
import spacy.tokens as spacy_toks
import trankit


spacy.prefer_gpu(gpu_id=0) # type: ignore


def _clean_metadata_text(text: str) -> str:
    return " ".join(text.split())


class BaseProcessor:
    def process(self, text: str) -> SentenceList:
        raise NotImplementedError


class StanzaProcessor(BaseProcessor):
    def __init__(self):
        self.nlp = stanza.Pipeline(
            "ro", processors='tokenize,pos,lemma,depparse',
            download_method=DownloadMethod.REUSE_RESOURCES)

    def process(self, text: str) -> SentenceList:
        doc: stanza_doc.Document = self.nlp(text) # type: ignore
        sentences = []

        for s in doc.sentences:
            sent: stanza_doc.Sentence = s
            tokens = []
            
            for w in sent.words:
                word: stanza_doc.Word = w
                tokens.append(Token({
                    "id": word.id,
                    "form": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": word.head,
                    "deprel": word.deprel,
                    "deps": None,
                    "misc": None
                }))
            # end for w

            sentences.append(
                TokenList(tokens, metadata=Metadata({"text": _clean_metadata_text(sent.text)}))) # type: ignore
        # end for s

        return SentenceList(sentences)


class SpacyProcessor(BaseProcessor):
    def __init__(self):
        self.nlp = spacy.load('ro_core_news_sm')

    def process(self, text: str) -> SentenceList:
        doc: spacy_toks.Doc = self.nlp(text)
        sentences = []

        for s in doc.sents:
            sent: spacy_toks.Span = s
            tokens = []
            
            for i, t in enumerate(sent, start=1):
                token: spacy_toks.Token = t

                if token.dep_ == "ROOT":
                    head = 0
                else:
                    head = token.head.i - sent.start + 1
                # end if

                feats_dict = token.morph.to_dict()
                feats = "|".join(f"{k}={v}" for k, v in sorted(feats_dict.items())) or None
                misc = "SpaceAfter=No" if token.whitespace_ == "" else None

                tokens.append(Token({
                    "id": i,
                    "form": token.text,
                    "lemma": token.lemma_,
                    "upos": token.pos_,
                    "xpos": token.tag_,
                    "feats": feats,
                    "head": head,
                    "deprel": token.dep_,
                    "misc": misc
                }))
            # end for t

            sentences.append(TokenList(tokens, metadata=Metadata(
                {"text": _clean_metadata_text(sent.text)})))
        # end for s

        return SentenceList(sentences)


class TrankitProcessor(BaseProcessor):
    def __init__(self):
        self.nlp = trankit.Pipeline(lang='romanian', gpu=True)

    def process(self, text: str) -> SentenceList:
        doc = self.nlp(text)
        sentences = []

        for sent in doc["sentences"]:
            tokens = []
           
            for tok in sent["tokens"]:
                feats = tok["feats"] if "feats" in tok else None

                tokens.append(Token({
                    "id": tok["id"],
                    "form": tok["text"],
                    "lemma": tok["lemma"],
                    "upos": tok["upos"],
                    "xpos": tok["xpos"],
                    "feats": feats,
                    "head": tok["head"],
                    "deprel": tok["deprel"],
                    "deps": None,
                    "misc": None
                }))
            # end for tok
        
            sentences.append(
                TokenList(tokens, metadata=Metadata({"text": _clean_metadata_text(sent["text"])})))
        # end for sent

        return SentenceList(sentences)


def _serialize_sentence_list(sentences: SentenceList) -> str:
    return "\n\n".join(sentence.serialize().strip() for sentence in sentences) + "\n\n"


class TextToCoNLLU:
    def __init__(self, processor: BaseProcessor):
        self.processor = processor
        self.processor_cls = processor.__class__

    def process(self, file_path: Union[str, Path]) -> SentenceList:
        txt_path = Path(file_path)

        if not txt_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {txt_path}")
        # end if

        if txt_path.suffix != ".txt":
            raise ValueError(f"Expected a .txt file, got: {txt_path}")
        # end if

        conllu_path = txt_path.with_suffix(".conllu")

        # 1. Load cached CoNLL-U if it already exists
        if conllu_path.exists():
            with conllu_path.open("r", encoding="utf-8") as f:
                return parse(f.read())
            # end with
        # end if

        # 2. Otherwise process the raw text
        with txt_path.open("r", encoding="utf-8") as f:
            text = f.read()
        # end with

        sentences = self.processor.process(text)

        # 3. Cache result as name.conllu
        with conllu_path.open("w", encoding="utf-8") as f:
            f.write(_serialize_sentence_list(sentences=sentences))
        # end with

        return sentences
    
    def process_mp(self,
                   text_files: Union[List[str], List[Path]],
                   n_cpus: Optional[int] = None) -> List[SentenceList]:
        if not text_files:
            return []
        # end if

        if n_cpus is None:
            n_cpus = os.cpu_count()

            if n_cpus and n_cpus > 1:
                n_cpus -= 1
            else:
                n_cpus = 1
            # end if
        # end if

        n_cpus = max(1, min(n_cpus, len(text_files)))

        if n_cpus == 1:
            return [self.process(text_file) for text_file in text_files]
        # end if

        ctx = mp.get_context("spawn")

        with ctx.Pool(processes=n_cpus, initializer=_init_worker,
                      initargs=(self.processor_cls,)) as pool:
            return pool.map(_process_text_file_worker, text_files)
        # emd with

_MP_TEXT_TO_CONLLU: Union[TextToCoNLLU, None] = None


def _init_worker(processor_cls):
    global _MP_TEXT_TO_CONLLU
    _MP_TEXT_TO_CONLLU = TextToCoNLLU(processor_cls())


def _process_text_file_worker(text_file):
    if _MP_TEXT_TO_CONLLU is None:
        raise RuntimeError("Worker was not initialized")
    # end if

    start_time = time.perf_counter()
    result = _MP_TEXT_TO_CONLLU.process(text_file)
    delta_time = time.perf_counter() - start_time

    print(
        f'File [{str(text_file)}] has been processed in [{delta_time:.5f}] seconds')
    return result


def _collect_text_files(folder: Union[str, Path]) -> List[Path]:
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    # end if

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected a folder, got: {folder_path}")
    # end if

    return sorted(folder_path.glob("*.txt"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process Romanian .txt files into cached .conllu files."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing .txt files to process.",
    )
    parser.add_argument(
        "-n",
        "--n-cpus",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to os.cpu_count() - 1.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    text_files = _collect_text_files(args.folder)

    if not text_files:
        print(f"No .txt files found in {args.folder}")
        return
    # end if

    converter = TextToCoNLLU(StanzaProcessor())
    converter.process_mp(text_files, n_cpus=args.n_cpus)


if __name__ == "__main__":
    main()
