from conllu.models import TokenList, Token, SentenceList, Metadata
import stanza
import stanza.models.common.doc as stanza_doc
from stanza import DownloadMethod
import spacy
import spacy.tokens as spacy_toks
import trankit


spacy.prefer_gpu(gpu_id=0) # type: ignore


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

            sentences.append(TokenList(tokens, metadata=Metadata({"text": sent.text})))
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

            sentences.append(TokenList(tokens, metadata=Metadata({"text": sent.text})))
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
                TokenList(tokens, metadata=Metadata({"text": sent['text']})))
        # end for sent

        return SentenceList(sentences)


class TextToCoNLLU:
    def __init__(self, processor: BaseProcessor):
        self.processor = processor

    def process(self, text: str) -> SentenceList:
        return self.processor.process(text)
