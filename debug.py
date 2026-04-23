from processing import TextToCoNLLU, StanzaProcessor, SpacyProcessor, TrankitProcessor
from features import DistribVariabFeatures, RepeatRedundFeatures


with open('data\\small_feature_test_file.txt', mode='r', encoding='utf-8') as f:
    f_text = f.read()
# end with

tpc = TextToCoNLLU(processor=StanzaProcessor())
f_sentences = tpc.process(text=f_text)
thef = {}
dvf = DistribVariabFeatures(sentences=f_sentences)
thef.update(dvf.features())
rrf = RepeatRedundFeatures(sentences=f_sentences)
thef.update(rrf.features())
a = 1
