from processing import TextToCoNLLU, StanzaProcessor, SpacyProcessor, TrankitProcessor

text = "Prima propoziție de test. A doua propoziție, și-a dorit-o mult."
txtproc = TextToCoNLLU(processor=TrankitProcessor())
sentences = txtproc.process(text=text)
a = 1
