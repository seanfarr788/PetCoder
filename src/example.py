from pettag import DiseaseCoder

pipe = DiseaseCoder(framework='icd11')

text = "Cookie present with vomiting and diarrhea. Suspected gastroenteritis caused by parvovirus."
pipe.predict(text.upper())