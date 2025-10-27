from pettag import DiseaseCoder

pipe = DiseaseCoder()

text = "COOKIE PRESENTED TO JACKSON'S ON 25TH MAY 2025 BEFORE TRAVEL TO HUNGARY WITH PNEUMONIA. ISSUED PASSPORT (GB52354324)"
pipe.predict(text.upper())