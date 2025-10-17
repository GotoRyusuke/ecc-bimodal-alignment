def remove_punc(word):
    return ''.join(char for char in word if char.isalpha())

def preprocess(text, sep='|'):
    tokens = text.upper().split()
    words = [remove_punc(token) for token in tokens]
    return sep.join([word for word in words if len(word) > 0])