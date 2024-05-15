import re


# очистка мусора в тексте
def clean_html(raw):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    raw = re.sub(cleanr, ' ', raw)
    raw = re.sub('< nt >', '. ', raw)
    cleantext = re.sub("\\n", '. ', raw)
    return cleantext
