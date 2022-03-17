_pause = ["unk", "sos", "eos", "ap", "sp"]

_initials = [
    "b",
    "c",
    "ch",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "w",
    "x",
    "y",
    "z",
    "zh",
]

_finals = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "iu",
    "o",
    "ong",
    "ou",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn"
]


symbols = _pause + _initials + _finals

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def label_to_ids(phones):
    # use lower letter
    sequence = [_symbol_to_id[symbol.lower()] for symbol in phones]
    return sequence