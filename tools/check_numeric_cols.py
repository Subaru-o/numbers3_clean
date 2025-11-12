# check_numeric_cols.py
import pandas as pd, unicodedata as u
p = r'data/raw/20201102-20251031_Numbers3features.csv'
df = pd.read_csv(p, encoding='utf-8-sig')
obj = df.select_dtypes('object').columns

def norm(s):
    s = s.astype(str).map(lambda x: u.normalize('NFKC', x))
    s = s.str.replace('−', '-', regex=False)
    s = s.str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.replace('¥', '', regex=False).str.strip()
    return pd.to_numeric(s, errors='coerce')

pre = set(df.select_dtypes('number').columns)
df[obj] = df[obj].apply(norm)
post = set(df.select_dtypes('number').columns)
print(sorted(post - pre))
