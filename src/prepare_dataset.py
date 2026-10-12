import pandas as pd
import re
import html
import unicodedata
from datasets import load_dataset
from transformers import AutoTokenizer

tok_llama_en = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-instruct")
tok_phi_en = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")
tokenizere_en = [tok_llama_en, tok_phi_en]

tok_llama_ro = AutoTokenizer.from_pretrained("OpenLLM-Ro/RoLlama3.1-8b-Instruct")
tok_mistral_ro = AutoTokenizer.from_pretrained("OpenLLM-Ro/RoMistral-7b-Instruct")
tokenizere_ro = [tok_llama_ro, tok_mistral_ro]

PROMPT_RO = """Ești un expert juridic. Rezumă punctele principale ale următorului document.
Regula 1: Nu folosi fraze introductive.
Regula 2: Păstrează textul concis.
Regula 3: Asigură-te că toate numerele, termenele limită și drepturile legale sunt păstrate exact și sunt incluse explicit în rezumatul tău.

Document:
"""
PROMPT_EN = """As a legal expert, summarize the main points of the following document.
Rule 1: Do not use introductory phrases.
Rule 2: Keep the text concise.
Rule 3: Ensure all numbers, deadlines, and legal rights are preserved exactly and are explicitly included in your summary.

Document:
"""

def clean_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocesare_optimizata(nume_dataset, filtru_cat, col_text, col_summary, prompt_lang, lista_tokenizere, tokenizer_baza, nume_fisier):
    print(f"\n Procesare {nume_dataset}")

    ds = load_dataset(nume_dataset, split="train")
    df = pd.DataFrame(ds)

    if filtru_cat:
        df = df[df['Category'].str.lower().isin(filtru_cat)]

    df = df.dropna(subset=[col_text, col_summary])

    df['text_clean'] = df[col_text].apply(clean_text)
    df['summary_clean'] = df[col_summary].apply(clean_text)
    df = df.drop_duplicates(subset=['text_clean'])

    df['formatted_text'] = df.apply(
        lambda row: f"{prompt_lang}{row['text_clean']}\n\nRezumat:\n{row['summary_clean']}",
        axis=1
    )

    print(f"Masurare tokeni")

    def calculeaza_max_tokeni(text):
        lungimi = [len(tok.encode(text)) for tok in lista_tokenizere]
        return max(lungimi)

    df['max_tokens'] = df['formatted_text'].apply(calculeaza_max_tokeni)

    df['sum_tokens'] = df['summary_clean'].apply(lambda x: len(tokenizer_baza.encode(x)))
    df['text_tokens'] = df['text_clean'].apply(lambda x: len(tokenizer_baza.encode(x)))

    masca = (
        (df['max_tokens'] <= 2048) &
        (df['text_tokens'] >= 150) &
        (df['sum_tokens'] >= 20) &
        (df['sum_tokens'] < (df['text_tokens'] * 0.5))
    )

    df_curat = df[masca]
    print(f"Documente compatibile cu arhitecturile selectate: {len(df_curat)}")

    if len(df_curat) >= 800:
        df_final = df_curat.sample(n=800, random_state=42)
        df_final[['formatted_text']].to_csv(nume_fisier, index=False)
        print(f"SALVAT: {nume_fisier} (800 legi)")
    else:
        print("EROARE")

preprocesare_optimizata(
    nume_dataset="readerbench/ro-text-summarization",
    filtru_cat=["justitie", "guvern"],
    col_text="Content",
    col_summary="Summary",
    prompt_lang=PROMPT_RO,
    lista_tokenizere=tokenizere_ro,
    tokenizer_baza=tok_llama_ro,
    nume_fisier="dataset_ro_preprocesat_800.csv"
)

preprocesare_optimizata(
    nume_dataset="billsum",
    filtru_cat=None,
    col_text="text",
    col_summary="summary",
    prompt_lang=PROMPT_EN,
    lista_tokenizere=tokenizere_en,
    tokenizer_baza=tok_llama_en,
    nume_fisier="dataset_en_preprocesat_800.csv"
)