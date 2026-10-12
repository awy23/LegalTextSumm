import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
stop_words_en = stopwords.words('english')
stop_words_ro = stopwords.words('romanian')

stop_words_ro.extend(['art', 'nr', 'alin', 'privind', 'care', 'este', 'pentru', 'din', 'în', 'la'])
stop_words_en.extend(['section', 'act', 'shall', 'state', 'united', 'states'])

ds_billsum = load_dataset("FiscalNote/billsum", split="train[:2000]")
df_billsum = pd.DataFrame(ds_billsum)

ds_rb = load_dataset("readerbench/ro-text-summarization", split="train")
df_rb = pd.DataFrame(ds_rb)
df_rb = df_rb[df_rb['Category'].str.lower().isin(["guvern", "justitie"])]
df_rb = df_rb.dropna(subset=['Content', 'Summary'])

def calculate_statistics(df, col_text, col_summary):
    df['text_word_count'] = df[col_text].apply(lambda x: len(str(x).split()))
    df['summary_word_count'] = df[col_summary].apply(lambda x: len(str(x).split()))
    df['compression_ratio'] = df['summary_word_count'] / df['text_word_count']
    return df

df_billsum = calculate_statistics(df_billsum, 'text', 'summary')
df_rb = calculate_statistics(df_rb, 'Content', 'Summary')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(df_billsum['text_word_count'], bins=50, ax=axes[0, 0], color='blue')
axes[0, 0].set_title('BillSum: Word Count Distribution (Source Text)')
axes[0, 0].set_xlabel('Word Count')
axes[0, 0].set_ylabel('Count')

sns.histplot(df_billsum['summary_word_count'], bins=50, ax=axes[0, 1], color='cyan')
axes[0, 1].set_title('BillSum: Word Count Distribution (Summary)')
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('Count')

sns.histplot(df_rb['text_word_count'], bins=50, ax=axes[1, 0], color='red')
axes[1, 0].set_title('ReaderBench: Word Count Distribution (Source Text)')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Count')

sns.histplot(df_rb['summary_word_count'], bins=50, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('ReaderBench: Word Count Distribution (Summary)')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

print("\n Compression Ratio Statistics")
print(f"BillSum Mean Compression: {df_billsum['compression_ratio'].mean():.2f}")
print(f"ReaderBench Mean Compression: {df_rb['compression_ratio'].mean():.2f}")

def plot_top_ngrams(corpus, n_gram_range, stopwords_lang, title, color, ax):
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words=stopwords_lang).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:15]

    df_freq = pd.DataFrame(words_freq, columns=['N-gram', 'Frequency'])
    sns.barplot(x='Frequency', y='N-gram', data=df_freq, color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('N-gram')

print("\nGenerate word frequency plots")
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 12))

plot_top_ngrams(df_billsum['text'].head(1000), (2,2), stop_words_en, 'Top 15 Bigrams - BillSum (English)', 'blue', axes2[0])
plot_top_ngrams(df_rb['Content'].head(1000), (2,2), stop_words_ro, 'Top 15 Bigrams - ReaderBench (Romanian)', 'red', axes2[1])

plt.tight_layout()
plt.show()

kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

def extract_keywords_sample(df, col_text, n_samples=5):
    sample_texts = df[col_text].sample(n_samples, random_state=42).tolist()
    results = []
    for text in sample_texts:
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=5)
        results.append([kw[0] for kw in keywords])
    return results

print("\n KeyBERT: Representative Keywords (BillSum EN)")
keywords_billsum = extract_keywords_sample(df_billsum, 'text')
for i, kw_list in enumerate(keywords_billsum):
    print(f"Doc {i+1}: {kw_list}")

print("\n KeyBERT: Representative Keywords (ReaderBench RO)")
keywords_rb = extract_keywords_sample(df_rb, 'Content')
for i, kw_list in enumerate(keywords_rb):
    print(f"Doc {i+1}: {kw_list}")