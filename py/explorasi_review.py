
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download('stopwords')

#Baca Data#
df = pd.read_csv("reviews.csv")  # ganti dengan path lokalmu jika perlu

#Bersihkan Kolom Rating#
df['Review Rating'] = df['Review Rating'].str.extract(r'(\d+)').astype('Int64')

#Ubah Tanggal#
df['Review date'] = pd.to_datetime(df['Review date'], dayfirst=True, errors='coerce')

#Hapus Stopwords Bahasa Inggris#
stop_words = set(stopwords.words('english'))

def remove_stopwords_en(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

df['Cleaned_Review'] = df['Review_body'].astype(str).apply(remove_stopwords_en)

#Tambahkan Panjang Review#
df['Review_length'] = df['Cleaned_Review'].apply(len)

#Visualisasi: Distribusi Rating#
plt.figure(figsize=(8, 5))
sns.countplot(x='Review Rating', data=df, palette='viridis')
plt.title("Distribusi Review Rating")
plt.xlabel("Rating")
plt.ylabel("Jumlah Review")
plt.tight_layout()
plt.show()

#Tren Review per Hari#
review_per_day = df['Review date'].value_counts().sort_index()
plt.figure(figsize=(12, 5))
review_per_day.plot(marker='o')
plt.title("Jumlah Review per Hari")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Review")
plt.grid()
plt.tight_layout()
plt.show()

#Panjang Review Berdasarkan Rating#
plt.figure(figsize=(10, 6))
sns.boxplot(x='Review Rating', y='Review_length', data=df)
plt.title("Panjang Review Berdasarkan Rating")
plt.tight_layout()
plt.show()

#Kata yang Sering Muncul (Setelah Bersih)#
all_words = ' '.join(df['Cleaned_Review'].dropna()).split()
common_words = Counter(all_words).most_common(20)
common_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

plt.figure(figsize=(10, 6))
sns.barplot(y='Word', x='Frequency', data=common_df, palette='mako')
plt.title("20 Kata yang Paling Sering Muncul dalam Review")
plt.tight_layout()
plt.show()

#Sentimen: Positif vs Negatif#
df['Sentiment'] = df['Review Rating'].apply(lambda x: 'Positif' if x >= 7 else 'Negatif')
plt.figure(figsize=(6, 4))
sns.countplot(x='Sentiment', data=df, palette='pastel')
plt.title("Distribusi Sentimen Review")
plt.tight_layout()
plt.show()

#Korelasi Panjang Review & Rating#
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Review_length', y='Review Rating', data=df, alpha=0.5)
plt.title("Hubungan Panjang Review dan Rating")
plt.xlabel("Panjang Review (karakter)")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

#Word Cloud#
all_text = ' '.join(df['Cleaned_Review'].dropna())
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(all_text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud dari Review (Setelah Hapus Stopwords)", fontsize=16)
plt.tight_layout()
plt.show()
