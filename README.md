# NLP_example

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge" alt="NLTK" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
</p>

## 🇬🇧 Overview

Sentiment classification of restaurant reviews: text cleaning with regex, stop-word removal and Porter stemming with NLTK, a bag-of-words model with `CountVectorizer`, and a Gaussian Naive Bayes classifier. `nlp.txt` holds Turkish notes on NLP concepts.

**Quick start:** `pip install -r requirements.txt && python nlp.py`

## 🇹🇷 Proje hakkında

Restoran yorumlarının olumlu/olumsuz olarak sınıflandırılması. `nlp.txt` dosyasında doğal dil işleme kavramlarına dair Türkçe notlar vardır.

## ✨ Özellikler

- Düzenli ifadelerle metin temizleme
- NLTK ile stop-word ayıklama ve Porter kök bulma
- `CountVectorizer` ile kelime torbası (2000 özellik)
- Gaussian Naive Bayes ve karışıklık matrisi

## ⚙️ Kurulum ve çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

İlk çalıştırmada NLTK durak kelimelerini indirin:

```bash
python -c "import nltk; nltk.download('stopwords')"
python nlp.py
```

## 📁 Dosya yapısı

```text
NLP_example/
├── nlp.py
├── nlp.txt
└── Restaurant_Reviews.csv
```
