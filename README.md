# 📚 Book Recomender – Chatbot AI pentru recomandări de cărți

Un asistent AI care recomandă cărți pe baza intereselor utilizatorului, folosind RAG (Retrieval-Augmented Generation) cu ChromaDB și OpenAI GPT.

---

## 🚀 Funcționalități

* Chatbot AI care înțelege întrebări despre cărți și răspunde conversațional
* Vector store (ChromaDB) pentru căutare semantică
* Tool calling pentru rezumat detaliat
* Interfață web simplă în Streamlit
* Moderare pentru întrebări necorespunzătoare

---

## 🔧 Setup local

### 1. Clonează proiectul

```bash
git clone https://github.com/PeleMihnea/Book-Recomender.git
cd Book-Recomender
```

### 2. Adaugă cheia OpenAI

Creează un fișier `.env`:

```env
OPENAI_API_KEY=my-secret-key
```

### 3. Instalează dependințele (opțional, pentru test local)

```bash
pip install -r requirements.txt
```

### 4. Rulează local aplicația (fără Docker)

Într-un terminal:

```bash
uvicorn src.backend.app:app --reload
```

În alt terminal:

```bash
streamlit run src/frontend/app.py
```

---

## 🧪 Teste rapide

### Exemple întrebări:

* "Vreau o carte despre libertate și control social"
* "Ce îmi recomanzi dacă iubesc poveștile fantastice?"
* "Ce este 1984?"

---
