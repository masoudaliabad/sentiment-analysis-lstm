# تحلیل احساسات با شبکه LSTM 🧠

پروژه‌ای برای دسته‌بندی احساسات متن توییتر با استفاده از شبکه عصبی LSTM

---

## درباره پروژه

در این پروژه یک مدل یادگیری عمیق بر پایه LSTM ساخته شده که متن توییت‌ها را به سه دسته تقسیم می‌کند:

- مثبت (Positive)
- منفی (Negative)
- خنثی (Neutral)

دیتاست مورد استفاده شامل **73,996 توییت** است که از Kaggle گرفته شده.

---

## مراحل پروژه

**1. بارگذاری و بررسی داده**
- خواندن فایل CSV با ستون‌های label و message
- بررسی مقادیر null و حذف آن‌ها

**2. پیش‌پردازش متن**
- تبدیل به حروف کوچک
- حذف تگ‌های HTML با BeautifulSoup
- حذف لینک‌ها با regex
- حذف علائم نگارشی
- حذف Stop words با spaCy

**3. ساخت مدل LSTM**
- لایه Embedding
- لایه LSTM
- لایه Dense با خروجی 3 کلاس
- بهینه‌ساز Adam

**4. ارزیابی**
- رسم Confusion Matrix
- پیش‌بینی روی متن جدید با دقت بالا

---

## نتیجه مدل

مدل در پیش‌بینی متن جدید دقت بالایی دارد. به عنوان نمونه برای جمله مثبت، احتمال **97.4%** را به کلاس Positive اختصاص داد.

---

## تکنولوژی‌های استفاده شده

- Python
- TensorFlow / Keras
- LSTM Neural Network
- spaCy
- BeautifulSoup
- Pandas / NumPy
- Matplotlib / Seaborn

---

## نحوه اجرا

```bash
pip install tensorflow pandas numpy spacy matplotlib seaborn beautifulsoup4
python -m spacy download en_core_web_sm
```

سپس فایل notebook را در Jupyter یا VS Code اجرا کنید.

---

## بارگذاری مدل ذخیره شده

```python
import pickle
import tensorflow as tf

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model("LSTM_Sentiment_analysis.h5")
```

---

## سازنده

[masoudaliabad](https://github.com/masoudaliabad)
