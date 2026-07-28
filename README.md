# 🛡️ Amazon Sahte Yorum Tespit Sistemi

## 📌 Proje Hakkında

Bu proje, Amazon ürün yorumlarının sahte (manipülatif) veya gerçek olduğunu makine öğrenmesi yöntemleriyle tespit etmek amacıyla geliştirilmiştir.

Kullanıcı tarafından girilen yorum metni ve yoruma ait davranışsal veriler analiz edilerek eğitilmiş makine öğrenmesi modeli sayesinde yorumun gerçek veya sahte olduğu tahmin edilmektedir.

Uygulama, kullanıcı dostu bir arayüz sunmak amacıyla **Streamlit** kullanılarak geliştirilmiştir.

---

## 🚀 Özellikler

- Amazon yorumlarını analiz eder.
- Sahte ve gerçek yorumları sınıflandırır.
- Streamlit ile geliştirilmiş kullanıcı arayüzüne sahiptir.
- TF-IDF yöntemi ile metin vektörleştirme yapar.
- Davranışsal meta verileri analiz eder.
- Tahmin sonucunu güven oranıyla birlikte gösterir.
- Kullanıcı girişlerini kontrol ederek hatalı veri girişini engeller.

---

## 🧠 Kullanılan Teknolojiler

- Python
- Streamlit
- Scikit-Learn
- Support Vector Machine (SVM)
- TF-IDF Vectorizer
- MinMaxScaler
- Pandas
- NumPy
- SciPy
- NLTK
- Joblib

---

## 📊 Kullanılan Veriler

Model tahmin yaparken aşağıdaki bilgileri kullanmaktadır.

### Yorum Metni

- Kullanıcının yazdığı ürün yorumu

### Davranışsal Veriler

- Yıldız Puanı
- Faydalı Bulan Kişi Sayısı
- Toplam Oy Sayısı
- Yorumun Yayınlanmasının Üzerinden Geçen Gün Sayısı

---

## ⚙️ Çalışma Mantığı

1. Kullanıcı yorum metnini sisteme girer.
2. Metin temizleme işlemleri uygulanır.
3. Stopwords kaldırılır.
4. TF-IDF ile metin sayısal hale dönüştürülür.
5. Sayısal veriler MinMaxScaler ile ölçeklendirilir.
6. Metin ve davranışsal veriler birleştirilir.
7. Eğitilmiş SVM modeli tahmin gerçekleştirir.
8. Sonuç ve güven oranı kullanıcıya gösterilir.

---

## ▶️ Kurulum

Gerekli kütüphaneleri yükleyin.

```bash
pip install -r requirements.txt
```

Projeyi çalıştırın.

```bash
streamlit run app.py
```

---

## 📷 Uygulama Arayüzü

> Buraya uygulamanın ekran görüntüsü eklenecektir.

---

## 🎯 Projenin Amacı

Bu proje, doğal dil işleme (NLP) ve makine öğrenmesi tekniklerini kullanarak sahte ürün yorumlarını tespit edebilen kullanıcı dostu bir karar destek sistemi geliştirmek amacıyla hazırlanmıştır.

---

## 👩‍💻 Geliştirici

**Zehra Sultan Menteş**

Bilgisayar Mühendisliği

GitHub:
https://github.com/zehrasultanmentes
