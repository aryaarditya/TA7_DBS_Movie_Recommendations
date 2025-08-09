
# Laporan Proyek Sistem Rekomendasi Film dengan MovieLens 100K

## 1. Project Overview (Ulasan Proyek)

Perkembangan pesat platform streaming film memberikan banyak pilihan bagi pengguna, namun seringkali menyulitkan mereka menemukan film yang sesuai preferensi pribadi. Sistem rekomendasi film hadir sebagai solusi untuk membantu pengguna menemukan film yang relevan dan menarik berdasarkan minat mereka. 

Proyek ini memanfaatkan dataset MovieLens 100K yang berisi data rating pengguna terhadap film serta metadata film, untuk membangun sistem rekomendasi berbasis **Content-based Filtering** (berdasarkan kemiripan genre film) dan **Collaborative Filtering** (berdasarkan pola rating antar pengguna). 

Sistem ini diharapkan dapat meningkatkan pengalaman pengguna dalam mencari film dan membantu menemukan film baru sesuai minat mereka.


## 2. Business Understanding

### Problem Statements

- Pengguna layanan streaming sering kesulitan memilih film dari ribuan judul yang tersedia.
- Sistem pencarian manual kurang efisien dan kurang personalisasi.
- Diperlukan sistem rekomendasi yang mampu memberikan saran film yang relevan dan sesuai minat pengguna.

### Goals

- Membangun sistem rekomendasi film yang akurat dan relevan dengan preferensi pengguna.
- Menggunakan dua pendekatan utama: Content-based Filtering dan Collaborative Filtering.
- Melakukan evaluasi kuantitatif untuk mengukur performa sistem rekomendasi.

## 3. Data Understanding

### Dataset

Dataset yang digunakan adalah MovieLens 100K yang terdiri dari dua file utama:
- **u.data**: Data interaksi rating pengguna terhadap film, dengan kolom `userId`, `movieId`, `rating`, dan `timestamp`.
- **u.item**: Data metadata film, termasuk `movieId`, `title`, `release_date`, dan 19 kolom genre dengan format one-hot encoding.

### Statistik Data

```python
print(f"Jumlah user unik: {df['userId'].nunique()}")
print(f"Jumlah film unik: {df['movieId'].nunique()}")
print(f"Jumlah rating: {len(df)}")
```

Output:
```
Jumlah user unik: 943
Jumlah film unik: 1682
Jumlah rating: 100000
```

### Link Dataset

Dataset dapat diunduh dari [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/).

### Kondisi Data
Sebelum pemodelan, dilakukan analisis kualitas data sebagai berikut:
-Missing Value
Pada dataset ratings, tidak ditemukan missing value pada seluruh kolom (userId, movieId, rating, timestamp).
Pada dataset movies ditemukan missing value pada beberapa kolomn yaitu release_date 1 baris data kosong, video_release_date 1682 baris data kosong,
dan IMDb_URL 3 baris data kosong, sementara itu, seluruh kolom genre dan kolom movieId/title tidak memiliki missing value.
-Data Duplikat
Tidak ditemukan data duplikat pada kedua dataset.
-Outlier
Tidak ada outlier signifikan karena rating memiliki skala terbatas (1-5).

### Deskripsi Fitur Dataset
File u.data (ratings):
-userId: ID unik pengguna.
-movieId: ID unik film.
-rating: Penilaian pengguna terhadap film (skala 1-5).
-timestamp: Waktu pemberian rating dalam format UNIX timestamp.

File u.item (movies):
-movieId: ID unik film.
-title: Judul film beserta tahun rilis.
-release_date: Tanggal rilis film.
-video_release_date: Tanggal rilis video (jika ada).
-IMDb_URL: Link ke halaman IMDb film.
-19 kolom genre (one-hot encoding) seperti Action, Adventure, Animation, Comedy, dll., dengan nilai 0/1 untuk menandakan genre film.

### Variabel Data

- `userId` : ID unik pengguna.
- `movieId` : ID unik film.
- `rating` : Penilaian pengguna terhadap film (skala 1-5).
- `timestamp` : Waktu penilaian.
- `title` : Judul film.
- `genres` : Gabungan genre film (Animation, Comedy, Drama, dll.).

## 4. Data Preparation

Beberapa teknik data preparation yang diterapkan:

1. **Penggabungan Kolom Genre**  
Karena genre film pada dataset awal berupa one-hot encoding, dilakukan penggabungan genre ke dalam satu kolom bertipe string dengan pemisah `|` untuk memudahkan pemrosesan berbasis teks.

```python
def combine_genres(row):
    return '|'.join([g for g in genre_cols if row[g] == 1])

movies['genres'] = movies[genre_cols].apply(combine_genres, axis=1)
```

2. **Penggabungan Data Rating dan Metadata Film**  
Data rating dan film digabung berdasarkan `movieId` untuk menghasilkan dataset yang lengkap dengan informasi film pada tiap rating.

```python
df = ratings.merge(movies[['movieId', 'title', 'genres']], on='movieId')
```

3. **Transformasi Teks Genre dengan TF-IDF**  
TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah data genre yang berupa teks menjadi representasi vektor numerik. Ini penting agar model content-based filtering dapat mengukur kemiripan antar film berdasarkan genre.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])
```

4. **Prapemrosesan untuk Collaborative Filtering**  
Mempersiapkan data interaksi dalam format yang dapat diterima oleh library LightFM untuk pemodelan collaborative filtering.

```python
dataset = Dataset()
dataset.fit((x for x in ratings['userId']), (x for x in ratings['movieId']))
(interactions, weights) = dataset.build_interactions(
    [(row['userId'], row['movieId'], row['rating']) for idx, row in ratings.iterrows()]
)
```

## 5. Modeling and Result

### Content-Based Filtering

- Menggunakan matriks TF-IDF pada kolom genre untuk merepresentasikan film dalam bentuk vektor.
- Menghitung cosine similarity antar film berdasarkan vektor TF-IDF.
- Fungsi rekomendasi memberikan Top-N film dengan genre paling mirip terhadap film input user.

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommendations(title, top_n=5):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return 'Film tidak ditemukan.'
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]
```

**Contoh Output Content-Based Filtering:**  
Rekomendasi genre mirip dengan *Toy Story (1995)*

| title                              | genres                      |
|------------------------------------|-----------------------------|
| Aladdin and the King of Thieves    | Animation|Children's|Comedy |
| Aristocats, The                    | Animation|Children's        |
| Pinocchio                          | Animation|Children's        |
| Sword in the Stone, The            | Animation|Children's        |
| Fox and the Hound, The             | Animation|Children's        |

### Collaborative Filtering (LightFM)

- Model dibangun menggunakan algoritma WARP pada library LightFM.
- Model mempelajari pola interaksi pengguna-film dari data rating.
- Rekomendasi diberikan untuk user berdasarkan prediksi preferensi.

```python
from lightfm import LightFM

model = LightFM(loss='warp')
model.fit(interactions, epochs=10, num_threads=2)

def get_collab_recommendations(user_id, top_n=5):
    n_items = interactions.shape[1]
    scores = model.predict(user_id, np.arange(n_items))
    top_items = np.argsort(-scores)[:top_n]
    movie_ids = [list(dataset.mapping()[2].keys())[list(dataset.mapping()[2].values()).index(i)] for i in top_items]
    return movies[movies['movieId'].isin(movie_ids)][['title', 'genres']]
```

**Contoh Output Collaborative Filtering:**  
Rekomendasi film untuk user id 1

| title                    | genres                      |
|--------------------------|-----------------------------|
| Rock, The (1996)         | Action|Adventure|Thriller   |
| Independence Day (ID4)   | Action|Sci-Fi|War           |
| Contact (1997)           | Drama|Sci-Fi                |
| Scream (1996)            | Horror|Thriller             |
| Liar Liar (1997)         | Comedy                      |

## 6. Evaluation

### Metrik Evaluasi yang Digunakan

- **Precision@5** : Proporsi film relevan dalam top-5 rekomendasi.
- **Hit@5** : Apakah ada minimal 1 film relevan di top-5.
- **Recall@5** : Proporsi film relevan yang berhasil direkomendasikan dari seluruh film relevan.
- **MAP@5** (Mean Average Precision) : Rata-rata precision pada posisi top-5.
- **RMSE** : Root Mean Squared Error prediksi rating (jika tersedia rating asli).
- **Coverage** : Persentase item unik yang berhasil direkomendasikan ke user (keberagaman rekomendasi).

### Collaborative Filtering - Hit@5
Hit@5: Mengecek apakah minimal 1 film relevan muncul pada top-5 rekomendasi.

```python
def hit_at_k(recommended_ids, relevant_ids, k=5):
    hit = any(i in relevant_ids for i in recommended_ids[:k])
    return int(hit)

print(f'Hit@5 untuk user {user_id}: {hit_at_k(recommended_ids, user_rated, 5)}')
```
Output:
Hit@5 untuk user 1: 1

### Collaborative Filtering - Recall@5
Recall@5: Mengukur proporsi film relevan yang berhasil direkomendasikan pada top-5 rekomendasi.

```python
def recall_at_k(recommended_ids, relevant_ids, k=5):
    if len(relevant_ids) == 0:
        return 0.0
    hit_count = len(set(recommended_ids[:k]) & set(relevant_ids))
    return hit_count / len(relevant_ids)

print(f'Recall@5 untuk user {user_id}: {recall_at_k(recommended_ids, user_rated, 5):.2f}')
```
Output:
Recall@5 untuk user 1: 0.01

### Collaborative Filtering - Precision@5
Precision@5: Mengukur proporsi item pada top-5 rekomendasi yang benar-benar relevan.

```python
user_id = 1
user_rated = df[(df['userId'] == user_id) & (df['rating'] >= 4)]['movieId'].tolist()
recommended = get_collab_recommendations(user_id, top_n=5)
recommended_ids = movies[movies['title'].isin(recommended['title'])]['movieId'].tolist()
precision_at_5 = len(set(user_rated) & set(recommended_ids)) / 5
print(f'Precision@5 untuk user {user_id}: {precision_at_5:.2f}')
```
Output:  
Precision@5 untuk user 1: 0.40

### Collaborative Filtering - MAP@5
MAP@5 (Mean Average Precision@5): Menghitung rata-rata precision pada setiap posisi hingga 5 di daftar rekomendasi.

```python
def apk(actual, predicted, k=5):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0

print(f'MAP@5 untuk user {user_id}: {apk(user_rated, recommended_ids, 5):.2f}')
```
Output:
MAP@5 untuk user 1: 0.23

# Collaborative Filtering - RMSE
RMSE (Root Mean Squared Error): Mengukur rata-rata kesalahan prediksi rating oleh model terhadap data aktual.

```python
from sklearn.metrics import mean_squared_error

user_interactions = interactions.tocsr()[user_id]
known_items = user_interactions.indices
pred_ratings = model.predict(user_id, known_items)
actual_ratings = user_interactions.data

if len(actual_ratings) > 0:
    rmse = np.sqrt(mean_squared_error(actual_ratings, pred_ratings))
    print(f'RMSE untuk user {user_id}: {rmse:.2f}')
else:
    print('User belum punya data rating untuk evaluasi RMSE.')
```
Output:
RMSE untuk user 1: 2.83

# Collaborative Filtering - Coverage
Coverage: Mengukur persentase item unik di dataset yang berhasil direkomendasikan kepada seluruh user.

```python
all_recommended = set()
for uid in range(1, 11):
    recommended = get_collab_recommendations(uid, top_n=5)
    rec_ids = movies[movies['title'].isin(recommended['title'])]['movieId'].tolist()
    all_recommended.update(rec_ids)
coverage = len(all_recommended) / movies['movieId'].nunique()
print(f'Coverage 10 user: {coverage:.2%}')
```
Output:
Coverage 10 user: 1.25%

# Content-Based Filtering - Precision@5
Precision@5: Mengukur proporsi film dengan genre identik di antara 5 rekomendasi teratas pada Content-Based Filtering.

```python
def content_based_precision_at_k(input_title, k=5):
    input_genre = movies.loc[movies['title'] == input_title, 'genres'].values[0]
    ground_truth = movies[(movies['genres'] == input_genre) & (movies['title'] != input_title)].title.tolist()
    rekomendasi = content_based_recommendations(input_title, top_n=k)
    if isinstance(rekomendasi, str): 
        return 0
    recommended_titles = rekomendasi['title'].tolist()
    benar = len(set(recommended_titles) & set(ground_truth))
    return benar / k if k > 0 else 0

title_test = "Toy Story (1995)"
print(f'Content-Based Filtering - Precision@5 untuk {title_test}: {content_based_precision_at_k(title_test, 5):.2f}')
```
Output:
Content-Based Filtering - Precision@5 untuk Toy Story (1995): 0.20

# Content-Based Filtering - Recall@5
Recall@5: Mengukur proporsi film dengan genre identik dari seluruh ground truth yang berhasil direkomendasikan pada Content-Based Filtering.

```python
def content_based_recall_at_k(input_title, k=5):
    input_genre = movies.loc[movies['title'] == input_title, 'genres'].values[0]
    ground_truth = movies[(movies['genres'] == input_genre) & (movies['title'] != input_title)].title.tolist()
    if len(ground_truth) == 0:
        return 0
    rekomendasi = content_based_recommendations(input_title, top_n=k)
    if isinstance(rekomendasi, str):
        return 0
    recommended_titles = rekomendasi['title'].tolist()
    benar = len(set(recommended_titles) & set(ground_truth))
    return benar / len(ground_truth)

print(f'Content-Based Filtering - Recall@5 untuk {title_test}: {content_based_recall_at_k(title_test, 5):.2f}')
```
Output:
Content-Based Filtering - Recall@5 untuk Toy Story (1995): 1.00


### Hasil Evaluasi Model untuk User 1

| Metrik     | Nilai   | Penjelasan                                                                                                       |
|------------|---------|------------------------------------------------------------------------------------------------------------------|
| Hit@5      | 1       | Minimal satu film relevan muncul di top-5 rekomendasi                                                            |
| Recall@5   | 0.01    | Hanya 1% film relevan berhasil muncul di rekomendasi                                                             |
|Precision@5 | 0.40    | menunjukkan bahwa sekitar 40% dari 5 film teratas yang direkomendasikan memang benar-benar disukai oleh pengguna |
| MAP@5      | 0.23    | Rata-rata precision di posisi top-5 adalah 23%                                                                   |
| RMSE       | 2.83    | Rata-rata error prediksi rating dari nilai asli (skala 1-5)                                                      |
| Coverage   | 1.25%   | Model hanya merekomendasikan sekitar 1% dari seluruh koleksi film ke 10 user pertama                             |

### Hasil Content-Based Filtering

| Metrik     | Nilai   | Penjelasan                                                                                                          |
|------------|---------|---------------------------------------------------------------------------------------------------------------------|
|Recall@5    | 1.00    | Semua film yang memiliki genre sama berhasil masuk ke dalam 5 rekomendasi teratas                                   |
|Precision@5 | 0.20    | Jika hasilnya 0.20 (20%), berarti dari 5 rekomendasi teratas, 1 film memiliki genre yang sama dengan film masukkan. |

### Interpretasi dan Hubungan dengan Business Understanding

Hasil evaluasi menunjukkan bahwa sistem rekomendasi mampu memasukkan film relevan ke dalam Top-5 rekomendasi (Hit@5 = 1), yang berarti sistem dapat memberikan saran film yang sesuai dengan preferensi pengguna. Namun, nilai recall dan coverage yang rendah menandakan sistem belum mampu menangkap semua film relevan yang disukai pengguna dan kurang keberagaman dalam rekomendasi. RMSE yang cukup tinggi menunjukkan prediksi rating model masih memiliki sedikit kesalahan.
Dari sisi bisnis, sistem sudah menjawab problem statement terkait kesulitan pengguna dalam menemukan film yang sesuai dengan preferensi mereka. Goals untuk membangun sistem rekomendasi yang relevan sudah sebagian tercapai, terutama pada sisi akurasi rekomendasi Top-N. Namun, untuk meningkatkan pengalaman pengguna dan personalisasi lebih baik, sistem perlu dikembangkan lebih lanjut untuk meningkatkan keberagaman dan akurasi prediksi.
Dengan demikian, solusi yang diimplementasikan berdampak positif terhadap pengalaman pengguna layanan streaming film, dan dapat terus ditingkatkan agar memenuhi kebutuhan pengguna secara lebih menyeluruh.


## Penutup

Proyek ini berhasil membangun dua sistem rekomendasi film dengan pendekatan content-based dan collaborative filtering menggunakan dataset MovieLens 100K. Evaluasi awal menunjukkan potensi sistem dalam memberikan rekomendasi relevan, namun juga mengungkap beberapa kelemahan terutama dalam keberagaman dan akurasi prediksi rating. 

