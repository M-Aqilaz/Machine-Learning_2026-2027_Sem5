import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Coba import library ANN (jika belum terinstall, beri penanganan)
try:
    from annoy import AnnoyIndex
    has_annoy = True
except ImportError:
    has_annoy = False
    print("[PERINGATAN] Library 'annoy' belum terinstall di PyDroid3.")

try:
    import hnswlib
    has_hnsw = True
except ImportError:
    has_hnsw = False
    print("[PERINGATAN] Library 'hnswlib' belum terinstall di PyDroid3.")

try:
    import faiss
    has_faiss = True
except ImportError:
    has_faiss = False
    print("[PERINGATAN] Library 'faiss' belum terinstall di PyDroid3.")

# -------------------------------
# Dataset random
# -------------------------------
np.random.seed(42)
n_samples = 10000   # jumlah database vector
d = 128             # dimensi
print("Membuat dataset random 10.000 x 128...")
X = np.random.random((n_samples, d)).astype('float32')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
k = 10

# -------------------------------
# 1. Exact NN (brute-force)
# -------------------------------
print("\nMenjalankan Exact NN...")
start = time.time()
nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
nn.fit(X_scaled)
dist_exact, idx_exact = nn.kneighbors(X_scaled)
time_exact = time.time() - start
print(f"Exact NN done in {time_exact:.3f} s")

# -------------------------------
# 2. Annoy
# -------------------------------
time_annoy = None
if has_annoy:
    print("\nMenjalankan Annoy...")
    start = time.time()
    f = X_scaled.shape[1]
    index_annoy = AnnoyIndex(f, 'euclidean')
    for i, v in enumerate(X_scaled):
        index_annoy.add_item(i, v)
    index_annoy.build(10)
    idx_annoy = [index_annoy.get_nns_by_vector(v, k) for v in X_scaled]
    time_annoy = time.time() - start
    print(f"Annoy done in {time_annoy:.3f} s")

# -------------------------------
# 3. HNSW
# -------------------------------
time_hnsw = None
if has_hnsw:
    print("\nMenjalankan HNSW...")
    start = time.time()
    p_hnsw = hnswlib.Index(space='l2', dim=d)
    p_hnsw.init_index(max_elements=n_samples, ef_construction=200, M=16)
    p_hnsw.add_items(X_scaled)
    p_hnsw.set_ef(200)
    idx_hnsw, _ = p_hnsw.knn_query(X_scaled, k=k)
    time_hnsw = time.time() - start
    print(f"HNSW done in {time_hnsw:.3f} s")

# -------------------------------
# 4. FAISS IVF
# -------------------------------
time_faiss = None
if has_faiss:
    print("\nMenjalankan FAISS...")
    start = time.time()
    quantizer = faiss.IndexFlatL2(d)
    index_faiss = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
    index_faiss.train(X_scaled)
    index_faiss.add(X_scaled)
    index_faiss.nprobe = 10
    _, idx_faiss = index_faiss.search(X_scaled, k)
    time_faiss = time.time() - start
    print(f"FAISS IVF done in {time_faiss:.3f} s")

# -------------------------------
# Ringkasan Waktu
# -------------------------------
print("\n===============================")
print("=== Ringkasan Waktu (detik) ===")
print("===============================")
print(f"Exact NN : {time_exact:.3f} s")
if time_annoy is not None:
    print(f"Annoy    : {time_annoy:.3f} s")
else:
    print("Annoy    : (Tidak terinstall)")

if time_hnsw is not None:
    print(f"HNSW     : {time_hnsw:.3f} s")
else:
    print("HNSW     : (Tidak terinstall)")

if time_faiss is not None:
    print(f"FAISS    : {time_faiss:.3f} s")
else:
    print("FAISS    : (Tidak terinstall)")
