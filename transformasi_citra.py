import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Membaca gambar ----------------
img = cv2.imread("src/meow.jpg")
if img is None:
    raise FileNotFoundError("File 'src/meow.jpg' tidak ditemukan. Pastikan berada di folder yang sama.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tinggi, lebar = img.shape[:2]
pusat = (lebar // 2, tinggi // 2)


# ---------------- Fungsi rotasi (kanvas diperbesar agar tidak terpotong) ----------------
def rotasi(citra, sudut):
    t, l = citra.shape[:2]
    M = cv2.getRotationMatrix2D((l // 2, t // 2), sudut, 1.0)

    # Hitung ukuran kanvas baru
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    l_baru = int(t * sin + l * cos)
    t_baru = int(t * cos + l * sin)

    # Geser agar gambar tetap berada di tengah kanvas baru
    M[0, 2] += l_baru / 2 - l // 2
    M[1, 2] += t_baru / 2 - t // 2

    return cv2.warpAffine(citra, M, (l_baru, t_baru))


# ---------------- 2. Translation ----------------
M_translasi = np.float32([
    [1, 0, 150],   # geser 150 piksel ke kanan
    [0, 1, 75]     # geser 75 piksel ke bawah
])
hasil_translasi = cv2.warpAffine(img, M_translasi, (lebar, tinggi))

# ---------------- 3. Scaling (faktor 0,5 dan 2) ----------------
scaling_05 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
scaling_2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# ---------------- 4. Rotation (45°, 90°, 180°) ----------------
rotasi_45 = rotasi(img, 45)
rotasi_90 = rotasi(img, 90)
rotasi_180 = rotasi(img, 180)

# ---------------- 5. Flip (horizontal dan vertical) ----------------
flip_horizontal = cv2.flip(img, 1)   # 1 = horizontal
flip_vertical = cv2.flip(img, 0)     # 0 = vertical

# Informasi ukuran
print("Ukuran citra asli     :", img.shape)
print("Ukuran scaling 0,5    :", scaling_05.shape)
print("Ukuran scaling 2      :", scaling_2.shape)
print("Ukuran rotasi 45      :", rotasi_45.shape)
print("Ukuran rotasi 90      :", rotasi_90.shape)
print("Ukuran rotasi 180     :", rotasi_180.shape)

# ---------------- 6. Menampilkan citra asli dan seluruh hasil ----------------
daftar = [
    (img, f"Citra Asli ({lebar} x {tinggi})"),
    (hasil_translasi, "Translation (x=150, y=75)"),
    (scaling_05, f"Scaling 0,5 ({scaling_05.shape[1]} x {scaling_05.shape[0]})"),
    (scaling_2, f"Scaling 2 ({scaling_2.shape[1]} x {scaling_2.shape[0]})"),
    (rotasi_45, "Rotation 45°"),
    (rotasi_90, "Rotation 90°"),
    (rotasi_180, "Rotation 180°"),
    (flip_horizontal, "Horizontal Flip"),
    (flip_vertical, "Vertical Flip"),
]

plt.figure(figsize=(15, 12))

for i, (gambar, judul) in enumerate(daftar, start=1):
    plt.subplot(3, 3, i)
    plt.imshow(gambar)
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()
