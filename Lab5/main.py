# %%
import hashlib
import logging
import uuid

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
logging.basicConfig(
    filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s"
)


def log_audit(action, user, status):
    logging.info(f"User: {user} | Action: {action} | Status: {status}")


# %%

print("\nPrzygotowanie danych")
# Generowanie syntetycznego zbioru danych klinicznych
np.random.seed(42)
dane = {
    "imie_nazwisko": [
        "Jan Kowalski",
        "Anna Nowak",
        "Piotr Wiśniewski",
        "Maria Wójcik",
        "Krzysztof Kamiński",
    ],
    "wiek": np.random.randint(20, 80, 5),
    "bmi": np.round(np.random.uniform(18.5, 35.0, 5), 1),
    "diagnoza": [1, 0, 1, 1, 0],  # 1 - chory, 0 - zdrowy
    "notatki_lekarza": [
        "Wysokie ciśnienie",
        "Brak uwag",
        "Podejrzenie cukrzycy",
        "Stan stabilny",
        "Wymaga obserwacji",
    ],
}
dataframe = pd.DataFrame(dane)
print("Oryginalne dane:\n", dataframe)


# %%

print("ZADANIE 1: Porównanie algorytmów haszujących")
dane_testowe = b"To jest tajna informacja medyczna"

hash_md5 = hashlib.md5(dane_testowe).hexdigest()
hash_sha1 = hashlib.sha1(dane_testowe).hexdigest()
hash_sha256 = hashlib.sha256(dane_testowe).hexdigest()

print(f"MD5 (podatny na kolizje, 128-bit): {hash_md5}")
print(f"SHA-1 (przestarzały, 160-bit): {hash_sha1}")
print(f"SHA-256 (bezpieczny, 256-bit): {hash_sha256}")

# %%

print("ZADANIE 2: Pseudonimizacja i wpływ na model")
df_pseudonimizowane = dataframe.copy()
# Pseudonimizacja: zamiana danych identyfikujących na losowe tokeny
df_pseudonimizowane["patient_id"] = [
    str(uuid.uuid4()) for _ in range(len(df_pseudonimizowane))
]
df_pseudonimizowane.drop(columns=["imie_nazwisko"], inplace=True)
print("Dane po pseudonimizacji:\n", df_pseudonimizowane.head(2))

# Generalizacja wieku (anonimizacja) - podział na koszyki
df_anonimizowane = df_pseudonimizowane.copy()
df_anonimizowane["wiek"] = pd.cut(
    df_anonimizowane["wiek"], bins=[0, 30, 50, 100], labels=[1, 2, 3]
)

# Prosty test modelu (aby zobaczyć wpływ generalizacji)
X_pseudo = df_pseudonimizowane[["wiek", "bmi"]]
y = dataframe["diagnoza"]
X_anon = df_anonimizowane[["wiek", "bmi"]]

# Model na danych pseudonimizowanych (dokładnych)
clf1 = LogisticRegression().fit(X_pseudo, y)
# Model na danych zgeneralizowanych
clf2 = LogisticRegression().fit(X_anon, y)
print(
    f"Dokładność modelu (dane dokładne): {accuracy_score(y, clf1.predict(X_pseudo)):.2f}"
)
print(
    f"Dokładność modelu (dane zgeneralizowane): {accuracy_score(y, clf2.predict(X_anon)):.2f}"
)


# %%

print("ZADANIE 3: Detekcja naruszeń integralności (manipulacja danymi)")
# Symulacja zapisu do pliku i haszowania
dataframe.to_csv("dane_medyczne.csv", index=False)
with open("dane_medyczne.csv", "rb") as f:
    oryginalny_skrot = hashlib.sha256(f.read()).hexdigest()

print("Oryginalny skrót SHA-256:", oryginalny_skrot)

# Manipulacja danymi (zmiana jednego wpisu)
df_manipulowane = dataframe.copy()
df_manipulowane.loc[0, "bmi"] = 99.9
df_manipulowane.to_csv("dane_medyczne_zmienione.csv", index=False)

with open("dane_medyczne_zmienione.csv", "rb") as f:
    nowy_skrot = hashlib.sha256(f.read()).hexdigest()

print("Skrót po manipulacji danymi:", nowy_skrot)
if oryginalny_skrot != nowy_skrot:
    print("[!] Wykryto naruszenie integralności plików wejściowych!")

# %%

print("ZADANIE 4: System kontroli dostępu RBAC")
uprawnienia_rbac = {
    "administrator": [
        "imie_nazwisko",
        "wiek",
        "bmi",
        "diagnoza",
        "notatki_lekarza",
        "patient_id",
    ],
    "lekarz": [
        "wiek",
        "bmi",
        "diagnoza",
        "notatki_lekarza",
        "patient_id",
    ],  # Nie widzi imienia i nazwiska w tym systemie
    "analityk": ["wiek", "bmi", "diagnoza"],  # Widzi tylko dane zanonimizowane do badań
}


def pobierz_dane_dla_roli(rola, dataset):
    if rola in uprawnienia_rbac:
        dostepne_kolumny = [
            kol for kol in uprawnienia_rbac[rola] if kol in dataset.columns
        ]
        log_audit("Odczyt danych", rola, "SUKCES")
        return dataset[dostepne_kolumny]
    else:
        log_audit("Odczyt danych", rola, "ODMOWA")
        raise PermissionError("Brak dostępu dla tej roli!")


print(
    "Widok dla analityka:\n",
    pobierz_dane_dla_roli("analityk", df_pseudonimizowane).head(2),
)

# %%

print("ZADANIE 6: Szyfrowanie całego pliku i obsługa klucza")
key = Fernet.generate_key()
szyfr = Fernet(key)

# Zapisanie keya (w praktyce w bezpiecznym magazynie np. AWS KMS, HashiCorp Vault)
with open("secret.key", "wb") as keyfile:
    keyfile.write(key)

# Szyfrowanie pliku
with open("dane_medyczne.csv", "rb") as f:
    dane_jawne = f.read()
dane_zaszyfrowane = szyfr.encrypt(dane_jawne)

with open("dane_medyczne_zaszyfrowane.bin", "wb") as f:
    f.write(dane_zaszyfrowane)
print("Plik został zaszyfrowany keyem AES (Fernet).")

# Odszyfrowanie pliku
with open("dane_medyczne_zaszyfrowane.bin", "rb") as f:
    odczytane_zaszyfrowane = f.read()
odszyfrowane = szyfr.decrypt(odczytane_zaszyfrowane)
print("Pomyślnie odszyfrowano plik.")

print("ZADANIE 5 & 7: Analiza ryzyk i audyt (Wyniki tekstowe)")
# Symulacja audytu
with open("audit_log.txt", "r") as f:
    print("Ostatnie logi audytowe:")
    print(f.read())
