import streamlit as st
import pickle
import numpy as np

# Memuat model dan scaler dari file pickle
with open('random_forest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Fungsi untuk memprediksi spesies
def predict_species(length, weight, w_l_ratio):
    # Menyiapkan data input (dimensi 1x3)
    input_data = np.array([[length, weight, w_l_ratio]])
    
    # Lakukan scaling pada data input
    scaled_data = scaler.transform(input_data)
    
    # Prediksi spesies menggunakan model
    prediction = model.predict(scaled_data)
    
    # Kembalikan hasil prediksi
    return prediction[0]

# Antarmuka aplikasi menggunakan Streamlit
def main():
    st.title("Prediksi Spesies Ikan")
    st.write("Masukkan data ikan untuk memprediksi spesies berdasarkan panjang, berat, dan rasio berat terhadap panjang ikan.")

    # Input dari pengguna
    length = st.number_input("Masukkan panjang ikan (length):", min_value=0.0, step=0.1)
    weight = st.number_input("Masukkan berat ikan (weight):", min_value=0.0, step=0.1)
    w_l_ratio = st.number_input("Masukkan rasio berat terhadap panjang ikan (w_l_ratio):", min_value=0.0, step=0.01)

    # Tombol untuk memprediksi
    if st.button("Prediksi Spesies"):
        if length > 0 and weight > 0 and w_l_ratio > 0:
            # Prediksi spesies
            species = predict_species(length, weight, w_l_ratio)
            
            # Menampilkan hasil prediksi
            st.success(f"Prediksi spesies ikan: {species}")
        else:
            st.error("Pastikan semua input valid dan lebih besar dari 0!")

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
