import csv
import random
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# Function to generate a random sentence
def generate_random_sentence():
    words = ['apple', 'banana', 'cat', 'dog', 'elephant', 'fox', 'gorilla', 'horse', 'iguana', 'jaguar']
    sentence = ' '.join(random.choices(words, k=random.randint(3, 8)))
    return sentence

# Function to encrypt a string using AES
def encrypt_aes(plaintext, key):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext.encode()) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return ciphertext

# Function to encrypt a string using DES
def encrypt_des(plaintext, key):
    backend = default_backend()
    cipher = Cipher(algorithms.TripleDES(key), modes.ECB(), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(64).padder()
    padded_data = padder.update(plaintext.encode()) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return ciphertext

# Function to encrypt a string using ARC4
def encrypt_arc4(plaintext, key):
    backend = default_backend()
    cipher = Cipher(algorithms.ARC4(key), mode=None, backend=backend)
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
    return ciphertext

# List of encryption algorithms
encryption_algorithms = [
    {'name': 'AES', 'function': encrypt_aes},
    {'name': 'DES', 'function': encrypt_des},
    {'name': 'ARC4', 'function': encrypt_arc4}
]

# Generate and encrypt random sentences
data = []
for _ in range(100):
    sentence = generate_random_sentence()
    key = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=16)).encode()
    for algorithm in encryption_algorithms:
        encrypted_string = algorithm['function'](sentence, key)
        data.append([sentence, key.decode(), encrypted_string, algorithm['name']])

# Write data to CSV file
with open('encrypted_sentences.csv', 'a+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Original String', 'Key', 'Encrypted String', 'Encryption Name'])
    writer.writerows(data)

print("Encryption completed. Data saved in 'encrypted_sentences.csv'.")
