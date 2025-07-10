# 🧠 IIT Guwahati Student Face Recognition System

This project uses **InsightFace** with ONNX to recognize faces from a dataset of 1000+ IIT Guwahati students.

## 🚀 Features

- Matches uploaded face images with known student photos
- Returns matched student name and visual comparison
- Uses InsightFace embeddings and cosine similarity
- Colab and local compatible

## 🗂️ How It Works

1. Load known student images and extract embeddings
2. Upload a test image
3. Match it against the known embeddings using vector distance
4. Display matched result with name + side-by-side faces

## 📂 Project Structure

