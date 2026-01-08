# Project Report: AutoJudge - Programming Problem Difficulty Predictor

**Submitted by:** Yashasvi  
**Date:** January 8, 2026

---

## 1. Introduction

In competitive programming, figuring out how hard a problem actually is can be pretty subjective. What's "Easy" for one person might be "Medium" for another. For this project, I built "AutoJudge" to try and make that process objective. The idea was to create a system that reads a coding problem just like a human would and predicts its difficulty level (Easy, Medium, Hard) and gives it a specific difficulty score (1-10).

This tool is designed to help problem setters balance their contests and to help students find problems that match their current skill level.

## 2. Dataset Overview

To train the model, I collected a dataset of about 4,112 competitive programming problems. I sourced these from various online coding platforms to get a good variety of topics and writing styles.

Each entry in the dataset looks like this:
*   **Problem Text:** The title, description, and input/output examples.
*   **Labels:** The actual difficulty category (Easy/Medium/Hard) and a numerical score.

**Data Distribution:**
About 37% of the problems were "Hard", 27% "Medium", and 15% "Easy". I had to be careful with this imbalance during training to make sure the model didn't just guess "Hard" every time.

## 3. How I Processed the Data

Raw text isn't something machine learning models understand directly, so I had to clean it up first. In `backend/ml/dataset_loader.py`, I did the following:

1.  **Merging:** I combined the Title, Description, and Input/Output details into one long string. This helps the model see the "full picture."
2.  **Cleaning:** I stripped out all the HTML tags (since these were web-scraped), removed special characters that didn't add meaning, and converted everything to lowercase to keep things consistent.
3.  **Splitting:** I set aside 20% of the data just for testing. I used a "stratified split" to make sure the test set had the same proportion of Easy/Medium/Hard problems as the training set.

## 4. Feature Engineering (The "Secret Sauce")

This was the most interesting part. I didn't want to just use raw words, so I combined two different approaches to capture difficulty:

**A. The "Vibe" Features (Statistical)**
I wrote code to count specific things that usually make a problem hard:
*   **Length:** Longer problems are usually more complex to understand.
*   **Math Symbols:** If a problem has a lot of LaTeX logic symbols or math operators, it's usually harder.
*   **Keywords:** I looked for specific words like "graph", "dynamic programming", or "recursion". If these show up, the difficulty usually spikes.

**B. The Words (TF-IDF)**
I used TF-IDF to turn the text into numbers (vectors). This captures the rarity and importance of words. I looked at both single words (unigrams) and pairs of words (bigrams) to catch phrases like "shortest path."

## 5. Models Used

I chose **Random Forest** for this project because it handles this kind of mixed data (numerical counts + text vectors) really well and is less likely to overfit than a single Decision Tree.

*   **Classifier:** To predict the label (Easy/Medium/Hard).
*   **Regressor:** To predict the exact score (1-10).

I trained both models with 100 decision trees each.

## 6. Project Setup

I built this project locally on my Mac. Here is the tech stack I used:
*   **Languages:** Python (for ML) and TypeScript/React (for the UI).
*   **Libraries:** Scikit-learn, Flask, Numpy, Pandas.

## 7. Results

After training, I evaluated the model on the test set that it had never seen before.

**Classification Performance:**
The model reached an accuracy of about **51.6%**. While this might sound low, "difficulty" is very subjective even for humans. The model was quite good at distinguishing straightforward "Easy" problems from complex "Hard" ones, though it sometimes confused "Medium" with the other two categories.

**Regression Performance:**
*   **MAE (Mean Absolute Error):** 1.90. This means on a scale of 1-10, the model's prediction is usually within 2 points of the actual score.

## 8. Web Interface

To make this actually usable, I built a web app.
*   **Frontend:** A React-based UI where you can paste a problem description and get an instant analysis.
*   **Backend:** A Flask API that loads the trained `.joblib` models and serves the predictions.

*(Please see the attached screenshots in the repo for a look at the UI)*

## 9. Conclusion

Building AutoJudge was a great learning experience. It showed me that while we can predict difficulty to some extent using text features and keywords, capturing the "logic" difficulty of a problem is challenging. The current hybrid approach of combining statistical counts with TF-IDF worked better than just using one or the other.

Future improvements could involve using a transformer model like BERT to better understand the *context* of the problem, not just the keywords.

---
**[End of Report]**
