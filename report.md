# Project Report: AutoJudge - Programming Problem Difficulty Predictor

**Submitted by:** Yashasvi  
**Date:** January 8, 2026

---

## 1. Introduction

In competitive programming, figuring out how hard a problem actually is can be pretty subjective. What's "Easy" for one person might be "Medium" for another. For this project, I built "AutoJudge" to try and make that process objective. The idea was to create a system that reads a coding problem just like a human would and predicts its difficulty level (Easy, Medium, Hard) and gives it a specific difficulty score (1-10).

This tool is designed to help problem setters balance their contests and to help students find problems that match their current skill level.

## 2. Dataset Overview

To train the model, I utilized the provided dataset containing 4,112 competitive programming problems. This ensures the model is trained on the standard benchmark data required for the project.

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

*   **Classifier:** Random Forest Classifier (300 trees) to predict difficulty category.
*   **Regressor:** Gradient Boosting Regressor (500 estimators) to predict scores.
*   **Feature Stacking**: To solve the "Hard problem, Low score" disconnect, I implemented **Stacked Generalization**. The probabilities from the Classifier (e.g., `P(Hard)=0.9`) are fed as *extra features* into the Regressor. This allows the Regressor to learn that high confidence in "Hard" should correlate with higher scores (7-10), without using rigid rules.

## 6. Project Setup

I built this project locally on my Mac. Here is the tech stack I used:
*   **Languages:** Python (for ML) and TypeScript/React (for the UI).
*   **Libraries:** Scikit-learn, Flask, Numpy, Pandas.

### 7. Results and Evaluation

The trained model was evaluated on the test set (20% split). The results below show the performance of our **improved** models.

#### Classification Metrics (Difficulty Category)
Our Random Forest classifier achieved a test accuracy of **52.9%**, which is a significant improvement over baseline guessing (33%).

*   **Overall Accuracy:** 52.9%
*   **Precision:**
    *   Easy: 0.49
    *   Medium: 0.47
    *   Hard: 0.54
*   **Recall:**
    *   Easy: 0.44 (Significantly improved via dataset balancing)
    *   Medium: 0.10
    *   Hard: 0.88
*   **F1-Score:**
    *   Easy: 0.46
    *   Medium: 0.16
    *   Hard: 0.67

These results indicate that the model is particularly good at identifying "Hard" problems. The "Easy" class recognition has improved, though "Medium" problems are often misclassified as Hard or Easy due to feature overlap.

#### Regression Metrics (Difficulty Score)
The regression model predicts the exact difficulty score (1-10).

*   **Mean Absolute Error (MAE):** 1.92
*   **Root Mean Squared Error (RMSE):** 2.26

On average, the model's difficulty score prediction is within ±1.9 points of the actual score.

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
