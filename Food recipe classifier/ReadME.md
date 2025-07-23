# Food Recipes Classification with NLP

This project uses natural language processing (NLP) techniques to classify recipes as **nutritious** or **not nutritious** based on their ingredients and instructions. The classification model is trained on data from the **RecipeNLG dataset**, with nutritional labels derived from the **USDA FoodData Central API**.

---

## Project Objective

- Predict the nutritional quality of a recipe using only its text content (ingredients and instructions).
- Build a binary classifier that labels recipes as nutritious or not.
- Generate ground truth labels by mapping recipe ingredients to nutrient values and computing a nutritional score based on USDA guidelines.

---

## Data Sources

### RecipeNLG Dataset
This project uses the [RecipeNLG dataset](https://www.kaggle.com/datasets/wisekim/recipenlg) available on Kaggle.  
Due to licensing and size limitations, the dataset is **not included in this repository**.

**To use this project:**
1. Go to the [RecipeNLG dataset page](https://www.kaggle.com/datasets/wisekim/recipenlg).
2. Download the dataset manually (`recipenlg.zip`).
3. Extract the relevant JSON file(s) and place them in your working directory.

### USDA FoodData Central API
Nutritional values for individual ingredients are obtained using the [USDA FoodData Central API](https://fdc.nal.usda.gov/api-key-signup.html).  
An API key is required to query the database — sign up for one if needed.

---

## Repository Contents

- `recipe_classifier_presentation.ipynb`  
  The **main notebook** presenting the complete workflow: data loading, preprocessing, scoring, model training, and evaluation.

- `Build_final_df.ipynb`  
  Auxiliary notebook used to **construct the final training dataset** by computing nutritional scores and assigning binary labels.

- `Process_recipe_dictionary.ipynb`  
  Processes ingredient-level data by interacting with the **USDA FoodData Central API**, retrieving nutritional information, and storing it for scoring purposes.

---

## Method Overview

1. **Data Preprocessing**: Recipes are cleaned and normalized.
2. **Ingredient Scoring**: Ingredients are mapped to USDA entries via API and nutrient profiles are extracted.
3. **Labeling**: Each recipe receives a nutritional score based on its ingredients, which is converted to a binary label.
4. **Text Vectorization**: Ingredients and instructions are tokenized and transformed into feature vectors.
5. **Classification**: A binary classifier is trained and evaluated using standard metrics (accuracy, precision, recall, F1-score).

---

> **Note:** This repository currently does not include a `requirements.txt` file.  

