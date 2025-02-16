# Mohs Hardness Prediction 🔬🪶

## 1. Introduction 📖

Hardness is a crucial property of materials that defines their resistance to deformation, wear, and mechanical stress. Predicting hardness accurately is essential in materials science, metallurgy, and industrial applications. Machine learning techniques can help establish relationships between various atomic and chemical properties to predict hardness effectively.

In this project, we aim to develop a predictive model for hardness using a dataset containing atomic and physical properties of different materials. By leveraging regression algorithms, we will analyze key features and determine the most influential factors affecting hardness.

## 2. Aim 🎯

The primary objectives of this study are:

- 🔍 To analyze the relationship between atomic/chemical properties and hardness.
- 🤖 To apply various regression algorithms for hardness prediction.
- 📊 To identify the most influential features contributing to hardness.
- 📈 To evaluate model performance using appropriate metrics (R² score, RMSE, MAE).

## 3. Dataset and Column Explanation 🧪

The dataset consists of various material properties that serve as predictors for hardness. Below is a brief explanation of each column:

| Column Name                    | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `id`                            | Unique identifier for each material sample.                                 |
| `allelectrons_Total`            | Total number of electrons in the material.                                  |
| `density_Total`                 | Total density of the material.                                              |
| `allelectrons_Average`          | Average number of electrons per atomic unit.                                |
| `val_e_Average`                 | Average number of valence electrons.                                        |
| `atomicweight_Average`          | Average atomic weight of the elements in the material.                      |
| `ionenergy_Average`             | Average ionization energy of the elements.                                  |
| `el_neg_chi_Average`            | Average electronegativity of the elements.                                  |
| `R_vdw_element_Average`         | Average van der Waals radius of the elements.                              |
| `R_cov_element_Average`         | Average covalent radius of the elements.                                   |
| `zaratio_Average`               | Average ratio of nuclear charge to atomic radius.                           |
| `density_Average`               | Average material density.                                                  |
| `Hardness`                      | Target variable representing the material's hardness.                       |

## 4. Model Performance Results 📊

The following are the results for various regression models evaluated on the dataset:

| Model                      | R²      | RMSE    | MAE     |
|----------------------------|---------|---------|---------|
| **LGBMRegressor**           | 0.462   | 1.209   | 0.894   |
| **Gradient Boosting**      | 0.448   | 1.225   | 0.922   |
| **RandomForestRegressor**  | 0.431   | 1.243   | 0.921   |
| **XGBRegressor**           | 0.397   | 1.280   | 0.947   |
| **KNeighborsRegressor**    | 0.317   | 1.363   | 1.013   |
| **Ridge**                  | 0.225   | 1.451   | 1.172   |
| **Linear**                 | 0.225   | 1.451   | 1.172   |
| **ElasticNet**             | 0.060   | 1.598   | 1.419   |
| **Lasso**                  | -0.000  | 1.649   | 1.469   |
| **DecisionTreeRegressor**  | -0.148  | 1.767   | 1.223   |
| **Extra Tree**             | -0.159  | 1.775   | 1.272   |

## 5. Conclusion 📝

### 1. Summary of Findings 📑

In this study, we aimed to predict the hardness of materials using machine learning and deep learning models. We trained multiple regression models and evaluated their performance using metrics such as R² score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

- **Machine Learning Models**: Among the machine learning models, **LGBMRegressor** performed the best with:
  - R² Score: 0.462
  - RMSE: 1.209
  - MAE: 0.894

- **Deep Learning Model**: The deep learning model achieved:
  - R² Score: 0.288
  - Mean Squared Error: 1.391

This indicates that the deep learning model did not outperform the best machine learning models in this case.

### 2. Interpretation and Insights 💡

- **LGBMRegressor** emerged as the best model in terms of prediction accuracy, making it the most suitable choice for hardness prediction in this dataset.
- Tree-based ensemble methods (LGBM, Gradient Boosting, and Random Forest) performed better than linear models, suggesting that the relationship between features and hardness is non-linear.
- Deep learning underperformed compared to LGBM and Gradient Boosting, possibly due to limited data, model complexity, or hyperparameter tuning challenges.

## 6. Links 🌐

- **Kaggle Notebook**: [Mohs Hardness Prediction on Kaggle](https://www.kaggle.com/code/senasudemir/mohs-hardness-prediction?scriptVersionId=222815106)
- **Hugging Face Space**: [Mohs Hardness Prediction on Hugging Face](https://huggingface.co/spaces/Senasu/Mohs_Hardness_Prediction)

Happy learning! 🚀
