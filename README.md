Quantifying the Impact of Climate Change on
Agricultural Yield in Odisha Using LSTM and
Random Forest Models

Abstract—Climate change is a serious threat to agricultural
productivity, in particular, climate-sensitive regions like Odisha,
India. This work studies how changing weather patterns are
impacting crop yields across historical climate data and crop
production records for major crops in the state. We show that
using modern machine learning techniques like Long Short
Term Memory (LSTM) networks and Random Forest (RF)
models, models were built to predict crop yields from various
weather variables as temperature, rainfall, etc. After training and
evaluation, different methods were used for scoring metrics such
as MAE, Root Mean Squared Error (RMSE) and R-squared (R2).
One of the methods was Random Forest and given a high score
in all three parameters, it performed much better than LSTM
model. All the metrics showed consistent and reliable results
even when the models had two-dimensional input variables such
as temperature, rainfall, and so on. In addition, the findings
showed the potential consequences of climate scenarios for future
agricultural output.
Index Terms—Climate change, crop yields, Odisha, weather
data, machine learning, LSTM, Random Forest, agricultural
productivity, prediction, impact assessment.


Climate variability, such as unexpected droughts, unseasonal rainfall, and heatwaves, has a significant impact on agricultural productivity, especially in regions heavily dependent on crop yields like Odisha. This project leverages machine learning techniques, including Long Short-Term Memory (LSTM) networks and Random Forest, to predict crop yield losses due to extreme climate variability. The system uses weather data from the Indian Meteorological Department (IMD) and other climate sources to train predictive models that forecast potential losses based on historical trends. These predictions are evaluated using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R². This system aims to provide actionable insights to farmers by offering real-time alerts and visualizations, helping them take timely measures to mitigate the risks posed by climate fluctuations.

Agricultural yield prediction requires factoring in both re-
lationships in data as well as its temporal patterns. Quan-
titative analyses leveraging climate-yield relationships have
highlighted the sensitivity of production to environmental
changes. For the purpose of this study, a dual model
approach was employed, wherein a sophisticated regression
analysis was performed, utilizing the Random Forest and
Long Short-Term Memory (LSTM) models. Advanced AI-
based models, including deep learning frameworks like LSTM,
have demonstrated their efficacy in predicting agricultural
yields under changing climatic conditions. The models
were chosen for their sequential learning capabilities, and an
attempt was made to navigate the complexities in the overlap
of weather and agricultural data through a parallel model
implementation, as well as a comprehensive and comparative
study of the same.

Random Forest Implementation:

Implemented using scikit-learn's RandomForestRegressor to handle high-dimensional climate-agriculture data.

Configured with 100 decision trees for balancing efficiency and robustness.

Trees grow to natural depth, using internal validation metrics instead of predefined limits to capture complex relationships.

Uses bootstrap aggregating (bagging), training each tree on a random subset of data and features to reduce overfitting.

Final prediction computed as:

𝑦
^
=
1
𝑀
∑
𝑖
=
1
𝑀
ℎ
𝑖
(
𝑥
)
y
^
​
 = 
M
1
​
  
i=1
∑
M
​
 h 
i
​
 (x)
where M is the number of trees, hi is the i-th tree, and x is the input feature vector.

Maintains default scikit-learn parameters for tree depth and minimum samples split to allow natural variance in tree structures.

Feature Importance Analysis identified key factors:

Annual Rainfall → Importance score: 0.186

Area → Importance score: 0.157

Fertilizer Application → Importance score: 0.142

Hyperparameter tuning done via cross-validation:

n_estimators = 100

min_samples_split = 8

min_samples_leaf = 4

max_features = sqrt(total features)

bootstrap = True

random_state = 42 (for reproducibility)

Memory Optimization Techniques used to handle large feature spaces efficiently.

LSTM Architecture:

Designed to capture temporal complexities in agricultural yield predictions using TensorFlow 2.8.0 and Keras.

Input layer: (None, 1, 51), representing the preprocessed feature dimensions.

First LSTM Layer:

128 units with ReLU activation

return_sequences=True to preserve temporal information

Followed by Batch Normalization (momentum=0.99, epsilon=1e-3) to minimize internal covariate shift

Dropout Layer (rate=0.3) to prevent overfitting

Second LSTM Layer:

64 units with ReLU activation

Another Batch Normalization and Dropout Layer (rate=0.2)

Dense Layers:

32 units with ReLU activation

16 units with ReLU activation

Final output layer: 1 unit with linear activation for yield prediction

Training Configuration:

Optimizer: Adam (learning_rate=0.001, beta_1=0.9, beta_2=0.999)

Loss function: Mean Squared Error (MSE)

Batch size: 32

Early stopping used to regulate epochs

Learning rate adjustment via ReduceLROnPlateau:

Reduction factor = 0.2

Patience = 5 epochs

Minimum learning rate = 0.0001


The evaluation metrics were derived from the scikit-learn metrics module. The Mean Absolute Error (MAE) is calculated as:  

**MAE** = (1/n) * Σ |yᵢ - ŷᵢ|  

where `yᵢ` represents the actual yield value and `ŷᵢ` represents the predicted yield value.  

The Root Mean Square Error (RMSE) is computed as:  

**RMSE** = sqrt((1/n) * Σ (yᵢ - ŷᵢ)²)  

The coefficient of determination (**R²**) quantifies the proportion of variance explained by the model:  

**R²** = 1 - (Σ (yᵢ - ŷᵢ)²) / (Σ (yᵢ - ȳ)²)  

where `ȳ` is the mean of the observed values.  

Model accuracy is expressed as a percentage using:  

**Accuracy (%)** = (1 - (MAE / ȳ)) × 100  

Additionally, the Mean Absolute Percentage Error (MAPE) provides a scale-independent error metric:  

**MAPE** = (100/n) * Σ |(yᵢ - ŷᵢ) / yᵢ|  

These metrics were implemented using NumPy operations and scikit-learn’s metrics module, providing a comprehensive assessment of model performance across different scales and interpretations.
