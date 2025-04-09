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
