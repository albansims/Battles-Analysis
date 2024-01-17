library(randomForest)
library(dplyr)
library(caret)

# set working directory
setwd("C:/Users/Alban/Desktop/Battles Data")
# load dataset
battle_data <- read.csv("GeoBattleData_YZ.csv")

# Load necessary libraries
library(randomForest)
library(dplyr)
library(caret)

# Load your data
battle_data <- read.csv('path_to_your_data.csv')

# Preprocess the data: select relevant variables and convert factors if needed
model_data <- battle_data %>%
  select(acas_per, long, lat, dist_att, dist_def, # Corrected variable names
         elev_mean_h, elev_min_h, elev_max_h, elev_var_h, elev_sd_h, # Elevation variables
         a_frontage, d_frontage, soft_ar, tree_ar, river, # Terrain variables
         wet, overcast, temp, # Weather variables
         a_tks_to, a_tks_lt, a_tks_mb, a_sortie, a_arty, # Attacker's technology
         d_tks_to, d_tks_lt, d_tks_mb, d_sortie, d_arty, # Defender's technology
         ffr, afsr, dfsr, # Force ratios
         frontat, envelop, defoff, surprise, fort) %>% # Strategy variables
  na.omit() # Deal with missing values as necessary

# Encode categorical variables with one-hot encoding
dummies <- dummyVars(" ~ .", data = model_data)
model_data_preprocessed <- data.frame(predict(dummies, newdata = model_data))

# Normalize numerical variables
preProcess_range_model <- preProcess(model_data_preprocessed, method = c("range"))
model_data_normalized <- predict(preProcess_range_model, model_data_preprocessed)

# Split the data into training and test sets
set.seed(123) # For reproducibility
indexes <- createDataPartition(model_data_normalized$acas_per, p = 0.8, list = FALSE)
train_data <- model_data_normalized[indexes, ]
test_data <- model_data_normalized[-indexes, ]

# Train the random forest regression model
set.seed(123)
rf_model <- randomForest(acas_per ~ ., data = train_data, ntree = 500)

# Make predictions on the test set
predictions <- predict(rf_model, test_data)

# Evaluate the model performance
mse <- mean((predictions - test_data$acas_per)^2)
rmse <- sqrt(mse)

print(paste("Mean Squared Error: ", mse))
print(paste("Root Mean Squared Error: ", rmse))

# Feature importance
importance(rf_model)


# Cross-Validation
set.seed(123)
cv_model <- train(acas_per ~ ., data = train_data, method = "rf", trControl = trainControl("cv", number = 10))
print(cv_model)

# Residual Analysis on Training Data
train_predictions <- predict(rf_model, train_data)
residuals <- train_data$acas_per - train_predictions
plot(residuals, ylab = "Residuals")
abline(h = 0, col = "red")

# Feature Importance Reevaluation
importance(cv_model$finalModel)

# Learning Curve Analysis
learning_curve_data <- learning_curve_dat(train_data, outcome = "acas_per", train_control = trainControl(method = "cv", number = 10), method = "rf")
# Create the learning curve plot for RMSE
ggplot(learning_curve_data, aes(x = Training_Size, y = RMSE)) +
  geom_line() + 
  geom_point() +
  labs(x = "Training Size", y = "RMSE") +  # Customize axis labels
  ggtitle("Learning Curve for RMSE")      # Customize plot title

# Correlation Analysis Between Variables and Residuals
data_with_residuals <- cbind(train_data, residuals = residuals)
correlations <- cor(data_with_residuals)
print(correlations)

# Model Comparison with Linear Regression
lm_model <- lm(acas_per ~ ., data = train_data)
lm_predictions <- predict(lm_model, test_data)
lm_mse <- mean((lm_predictions - test_data$acas_per)^2)
print(paste("Linear Model MSE: ", lm_mse))
