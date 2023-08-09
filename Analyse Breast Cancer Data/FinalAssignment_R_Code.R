
#install.packages("MASS")

library(tidyverse)
library(caret)
library(glmnet)
library(pROC)
library(corrplot)
library(MASS)
library(ggplot2)

#================= Setting Working Directory and seed =================#
setwd("C://Users//surya//Desktop//Karan//Sem2//AppliedStatisticalModelling//FinalAssignment//HelperCode")
getwd()
set.seed(111)

#================= Read data =================#
data <- read.csv("chowdary.csv")
data <- data[, -1]

print(summary(data))


#================= Checking Null/Missing data =================#
missing_values <- sum(is.na(data))
cat("\n\nMissing values :", missing_values)


#================= Scaling all numeric columns in data =================#
numeric_columns <- data %>% select_if(is.numeric) %>% names()
data_normalized <- data %>% mutate(across(all_of(numeric_columns), scale))


#================= Modifyng predicted variable(tumour) in data =================#
#================= if tumour == B -> then tumour = 0 =================#
#================= if tumour == C -> then tumour = 1 =================#
data_normalized$tumour <- ifelse(data_normalized$tumour == "B", 0, 1)
print(head(data_normalized))


#================= Splitting data into Train and Test set =================#
train_index <- createDataPartition(data_normalized$tumour, p = 0.8, list = FALSE)

train_data <- data_normalized[train_index, ]
test_data <- data_normalized[-train_index, ]


#================= Separating features and predicted variable in Train data =================#
x_train <- model.matrix(tumour ~ ., data = train_data)[,-1]
y_train <- train_data$tumour


#================= Separating features and predicted variable in Test data =================#
x_test <- model.matrix(tumour ~ ., data = test_data)[,-1]
y_test <- test_data$tumour


#=======================================================#
#================= Normalization Tests =================#
#=======================================================#


#================= Shapiro-Wilk test on numeric columns =================#
shapiro_results <- sapply(train_data[, numeric_columns], function(x) shapiro.test(x)$p.value)

print(shapiro_results)

#================= Check variables do not follow a normal distribution =================#
cat("\n\nTotal Features to be tested for normal distribution:", ncol(train_data[, numeric_columns]))

non_normal_vars <- names(shapiro_results)[shapiro_results < 0.05]
cat("\n\nVariables that do not follow a normal distribution are: ", length(non_normal_vars))

#================= Q-Q plots for non-normal variables =================#
num_qq_plots <- 6
selected_non_normal_vars <- non_normal_vars[1:num_qq_plots]

par(mfrow = c(3, 2))

for (var in selected_non_normal_vars) {
  qqnorm(train_data[[var]], main = paste("Q-Q Plot for", var))
  qqline(train_data[[var]], col = "blue")
}


#=====================================================#
#================= Feature Selection =================#
#=====================================================#


#================= Lasso Regression =================#
cvfit_lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", nfolds = 10)

#================= Plot the cross-validation results for Lasso Regression =================#
par(mfrow = c(1, 1))
plot(cvfit_lasso)


#================= Compute Lasso model using Tuned Hyper-parameter =================#
best_lambda_lasso <- cvfit_lasso$lambda.min

cat("\n\nBest lambda for Lasso Model:", best_lambda_lasso)
model_lasso <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda_lasso)


#================= Extract coefficients from the Lasso model =================#
coefficients_lasso <- coef(model_lasso) 
coefficients_lasso <- coefficients_lasso[-1, ]
coefficients_lasso <- coefficients_lasso[order(abs(coefficients_lasso), decreasing = TRUE)]

selected_features_lasso <- coefficients_lasso != 0
selected_coefficients_lasso <- coefficients_lasso[selected_features_lasso]
selected_feature_names_lasso <- names(selected_coefficients_lasso)

cat("\n\nNumber of features selected using Lasso Regression:", sum(coefficients_lasso != 0)) 
cat("\n\nFeatures selected using Lasso Regression: \n")
print(matrix(selected_feature_names_lasso, ncol=1))


#================= Elastic-Net Regression =================#
cvfit_en <- cv.glmnet(x_train, y_train, alpha = 0.9, family = "binomial", nfolds = 10)

#================= Plot the cross-validation results for Elastic-Net Regression =================#
par(mfrow = c(1, 1))
plot(cvfit_en)

#================= Compute Elastic-Net model using Tuned Hyper-parameter =================#
best_lambda_en <- cvfit_en$lambda.min

cat("\n\nBest lambda for Elastic-Net Model:", best_lambda_en)
model_en <- glmnet(x_train, y_train, family = "binomial", alpha = 0.9, lambda = best_lambda_en)


#================= Extract coefficients from the Elastic-Net model =================#
coefficients_en <- coef(model_en) 
coefficients_en <- coefficients_en[-1, ]
coefficients_en <- coefficients_en[order(abs(coefficients_en), decreasing = TRUE)]

selected_features_en <- coefficients_en != 0
selected_coefficients_en <- coefficients_en[selected_features_en]
selected_feature_names_en <- names(selected_coefficients_en)

cat("\n\nNumber of features selected using Elastic-Net Regression:", sum(coefficients_en != 0)) 
cat("\n\nFeatures selected using Elastic-Net Regression: \n")
print(matrix(selected_feature_names_en, ncol=1))



#=====================================================#
#================= Final Model =================#
#=====================================================#

#================= Select Intersection of Features =================#
final_feature_set <- union(selected_feature_names_lasso, selected_feature_names_en)
cat("\n\nNumber of features selected using Feature Selection:", length(final_feature_set)) 
cat("\n\nFeatures selected using Feature Selection: \n")
print(matrix(final_feature_set, ncol=1))



#================= Co-relation Plot for Final Features =================#
train_data_final_features <- train_data[, unique(c("tumour", final_feature_set))]
test_data_final_features <- test_data[, unique(c("tumour", final_feature_set))]

cor_matrix <- cor(train_data_final_features[, !names(train_data_final_features) %in% c("tumour")])
corrplot(cor_matrix, method = "circle", type = "full", tl.col = "black")

# find indices of highly correlated variables
high_cor_idx <- findCorrelation(abs(cor_matrix), cutoff = 0.7)

# extract names of highly correlated variables
high_cor_vars <- names(train_data_final_features)[-1][high_cor_idx]

# print names of highly correlated variables
print(high_cor_vars)

#================= Train Logistic Regression model using Final Features =================#
train_data_final_features <- train_data_final_features[, !names(train_data_final_features) %in% high_cor_vars]
test_data_final_features <- test_data_final_features[, !names(test_data_final_features) %in% high_cor_vars]


logistic_regression_model <- glm(tumour ~ ., data = train_data_final_features)

print(summary(logistic_regression_model))



#================= Evaluate Logistic Regression model =================#
predictions <- predict(logistic_regression_model, newdata = test_data_final_features, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

confusion_matrix <- table(predicted_classes, test_data_final_features$tumour)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * precision * recall / (precision + recall)


cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 score:", f1_score, "\n")
cat("Confusion Matrix:\n")
print(confusion_matrix)


#================= Plots for Logistic Regression model =================#

logistic_regression_coefficients <- summary(logistic_regression_model)$coefficients[-1, 1]
coefficients_df <- data.frame(
  Feature = names(logistic_regression_coefficients),
  Coefficient = as.numeric(logistic_regression_coefficients)
)


# Create a bar graph for the coefficients
ggplot(coefficients_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = ifelse(coefficients_df$Coefficient > 0, "blue", "yellow")) +
  coord_flip() +
  theme_minimal() +
  labs(x = "Feature", y = "Coefficient", title = "Logistic Model Coefficients") +
  theme(plot.title = element_text(hjust = 0.5))


