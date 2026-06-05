
#installing and loading necessary libraries

#install.packages(c("tidyverse","caret","class","rpart","rpart.plot","randomForest","e1071"))

library(tidyverse)
library(caret)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)

#load the Dataset

data = read.csv(file.choose(), header = TRUE, sep = ",")

head(data)
str(data)

#Creating target variable

data$target <- as.factor(data$target)

#checking missing values 

colSums(is.na(data))

#Changing target value diseased-1 not_diseased-0

data$target <- factor(data$target, levels = c(1,0))

#Spliting the data 70-30

set.seed(123)

trainIndex <- createDataPartition(data$target, p=0.7, list=FALSE)

train <- data[trainIndex,]
test <- data[-trainIndex,]
train
test

#Logistic Regression 

log_model <- glm(target ~ ., data=train, family="binomial")

pred_log <- predict(log_model, test, type="response")

pred_log <- ifelse(pred_log > 0.5, "0", "1")

# IMPORTANT FIX
pred_log <- factor(pred_log, levels = c("1","0"))

# Ensure test also same
test$target <- factor(test$target, levels = c("1","0"))

log_acc <- confusionMatrix(pred_log, test$target)
log_acc
str(data)



#KNN

train_x <- scale(train[,-14])
test_x <- scale(test[,-14])

scaling_params <- list(
  center = attr(train_x, "scaled:center"),
  scale  = attr(train_x, "scaled:scale")
)

train_y <- train$target
test_y <- test$target

knn_pred <- knn(train_x, test_x, train_y, k=5)

knn_acc <- confusionMatrix(knn_pred, test_y)
knn_acc



#Decision tree

tree_model <- rpart(target ~ ., data=train, method="class")

rpart.plot(tree_model)

tree_pred <- predict(tree_model, test, type="class")

tree_acc <- confusionMatrix(tree_pred, test_y)
tree_acc


#Random forest

rf_model <- randomForest(target ~ ., data=train, ntree=100)

rf_pred <- predict(rf_model, test)

rf_acc <- confusionMatrix(rf_pred, test_y)
rf_acc

#SVM

svm_model <- svm(target ~ ., data=train)

svm_pred <- predict(svm_model, test)

svm_acc <- confusionMatrix(svm_pred, test_y)
svm_acc

#Comparing accuracies of models

results <- data.frame(
  Algorithm = c("Logistic Regression","KNN","Decision Tree","Random Forest","SVM"),
  Accuracy = c(
    log_acc$overall['Accuracy'],
    knn_acc$overall['Accuracy'],
    tree_acc$overall['Accuracy'],
    rf_acc$overall['Accuracy'],
    svm_acc$overall['Accuracy']
  ),
  Sensitivity = c(
    log_acc$byClass['Sensitivity'],
    knn_acc$byClass['Sensitivity'],
    tree_acc$byClass['Sensitivity'],
    rf_acc$byClass['Sensitivity'],
    svm_acc$byClass['Sensitivity']
  ),
  Specificity = c(
    log_acc$byClass['Specificity'],
    knn_acc$byClass['Specificity'],
    tree_acc$byClass['Specificity'],
    rf_acc$byClass['Specificity'],
    svm_acc$byClass['Specificity']
  )
)

print(results)



# Logistic
pred_log

# KNN
knn_pred

# Decision Tree
tree_pred

# Random Forest
rf_pred

# SVM
svm_pred

#ROC Curve

library(pROC)

prob_log <- predict(log_model, test, type="response")

roc_curve <- roc(test$target, prob_log)

plot(roc_curve)


#Ensemble Testing

pred_df <- data.frame(pred_log, knn_pred, tree_pred, rf_pred, svm_pred)

ensemble_pred <- apply(pred_df, 1, function(x) {
  names(sort(table(x), decreasing=TRUE))[1]
})

ensemble_pred <- factor(ensemble_pred, levels = c("1","0"))

confusionMatrix(ensemble_pred, test$target)

#logistic regression and RandomForest
combo <- data.frame(pred_log, rf_pred)

final <- apply(combo, 1, function(x){
  names(sort(table(x), decreasing=TRUE))[1]
})

final <- factor(final, levels = c("1","0"))

confusionMatrix(final, test$target)


# Accuracy Bar Plot
barplot(results$Accuracy,
        names.arg = results$Algorithm,
        main = "Model Accuracy Comparison",
        xlab = "Algorithms",
        ylab = "Accuracy",
        las = 2)

# Combine metrics
metrics <- rbind(results$Sensitivity, results$Specificity)

barplot(metrics,
        beside=TRUE,
        names.arg = results$Algorithm,
        legend.text = c("Sensitivity","Specificity"),
        main = "Sensitivity vs Specificity",
        las=2)

#Confusion Matrix HeatMaps

library(ggplot2)
plot_cm <- function(cm, title){
  df <- as.data.frame(cm$table)
  
  ggplot(df, aes(Prediction, Reference, fill=Freq)) +
    geom_tile() +
    geom_text(aes(label=Freq)) +
    ggtitle(title)
}

# Plot each model
plot_cm(log_acc, "Logistic Regression")
plot_cm(knn_acc, "KNN")
plot_cm(tree_acc, "Decision Tree")
plot_cm(rf_acc, "Random Forest")
plot_cm(svm_acc, "SVM")

#ROC Curve for all models

library(pROC)

# Logistic
prob_log <- predict(log_model, test, type="response")
roc_log <- roc(test$target, prob_log)

# Plot base
plot(roc_log, main="ROC Curve Comparison")

# Add others (approx using numeric predictions)
roc_knn <- roc(test$target, as.numeric(knn_pred))
lines(roc_knn)

roc_tree <- roc(test$target, as.numeric(tree_pred))
lines(roc_tree)

roc_rf <- roc(test$target, as.numeric(rf_pred))
lines(roc_rf)

roc_svm <- roc(test$target, as.numeric(svm_pred))
lines(roc_svm)



#install.packages(c("e1071", "gbm", "xgboost", "MASS", "nnet"))

library(e1071)

nb_model <- naiveBayes(target ~ ., data=train)
nb_pred <- predict(nb_model, test)

nb_acc <- mean(nb_pred == test$target)
nb_acc

library(gbm)

# Convert target to numeric
train$target_num <- as.numeric(as.character(train$target))

# Train model
gbm_model <- gbm(target_num ~ . -target, 
                 data=train, 
                 distribution="bernoulli", 
                 n.trees=100)

# Prediction
gbm_pred <- predict(gbm_model, test, n.trees=100, type="response")

# Convert back to class
gbm_pred <- ifelse(gbm_pred > 0.5, "1","0")
gbm_pred <- factor(gbm_pred, levels=c("1","0"))

# Accuracy
gbm_acc <- mean(gbm_pred == test$target)
gbm_acc


# ================================
# Final ENSEMBLE (Both for Evan & Shyam): LR + RF + NB + GBM
# ================================

# Combine predictions
ensemble_df <- data.frame(pred_log, rf_pred, nb_pred, gbm_pred)

# Majority voting
ensemble_pred <- apply(ensemble_df, 1, function(x){
  names(sort(table(x), decreasing=TRUE))[1]
})

# Convert to factor
ensemble_pred <- factor(ensemble_pred, levels = c("1","0"))

# Accuracy
ensemble_acc <- mean(ensemble_pred == test$target)

ensemble_acc

#install.packages("caret")
library(caret)
confusionMatrix(ensemble_pred, test$target)

saveRDS(train_x, "knn_train_x.rds")
saveRDS(train_y, "knn_train_y.rds")
saveRDS(scaling_params, "knn_scaling.rds")
saveRDS(tree_model, "model_tree.rds")
saveRDS(svm_model, "model_svm.rds")

getwd()
setwd("C:/Users/evan2/Desktop/Heart disease Project/HDP Website")