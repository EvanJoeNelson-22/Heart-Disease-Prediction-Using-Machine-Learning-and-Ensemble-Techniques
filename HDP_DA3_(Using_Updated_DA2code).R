#Heart Disease Prediction Using R (24MIA1029 Evan Joe Nelson G B)

# -------------------------------
# Install packages (only once)
# -------------------------------
# install.packages("ggplot2")
# install.packages("GGally")
# install.packages("e1071")
# install.packages("nnet")
# install.packages("xgboost")
# install.packages("caret")
# install.packages("pROC")
# install.packages("gbm")

# =========================================================
# STEP-1: IMPORT DATA + EDA
# =========================================================

data = read.csv(file.choose(), header = TRUE, sep = ",")

head(data)
str(data)
summary(data)

library(ggplot2)
library(GGally)

# Convert target to factor
data$target <- as.factor(data$target)

# Target Distribution
ggplot(data, aes(x = factor(target))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Heart Disease Distribution",
       x = "Target (0 = No, 1 = Yes)",
       y = "Count")

# Age vs Target
ggplot(data, aes(x = age, fill = factor(target))) +
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Age Distribution by Heart Disease",
       fill = "Target")

# Cholesterol vs Target
ggplot(data, aes(x = factor(target), y = chol)) +
  geom_boxplot(fill = "orange") +
  labs(title = "Cholesterol vs Heart Disease",
       x = "Target",
       y = "Cholesterol")

# Optional
# GGally::ggpairs(data)

# =========================================================
# STEP-2: DATA PREPROCESSING
# =========================================================

# Missing Values
colSums(is.na(data))

# Convert categorical variables
data$sex <- as.factor(data$sex)
data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)
data$ca <- as.factor(data$ca)
data$thal <- as.factor(data$thal)
data$target <- as.factor(data$target)

# Class distribution (extra improvement)
table(data$target)
prop.table(table(data$target))

str(data)
summary(data)
# =========================================================
# STEP-3: TRAIN-TEST SPLIT
# =========================================================

set.seed(123)

sample_split <- sample(c(TRUE, FALSE), nrow(data),
                       replace = TRUE, prob = c(0.7, 0.3))

train <- data[sample_split, ]
test  <- data[!sample_split, ]

# =========================================================
# FEATURE SCALING
# =========================================================

# Scale ONLY numeric columns using training data
train_num <- train[, sapply(train, is.numeric)]
test_num  <- test[, sapply(test, is.numeric)]

scaled_train <- scale(train_num)
scaled_test  <- scale(test_num,
                      center = attr(scaled_train, "scaled:center"),
                      scale  = attr(scaled_train, "scaled:scale"))

# Combine scaled + categorical
train_scaled <- cbind(as.data.frame(scaled_train),
                      train[, sapply(train, is.factor)])

test_scaled <- cbind(as.data.frame(scaled_test),
                     test[, sapply(test, is.factor)])

train_scaled$target <- train$target
test_scaled$target  <- test$target

# =========================================================
# MODEL-1: NAIVE BAYES
# =========================================================

library(e1071)

set.seed(123)

model_nb <- naiveBayes(target ~ ., data = train)

pred_nb <- predict(model_nb, test)

# Evaluation
#Confusion Matrix
table(pred_nb, test$target)

#Accuracy
mean(pred_nb == test$target)

# Split check
nrow(train) / nrow(data)
nrow(test) / nrow(data)

# =========================================================
# MODEL-2: NEURAL NETWORK
# =========================================================

library(nnet)

set.seed(123)

model_nn <- nnet(target ~ ., data = train_scaled,
                 size = 10, maxit = 500, decay = 0.01, trace = FALSE)

pred_nn <- predict(model_nn, test_scaled, type = "class")

# Evaluation
#Confusion Matrix
table(pred_nn, test_scaled$target)

#Accuracy
mean(pred_nn == test_scaled$target)

# =========================================================
# MODEL-3: XGBOOST (CLEANED)
# =========================================================

library(xgboost)

# Convert to matrix
train_x <- model.matrix(target ~ . -1, data = train)
test_x  <- model.matrix(target ~ . -1, data = test)

# Clean conversion
train_y <- as.numeric(train$target) - 1
test_y  <- as.numeric(test$target) - 1

dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest  <- xgb.DMatrix(data = test_x, label = test_y)

# Improved parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 5,
  eta = 0.05
)

set.seed(123)

model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 50
)

pred_xgb <- predict(model_xgb, dtest)
pred_xgb <- ifelse(pred_xgb > 0.5, 1, 0)

pred_xgb <- as.factor(pred_xgb)
test_y_factor <- as.factor(test_y)

# Evaluation
#Confusion Matrix
table(pred_xgb, test_y_factor) 

#Accuracy
mean(pred_xgb == test_y_factor)

# =========================================================
# PERFORMANCE COMPARISON
# =========================================================

library(caret)

pred_nb <- factor(pred_nb, levels = c(0,1))
test$target <- factor(test$target, levels = c(0,1))

cm_nb <- confusionMatrix(pred_nb, test$target)

pred_nn <- factor(pred_nn, levels = c(0,1))

cm_nn <- confusionMatrix(pred_nn, test_scaled$target)

pred_xgb <- factor(pred_xgb, levels = c(0,1))
test_y_factor <- factor(test_y, levels = c(0,1))

cm_xgb <- confusionMatrix(pred_xgb, test_y_factor)

results <- data.frame(
  Algorithm = c("Naive Bayes", "Neural Network", "XGBoost"),
  
  Accuracy = c(cm_nb$overall['Accuracy'],
               cm_nn$overall['Accuracy'],
               cm_xgb$overall['Accuracy']),
  
  Sensitivity = c(cm_nb$byClass['Sensitivity'],
                  cm_nn$byClass['Sensitivity'],
                  cm_xgb$byClass['Sensitivity']),
  
  Specificity = c(cm_nb$byClass['Specificity'],
                  cm_nn$byClass['Specificity'],
                  cm_xgb$byClass['Specificity'])
)

print(results)

# =========================================================
# ROC CURVE
# =========================================================

library(pROC)

prob_nb <- predict(model_nb, test, type = "raw")[,2]
prob_nn <- as.numeric(predict(model_nn, test_scaled, type = "raw"))
prob_xgb <- predict(model_xgb, dtest)

test_num <- as.numeric(as.character(test$target))
test_scaled_num <- as.numeric(as.character(test_scaled$target))
test_y_num <- as.numeric(as.character(test_y_factor))

roc_nb <- roc(test_num, prob_nb)
roc_nn <- roc(test_scaled_num, prob_nn)
roc_xgb <- roc(test_y_num, prob_xgb)

plot(roc_nb, col="blue", main="ROC Curve Comparison")
lines(roc_nn, col="red")
lines(roc_xgb, col="green")

legend("bottomright",
       legend=c("Naive Bayes","Neural Network","XGBoost"),
       col=c("blue","red","green"),
       lwd=2)

# =========================================================
# DA-3: ADVANCED MODELS + ENSEMBLE
# =========================================================

# -------------------------------
# Load Libraries
# -------------------------------
library(gbm)
library(adabag)

# =========================================================
# MODEL-4: GRADIENT BOOSTING (GBM)
# =========================================================

set.seed(123)

train_gb <- train
test_gb  <- test

train_gb$target <- as.numeric(as.character(train_gb$target))
test_gb$target  <- as.numeric(as.character(test_gb$target))

gb_model <- gbm(target ~ ., 
                data = train_gb,
                distribution = "bernoulli",
                n.trees = 300,
                interaction.depth = 4,
                shrinkage = 0.03,
                n.minobsinnode = 5,
                verbose = FALSE)

# Prediction
gb_prob <- predict(gb_model, test_gb, n.trees = 300, type = "response")
gb_pred <- ifelse(gb_prob > 0.5, 1, 0)
gb_pred <- factor(gb_pred, levels = c(0,1))

# Evaluation
cat("\nGradient Boosting Results:\n")
cm_gb <- confusionMatrix(gb_pred, test$target)
print(cm_gb)

# =========================================================
# MODEL-5: ADABOOST
# =========================================================

set.seed(123)

ada_model <- boosting(target ~ ., data = train, mfinal = 50)

ada_pred <- predict(ada_model, test)
ada_class <- factor(ada_pred$class, levels = c(0,1))

# Evaluation
cat("\nAdaBoost Results:\n")
cm_ada <- confusionMatrix(ada_class, test$target)
print(cm_ada)

# =========================================================
# MODEL-6: LOGISTIC REGRESSION (Shyam's Model)
# =========================================================

model_lr <- glm(target ~ ., data = train, family = "binomial")

# Probability (IMPORTANT)
prob_lr <- predict(model_lr, test, type = "response")

# Class prediction
pred_lr <- ifelse(prob_lr > 0.5, 1, 0)
pred_lr <- factor(pred_lr, levels = c(0,1))

# Evaluation
cat("\nLogistic Regression Results:\n")
cm_lr <- confusionMatrix(pred_lr, test$target)
print(cm_lr)

# =========================================================
# MODEL-7: RANDOM FOREST (Shyam's Model)
# =========================================================

library(randomForest)

set.seed(123)

model_rf <- randomForest(target ~ ., data = train, ntree = 300,mtry = 3)

# Probability (IMPORTANT)
prob_rf_all <- predict(model_rf, test, type = "prob")

# Ensure correct column
colnames(prob_rf_all)

# Assuming columns are "0" "1"
prob_rf <- prob_rf_all[,2]

# Class prediction
rf_pred <- ifelse(prob_rf > 0.5, 1, 0)
rf_pred <- factor(rf_pred, levels = c(0,1))

# Evaluation
cat("\nRandom Forest Results:\n")
cm_rf <- confusionMatrix(rf_pred, test$target)
print(cm_rf)


# =========================================================
# WEIGHTED ENSEMBLE (IMPROVED) (Not used In our project, Comparison purposes Only)
# =========================================================

# Convert predictions to numeric
# nb_num  <- as.numeric(as.character(pred_nb))
# nn_num  <- as.numeric(as.character(pred_nn))
# xgb_num <- as.numeric(as.character(pred_xgb))
# gb_num  <- as.numeric(as.character(gb_pred))
# ada_num <- as.numeric(as.character(ada_class))

# Apply weights
# ensemble_score <- (2 * nb_num) +
#  (1 * nn_num) +
#  (2 * gb_num) +
#  (1 * ada_num)

#Weak model (XGB) noise removed → better decisions 

# Total weight = 6 → majority = 3
# ensemble_pred <- ifelse(ensemble_score >= 3, 1, 0)
# ensemble_pred <- factor(ensemble_pred, levels = c(0,1))

# Evaluation
# cat("\nWeighted Ensemble Results:\n")
# confusionMatrix(ensemble_pred, test$target)


# =========================================================
# WEIGHTED PROBABILITY ENSEMBLE (My own individual ensemble, without combining) (BEST) 
# =========================================================

ensemble_prob <- (2*prob_nb +           #(Higher accuracy than Weighted Voting ensemble)
                    prob_nn + 
                    2*gb_prob + 
                    ada_pred$prob[,2]) / 6

ensemble_pred_prob <- ifelse(ensemble_prob > 0.5, 1, 0)
ensemble_pred_prob <- factor(ensemble_pred_prob, levels = c(0,1))

cat("\nWeighted Probability Ensemble Results:\n")
cm_ens <- confusionMatrix(ensemble_pred_prob, test$target)
print(cm_ens)


# =========================================================
# FINAL HYBRID ENSEMBLE (NB + GBM + RF + LR) (Combining both Shyam's and my models)
# =========================================================

ensemble_prob_hybrid <- (
  3*prob_nb + 
    2*gb_prob + 
    3*prob_rf + 
    1*prob_lr
) / 9

ensemble_pred_hybrid <- ifelse(ensemble_prob_hybrid > 0.48, 1, 0)
ensemble_pred_hybrid <- factor(ensemble_pred_hybrid, levels = c(0,1))

cat("\nHybrid Ensemble Results:\n")
cm_hybrid <- confusionMatrix(ensemble_pred_hybrid, test$target)
print(cm_hybrid)


# =========================================================
# FINAL COMPARISON TABLE (ALL of my models + Hybrid Ensemble)
# =========================================================

results_final <- data.frame(
  Model = c("Naive Bayes", "Neural Network", "XGBoost",
            "Gradient Boosting", "AdaBoost", "Ensemble", "Hybrid Ensemble"),
  
  Accuracy = c(
    cm_nb$overall['Accuracy'],
    cm_nn$overall['Accuracy'],
    cm_xgb$overall['Accuracy'],
    cm_gb$overall['Accuracy'],
    cm_ada$overall['Accuracy'],
    cm_ens$overall['Accuracy'],
    cm_hybrid$overall['Accuracy']
  ),
  
  Sensitivity = c(
    cm_nb$byClass['Sensitivity'],
    cm_nn$byClass['Sensitivity'],
    cm_xgb$byClass['Sensitivity'],
    cm_gb$byClass['Sensitivity'],
    cm_ada$byClass['Sensitivity'],
    cm_ens$byClass['Sensitivity'],
    cm_hybrid$byClass['Sensitivity']
  ),
  
  Specificity = c(
    cm_nb$byClass['Specificity'],
    cm_nn$byClass['Specificity'],
    cm_xgb$byClass['Specificity'],
    cm_gb$byClass['Specificity'],
    cm_ada$byClass['Specificity'],
    cm_ens$byClass['Specificity'],
    cm_hybrid$byClass['Specificity']
  )
)

cat("\nFINAL MODEL COMPARISON:\n")
print(results_final)

# =========================================================
# FINAL ROC CURVE (ALL of my own individual MODELS + Hybrid Ensemble)
# =========================================================

# Probabilities for new models
roc_gb <- roc(test_num, gb_prob)
roc_ada <- roc(test_num, as.numeric(ada_pred$prob[,2]))
roc_ens <- roc(test_num, ensemble_prob)
roc_lr <- roc(test_num, prob_lr)
roc_rf <- roc(test_num, prob_rf)
roc_hybrid <- roc(test_num, ensemble_prob_hybrid)

# Plot all ROC curves
plot(roc_nb, col="blue", main="Final ROC Comparison")
lines(roc_nn, col="red")
lines(roc_xgb, col="green")
lines(roc_gb, col="purple")
lines(roc_ada, col="orange")
lines(roc_ens, col="black")
lines(roc_lr, col="brown")
lines(roc_rf, col="pink")
lines(roc_hybrid, col="darkgreen", lwd=3)


legend("bottomright",
       legend=c("Naive Bayes","Neural Network","XGBoost",
                "Gradient Boosting","AdaBoost","Ensemble",
                "Logistic Regression","Random Forest","Hybrid Ensemble"),
       col=c("blue","red","green","purple","orange","black",
             "brown","pink","yellow"),
       lwd=c(2,2,2,2,2,2,2,2,3))

# AUC Values
cat("\nAUC Scores:\n")
cat("Naive Bayes:", auc(roc_nb), "\n")
cat("Neural Network:", auc(roc_nn), "\n")
cat("XGBoost:", auc(roc_xgb), "\n")
cat("Gradient Boosting:", auc(roc_gb), "\n")
cat("AdaBoost:", auc(roc_ada), "\n")
cat("Ensemble:", auc(roc_ens), "\n")
cat("Logistic Regression:", auc(roc_lr), "\n")
cat("Random Forest:", auc(roc_rf), "\n")
cat("Hybrid Ensemble:", auc(roc_hybrid), "\n")

# =========================================================
# END OF PROJECT
# =========================================================

# Save all models

saveRDS(model_lr, "model_lr.rds")
saveRDS(model_rf, "model_rf.rds")
saveRDS(model_nb, "model_nb.rds")
saveRDS(gb_model, "model_gbm.rds")
saveRDS(model_nn, "model_nn.rds")
saveRDS(model_xgb, "model_xgb.rds")
saveRDS(ada_model, "model_ada.rds")

getwd()
setwd("C:/Users/evan2/Desktop/Heart disease Project/HDP Website")
saveRDS(train, "train_template.rds")