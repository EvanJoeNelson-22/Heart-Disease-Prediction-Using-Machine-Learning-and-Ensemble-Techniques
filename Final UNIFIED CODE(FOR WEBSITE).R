# =========================================================
# FINAL UNIFIED TRAINING PIPELINE
# =========================================================

library(e1071)
library(randomForest)
library(rpart)
library(nnet)
library(xgboost)
library(gbm)
library(adabag)
library(caret)

# Load data
data <- read.csv(file.choose())

# =========================================================
# UNIFIED PREPROCESSING (CRITICAL)
# =========================================================

data$target <- factor(data$target, levels = c(0,1))

data$sex <- factor(data$sex, levels = c(0,1))
data$cp <- factor(data$cp, levels = c(0,1,2,3))
data$fbs <- factor(data$fbs, levels = c(0,1))
data$restecg <- factor(data$restecg, levels = c(0,1,2))
data$exang <- factor(data$exang, levels = c(0,1))
data$slope <- factor(data$slope, levels = c(0,1,2))
data$ca <- factor(data$ca, levels = c(0,1,2,3))
data$thal <- factor(data$thal, levels = c(0,1,2,3))

# Split
set.seed(123)
trainIndex <- createDataPartition(data$target, p=0.7, list=FALSE)
train <- data[trainIndex,]
test <- data[-trainIndex,]

# =========================================================
# MODELS (ALL FIXED)
# =========================================================

# Logistic
model_lr <- glm(target ~ ., data=train, family="binomial")

# Naive Bayes
model_nb <- naiveBayes(target ~ ., data=train)

# Random Forest
model_rf <- randomForest(target ~ ., data=train, ntree=200)

# Decision Tree
model_tree <- rpart(target ~ ., data=train)

# SVM (FIXED)
model_svm <- svm(target ~ ., data=train, probability=TRUE)

# =========================================================
# NUMERIC DATA (FOR NN, XGB, GBM)
# =========================================================

train_num <- data.frame(lapply(train, function(x) as.numeric(as.character(x))))
test_num  <- data.frame(lapply(test, function(x) as.numeric(as.character(x))))

# Neural Network (FIXED)
model_nn <- nnet(target ~ ., data=train_num, size=10, maxit=500, decay=0.01, trace=FALSE)

# =========================================================
# XGBOOST (FINAL WORKING VERSION)
# =========================================================

train_matrix <- as.matrix(train_num[,-which(names(train_num)=="target")])
test_matrix  <- as.matrix(test_num[,-which(names(test_num)=="target")])

# Convert target properly
y_train <- as.numeric(as.character(train$target))
y_test  <- as.numeric(as.character(test$target))

# Create DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = y_train)
dtest  <- xgb.DMatrix(data = test_matrix, label = y_test)

# Train model using xgb.train
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss"
)

model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

# GBM (FIXED)
model_gbm <- gbm(target ~ ., data=train_num, distribution="bernoulli", n.trees=200)

# AdaBoost
model_ada <- boosting(target ~ ., data=train)

# =========================================================
# SAVE MODELS
# =========================================================

setwd("C:/Users/evan2/Desktop/Heart disease Project/HDP Website")

saveRDS(model_lr, "model_lr.rds")
saveRDS(model_nb, "model_nb.rds")
saveRDS(model_rf, "model_rf.rds")
saveRDS(model_tree, "model_tree.rds")
saveRDS(model_svm, "model_svm.rds")
saveRDS(model_nn, "model_nn.rds")
saveRDS(model_xgb, "model_xgb.rds")
saveRDS(model_gbm, "model_gbm.rds")
saveRDS(model_ada, "model_ada.rds")

# Save template for API
saveRDS(train, "train_template.rds")