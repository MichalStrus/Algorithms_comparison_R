# Wczytanie potrzebnych bibliotek
library(rpart)
library(class)
library(neuralnet)
library(caret)
library(data.tree)
library(ggplot2)
library(rpart.plot)
source("Funkcje.R")






boston_data <- read.csv("boston.csv") #Wczytanie zbioru do regresji 



data <- read.csv("heart.csv") # Wczytanie zbioru do klasyfikacji binarnej i ustawienie factorów
#data <- data[1:500,]
data$Sex <- as.factor(data$Sex)
data$ChestPainType <- as.factor(data$ChestPainType)
data$RestingECG <- as.factor(data$RestingECG)
data$ExerciseAngina <- as.factor(data$ExerciseAngina)
data$ST_Slope <- as.factor(data$ST_Slope)
data$HeartDisease <- as.factor(data$HeartDisease)


data_students <- read.csv("students.csv", sep = ";") #Wczytanie zbioru do klasyfikacji wieloklasowej
data_students <- data_students[1:2000,]
data_students$Target <- factor(data_students$Target, levels = c("Dropout", "Enrolled", "Graduate"))

#### Crosswalidacja

##KNN
parTune_example <- list(k = c(3, 5, 7))
results <- CrossValidTune(dane = data[1:200,], kFold = 5, parTune = parTune_example, seed = 123, algorithm = "KNN")
print(results)#KNN, binarna

parTune_example <- data.frame(k = c(3, 5, 7))
results <- CrossValidTune(dane = data_students[1:200,], kFold = 5, parTune = parTune_example, seed = 123, algorithm = "KNN")# KNN, wieloklasowa
print(results)

parTune_example <- data.frame(k = c(3, 5, 7))
results <- CrossValidTune(dane = boston_data[1:200,], kFold = 5, parTune = parTune_example, seed = 123, algorithm = "KNN")#KNN, regresja
print(results)

## Drzewa decyzyjne
parTune <- list(max_depth = c(7, 10), min_samples = c(10,15), pruning_method= c("prune"), criterion = c("Entropy"), cf = c(0.1))  # Lista parametrów dla drzewa
results <- CrossValidTune(data, kFold = 5, parTune = parTune, seed = 123, algorithm = "DecisionTree")#Drzewo decyzyjne, binarna
print(results)


parTune <- list(max_depth = c(10, 15), min_samples = c(10, 15), pruning_method= c("prune"), criterion = c("SS"), cf = c(0.1, 0.2))
results <- CrossValidTune(dane = boston_data, kFold = 5, parTune = parTune, seed = 123, algorithm = "DecisionTree")#Drzewo decyzyjne, regresja
print(results)


parTune <- list(max_depth = c(10,15), min_samples = c(10,15), pruning_method= c("prune"), criterion = c("Entropy"), cf = c(0.1, 0.2))
results <- CrossValidTune(dane = data_students, kFold = 5, parTune = parTune, seed = 123, algorithm = "DecisionTree")#Drzewo decyzyjne, wieloklasowa
print(results)


## Sieci neuronowe
dh <- data$HeartDisease
onehot_data <- do.call(cbind, lapply(data[, sapply(data, is.factor)], function(x) model.matrix(~ x - 1)))
onehot_data[,-c(ncol(onehot_data)-1,ncol(onehot_data))]
numeric_data <- data[, sapply(data, is.numeric)]
data_nn <- cbind(numeric_data,onehot_data,dh)


parTune <- list(hidden_units = list(c(32,16)), activation_hidden = c("sigmoid", "tanh"),lr=c(0.01), iter=c(1000), seed=c(123), activation_output=c("sigmoid"))
results <- CrossValidTune(dane = data_nn, kFold = 5, parTune = parTune, seed = 123, algorithm = "NN")# sieci neuronowe, binarna
print(results)


parTune <- list(hidden_units = list(c(64,32), c(32,16)), activation_hidden = c("sigmoid", "tanh"),lr=c(0.01), iter=c(1000), seed=c(123), activation_output=c("linear"))
results <- CrossValidTune(dane = boston_data, kFold = 5, parTune = parTune, seed = 123, algorithm = "NN")# regresja, działa
results


parTune <- list(hidden_units = list(c(32,16), c(10,5)), activation_hidden = c("sigmoid", "tanh"),lr=c(0.01), iter=c(1000), seed=c(123), activation_output=c("softmax"))
results <- CrossValidTune(dane = data_students, kFold = 5, parTune = parTune, seed = 123, algorithm = "NN")# działa wieloklasowa
results





#### Regresja
set.seed(123)
trainIndex <- createDataPartition(boston_data$MEDV, p = 0.7, list = FALSE)
trainData <- boston_data[trainIndex, ]
testData <- boston_data[-trainIndex, ]

# Mój algorytm
tree <- decisionTree(colnames(boston_data)[ncol(boston_data)], colnames(boston_data)[-ncol(boston_data)], trainData, "SS", 15, 1, "prune", 0.1)
print(tree)


predictions <- predictTree(tree, testData)
ModelEval(testData$MEDV, predictions)


### 1. DRZEWA DECYZYJNE ###
tree_model <- rpart(MEDV ~ ., data = trainData, method = "anova")
printcp(tree_model)
plotcp(tree_model)

grid <- expand.grid(cp = seq(0.001, 0.2, by = 0.005))
ctrl <- trainControl(method = "cv", number = 5)


tuned_tree <- train(
  MEDV ~ ., data = trainData, method = "rpart",
  tuneGrid = grid, trControl = ctrl
)

best_params <- tuned_tree$bestTune
print(best_params)


best_tree <- rpart(MEDV ~ ., data = trainData, method = "anova",
                   cp = best_params$cp)

ModelEval(testData$MEDV,predict(best_tree, testData))



### 2. K-NEAREST NEIGHBORS (KNN) ###
set.seed(123)
boston_data_knn <- boston_data[1:200,]
trainIndex <- createDataPartition(boston_data_knn$MEDV, p = 0.7, list = FALSE)
trainData <- boston_data_knn[trainIndex, ]
testData <- boston_data_knn[-trainIndex, ]

# Mój algorytm

model <- KNNtrain(trainData[,-ncol(trainData)], trainData[, ncol(trainData)], k = 3, 0, 1)
y_train_pred <- KNNpred(model, testData[,-ncol(testData)])

print(y_train_pred)
ModelEval(testData$MEDV, y_train_pred)


preProc <- preProcess(trainData[, -which(names(trainData) == "MEDV")], method = "range")
trainData_scaled <- predict(preProc, trainData)
testData_scaled <- predict(preProc, testData)

grid_knn <- expand.grid(k = seq(1, 20, by = 1))
ctrl_knn <- trainControl(method = "cv", number = 5)

tuned_knn <- train(
  MEDV ~ ., data = trainData_scaled, method = "knn",
  tuneGrid = grid_knn, trControl = ctrl_knn
)

# Sprawdzenie najlepszego parametru k
best_knn_params <- tuned_knn$bestTune
print(best_knn_params)

best_knn_model <- knnreg(MEDV ~ ., data = trainData_scaled, k = best_knn_params$k)
knn_predictions <- predict(best_knn_model, newdata = testData_scaled)

ModelEval(testData$MEDV, knn_predictions)




###########################
### 3. SIECI NEURONOWE ###
set.seed(123)
trainIndex <- createDataPartition(boston_data$MEDV, p = 0.7, list = FALSE)
trainData <- boston_data[trainIndex, ]
testData <- boston_data[-trainIndex, ]

#Mój algorytm
parTune <- list(Yname = colnames(trainData)[ncol(trainData)], 
                                 Xnames = colnames(trainData)[-ncol(trainData)], 
                                 data = trainData, 
                                 h = c(128, 64),
                                 lr = 0.01, 
                                 iter = 10000, 
                                 seed = 123)
                 
result <- trainNN(parTune$Yname, parTune$Xnames, parTune$data, 
                  parTune$h, parTune$lr, parTune$iter, parTune$seed, activation_hidden = "sigmoid")
                 
X_test <- as.matrix(testData[,-ncol(testData)])
X_test <- scale(X_test)
                 
predictions_result <- predictNN(X_test, result$weights, result$biases,
                                h = c(128, 64), activation_output = "linear", min_Y = result$min_Y, max_Y = result$max_Y)

ModelEval(testData$MEDV,predictions_result)


# Normalizacja Min-Max
preProc <- preProcess(trainData[, -which(names(trainData) == "MEDV")], method = "range")
trainData_scaled <- predict(preProc, trainData)
testData_scaled <- predict(preProc, testData)

min_MEDV <- min(trainData$MEDV)
max_MEDV <- max(trainData$MEDV)

trainData_scaled$MEDV <- (trainData$MEDV - min_MEDV) / (max_MEDV - min_MEDV)
testData_scaled$MEDV <- (testData$MEDV - min_MEDV) / (max_MEDV - min_MEDV)

# Trenowanie modelu sieci neuronowej (2 warstwy ukryte)
nn_model <- neuralnet(
  MEDV ~ ., data = trainData_scaled,
  hidden = c(32, 16),
  stepmax = 1e6,
  linear.output = TRUE
)

# Wizualizacja sieci neuronowej
#plot(nn_model)

# Predykcja
nn_predictions <- compute(nn_model, testData_scaled[, -which(names(testData_scaled) == "MEDV")])$net.result
nn_predictions_original <- nn_predictions * (max_MEDV - min_MEDV) + min_MEDV
# Ocena modelu
ModelEval(testData$MEDV, nn_predictions_original)




#### Klasyfikacja binarna
##############################

## Drzewa decyzyjne
set.seed(123)

trainIndex <- createDataPartition(data$HeartDisease, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Mój algorytm
tree <- decisionTree(colnames(data)[ncol(data)], colnames(data)[-ncol(data)], trainData, "Entropy", 10, 2, "prune", 0.1)
print(tree)

predictions <- predictTree(tree, testData)
eval <- ModelEval(testData$HeartDisease, predictions[[1]])
print(eval)

conf_matrix <- eval$ConfMat

cm <- as.data.frame(as.table(conf_matrix))

colnames(cm) <- c("Actual", "Predicted", "Freq")


# Create the plot
ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix my Tree", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))

### Wbudowany model drzewa decyzyjnego

tree_model <- rpart(HeartDisease ~ ., data = trainData, method = "class")

printcp(tree_model)
plotcp(tree_model)

grid <- expand.grid(cp = seq(0.001, 0.2, by = 0.005))

ctrl <- trainControl(method = "cv", number = 5)

tuned_tree <- train(
  HeartDisease ~ ., data = trainData, method = "rpart",
  tuneGrid = grid, trControl = ctrl
)

best_params <- tuned_tree$bestTune
print(best_params)

best_tree <- rpart(HeartDisease ~ ., data = trainData, method = "class",
                   cp = best_params$cp)

eval <- ModelEval(testData$HeartDisease,predict(best_tree, testData, type = "prob")[,1])
print(eval)

conf_matrix <- eval$ConfMat

cm <- as.data.frame(as.table(conf_matrix))

colnames(cm) <- c("Actual", "Predicted", "Freq")


# Create the plot
ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix R-Tree", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))

#####################################
### 2. K-NEAREST NEIGHBORS (KNN) ###
set.seed(123)
data_knn <- data[1:200,]
trainIndex <- createDataPartition(data_knn$HeartDisease, p = 0.7, list = FALSE)
trainData <- data_knn[trainIndex, ]
testData <- data_knn[-trainIndex, ]

#Mój algorytm
model <- KNNtrain(trainData[,-ncol(trainData)], trainData[, ncol(trainData)], k = 3, 0, 1)
y_train_pred <- KNNpred(model, testData[,-ncol(testData)])

print(y_train_pred)
eval <- ModelEval(testData$HeartDisease, y_train_pred[,1])
print(eval)

conf_matrix <- eval$ConfMat

cm <- as.data.frame(as.table(conf_matrix))

colnames(cm) <- c("Actual", "Predicted", "Freq")

ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix My-KNN", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))


preProc <- preProcess(trainData[, -which(names(trainData) == "HeartDisease")], method = "range")
trainData_scaled <- predict(preProc, trainData)
testData_scaled <- predict(preProc, testData)

trainData_scaled$HeartDisease <- factor(trainData_scaled$HeartDisease, 
                                      levels = c(0, 1), 
                                      labels = c("No", "Yes"))

testData_scaled$HeartDisease <- factor(testData_scaled$HeartDisease, 
                                    levels = c(0, 1), 
                                    labels = c("No", "Yes"))

grid_knn <- expand.grid(k = seq(1, 20, by = 1))
ctrl_knn <- trainControl(method = "cv", number = 5, classProbs = TRUE)

tuned_knn <- train(
  HeartDisease ~ ., data = trainData_scaled, method = "knn",
  tuneGrid = grid_knn, trControl = ctrl_knn
)


best_knn_params <- tuned_knn$bestTune
print(best_knn_params)

knn_predictions <- predict(tuned_knn, newdata = testData_scaled, type="prob")
colnames(knn_predictions) <- c("0", "1")
eval <- ModelEval(testData$HeartDisease, knn_predictions[,1])
print(eval)

conf_matrix <- eval$ConfMat
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix R-KNN", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))

###############################

### 3. SIECI NEURONOWE ###
# Normalizacja danych
set.seed(123)

trainIndex <- createDataPartition(data_nn$dh, p = 0.7, list = FALSE)
trainData <- data_nn[trainIndex, ]
testData <- data_nn[-trainIndex, ]

#### Mój algorytm
parTune <- list(Yname = colnames(trainData)[ncol(trainData)], 
                Xnames = colnames(trainData)[-ncol(trainData)], 
                data = trainData, 
                h = c(32, 16),
                lr = 0.01, 
                iter = 10000, 
                seed = 123)

result <- trainNN(parTune$Yname, parTune$Xnames, parTune$data, 
                  parTune$h, parTune$lr, parTune$iter, parTune$seed, activation_hidden = "sigmoid")

X_test <- as.matrix(testData[,-ncol(testData)])
X_test <- scale(X_test)

predictions_result <- predictNN(X_test, result$weights, result$biases,
                                h = c(32, 16), activation_output = "sigmoid", min_Y = result$min_Y, max_Y = result$max_Y)

eval <- ModelEval(testData$dh,1-predictions_result)
print(eval)

conf_matrix <- eval$ConfMat
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix My Neural Network", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))




# Normalizacja Min-Max
preProc <- preProcess(trainData[, -which(names(trainData) == "dh")], method = "range")
trainData_scaled <- predict(preProc, trainData)
testData_scaled <- predict(preProc, testData)

trainData_scaled$dh <- factor(trainData_scaled$dh, 
                                        levels = c(0, 1), 
                                        labels = c("No", "Yes"))

testData_scaled$dh <- factor(testData_scaled$dh, 
                                       levels = c(0, 1), 
                                       labels = c("No", "Yes"))

nn_model <- neuralnet(
  dh ~ ., data = trainData_scaled,
  hidden = c(32, 16),
  stepmax = 1e6,
  linear.output = FALSE
)

#plot(nn_model)

nn_predictions <- compute(nn_model, testData_scaled[, -which(names(testData_scaled) == "dh")])$net.result

# Ocena modelu
eval <- ModelEval(testData$dh, nn_predictions[,1])
print(eval)

conf_matrix <- eval$ConfMat
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix R build-Neural Network", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))

##### Klasyfikacja wieloklasowa

### 1. DRZEWA DECYZYJNE ###
set.seed(123)
trainIndex <- createDataPartition(data_students$Target, p = 0.7, list = FALSE)
trainData <- data_students[trainIndex, ]
testData <- data_students[-trainIndex, ]

# Mój algorytm

tree <- decisionTree(colnames(data_students)[ncol(data_students)], colnames(data_students)[-ncol(data_students)], trainData, "Entropy", 15, 5, "prune", 0.1)
print(tree)

predictions <- predictTree(tree, testData)
eval <- ModelEval(testData$Target, predictions)
print(eval)

conf_matrix <- eval$Matrix
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix My Decision Tree multiclass", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))




#### Wbudowany algorytm
tree_model <- rpart(Target ~ ., data = trainData, method = "class")
printcp(tree_model)
plotcp(tree_model)

grid <- expand.grid(cp = seq(0.001, 0.2, by = 0.005))

ctrl <- trainControl(method = "cv", number = 5)

tuned_tree <- train(
  Target ~ ., data = trainData, method = "rpart",
  tuneGrid = grid, trControl = ctrl
)

best_params <- tuned_tree$bestTune
print(best_params)

best_tree <- rpart(Target ~ ., data = trainData, method = "class",
                   cp = best_params$cp)

tree_prob <- as.data.frame(predict(best_tree, testData, type = "prob"))
predicted_classes <- colnames(tree_prob)[max.col(tree_prob, ties.method = "first")]
tree_prob_with_class <- cbind(tree_prob, PredictedClass = factor(predicted_classes, levels = colnames(tree_prob)))


eval <- ModelEval(testData$Target, tree_prob_with_class)
print(eval)

conf_matrix <- eval$Matrix
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix R build Decision Tree multiclass", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))


###K-NEAREST NEIGHBORS (KNN) ###
set.seed(123)
data_students_knn <- data_students[1:200,]
trainIndex <- createDataPartition(data_students_knn$Target, p = 0.7, list = FALSE)
trainData <- data_students_knn[trainIndex, ]
testData <- data_students_knn[-trainIndex, ]

#### Mój model
model <- KNNtrain(trainData[,-ncol(trainData)], trainData[, ncol(trainData)], k = 3, 0, 1)
y_train_pred <- KNNpred(model, testData[,-ncol(testData)])

print(y_train_pred)
eval <- ModelEval(testData$Target, y_train_pred)
print(eval)

conf_matrix <- eval$Matrix
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix My KNN multiclass", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))


preProc <- preProcess(trainData[, -which(names(trainData) == "Target")], method = "range")
trainData_scaled <- predict(preProc, trainData)
testData_scaled <- predict(preProc, testData)


grid_knn <- expand.grid(k = seq(1, 20, by = 1))

ctrl_knn <- trainControl(method = "cv", number = 5, classProbs = TRUE)

tuned_knn <- train(
  Target ~ ., data = trainData_scaled, method = "knn",
  tuneGrid = grid_knn, trControl = ctrl_knn
)

best_knn_params <- tuned_knn$bestTune
print(best_knn_params)

knn_probs <- predict(tuned_knn, newdata = testData_scaled, type = "prob")
predicted_classes <- colnames(knn_probs)[max.col(knn_probs, ties.method = "first")]
knn_probs <- cbind(knn_probs, PredictedClass = factor(predicted_classes, levels = colnames(knn_probs)))

eval <- ModelEval(testData$Target, knn_probs)
print(eval)

conf_matrix <- eval$Matrix
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix R build KNN multiclass", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))

### 3. SIECI NEURONOWE ###
# Normalizacja danych


set.seed(123)
trainIndex <- createDataPartition(data_students$Target, p = 0.7, list = FALSE)
trainData <- data_students[trainIndex, ]
testData <- data_students[-trainIndex, ]

#Mój algorytm

parTune <- list(Yname = colnames(trainData)[ncol(trainData)], 
                Xnames = colnames(trainData)[-ncol(trainData)], 
                data = trainData, 
                h = c(32, 16),
                lr = 0.01, 
                iter = 10000, 
                seed = 123)

result <- trainNN(parTune$Yname, parTune$Xnames, parTune$data, 
                  parTune$h, parTune$lr, parTune$iter, parTune$seed, activation_hidden = "sigmoid")

X_test <- as.matrix(testData[,-ncol(testData)])
X_test <- scale(X_test)

predictions_result <- predictNN(X_test, result$weights, result$biases,
                                h = c(32, 16), activation_output = "softmax", min_Y = result$min_Y, max_Y = result$max_Y, Y_levels=levels(testData[,ncol(testData)]))

eval <- ModelEval(testData$Target,predictions_result)
print(eval)

conf_matrix <- eval$Matrix
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix My Neural Network multiclass", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))


# Normalizacja Min-Max
preProc <- preProcess(trainData[, -which(names(trainData) == "Target")], method = "range")
trainData_scaled <- predict(preProc, trainData)
testData_scaled <- predict(preProc, testData)

# Convert target variable to factor with multiple levels
trainData_scaled$Target <- factor(trainData_scaled$Target)
testData_scaled$Target <- factor(testData_scaled$Target)


nn_model <- neuralnet(
  Target ~ ., data = trainData_scaled,
  hidden = c(64, 32),
  stepmax = 1e6,
  linear.output = FALSE,
  lifesign = "minimal"
)


#plot(nn_model)

# Predykcja (prawdopodobieństwa dla każdej klasy)
nn_predictions_prob <- compute(nn_model, testData_scaled[, -which(names(testData_scaled) == "Target")])$net.result
colnames(nn_predictions_prob) <- levels(trainData$Target)

nn_class_predictions <- apply(nn_predictions_prob, 1, function(x) colnames(nn_predictions_prob)[which.max(x)])


nn_class_predictions <- factor(nn_class_predictions, levels = levels(testData_scaled$Target))
nn_preds <- as.data.frame(nn_predictions_prob)
nn_preds$Predicted_class <- nn_class_predictions
# Ocena modelu (np. dokładność)
eval <- ModelEval(testData$Target, nn_preds)
print(eval)

conf_matrix <- eval$Matrix
cm <- as.data.frame(as.table(conf_matrix))
colnames(cm) <- c("Actual", "Predicted", "Freq")


ggplot(cm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue", na.value = "gray") +
  theme_minimal() +
  labs(title = "Confusion Matrix R build Neural Network multiclass", x = "Predicted", y = "Actual") +
  theme(axis.text = element_text(size = 12))













