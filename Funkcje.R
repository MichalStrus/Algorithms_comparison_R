
library(data.tree)
# Funkcja do normalizacji wartości
MinMax <- function(x, newMin = 0, newMax = 1) {
  Min <- min(x)
  Max <- max(x)
  if(Min != Max){
    part1 <- (x - Min) / (Max - Min)
    part2 <- (newMax - newMin)
    part3 <- newMin
    result <- part1 * part2 + part3
  }
  else{
    result <- x
  }
  attr(result, "MinMaxOrg") <- c(min = Min, max = Max)
  return(result)
}

# KNN training function
KNNtrain <- function(X, y_target, k, XminNew, XmaxNew) {
  if (anyNA(X) || anyNA(y_target)) {
    stop("Missing data")
  }
  if (k <= 0) {
    stop("Invalid k")
  }
  if (is.data.frame(X) == 0 && is.matrix(X) == 0) {
    stop("Invalid data format")
  }
  
  X_norm <- X
  for (col in colnames(X)) {
    if (is.factor(X[[col]]) == FALSE) {
      X_norm[[col]] <- MinMax(X[[col]], newMax = XmaxNew, newMin = XminNew)
    } else {
      X_norm[[col]] <- X[[col]]
    }
    if (is.numeric(X_norm[[col]])) {
      min_val <- min(X_norm[[col]], na.rm = TRUE)
      max_val <- max(X_norm[[col]], na.rm = TRUE)
      attr(X_norm[[col]], "min") <- min_val
      attr(X_norm[[col]], "max") <- max_val
    } else {
      if (is.ordered(X_norm[[col]])) {
        min_val <- min(as.numeric(X_norm[[col]]), na.rm = TRUE)
        max_val <- max(as.numeric(X_norm[[col]]), na.rm = TRUE)
        attr(X_norm[[col]], "min") <- min_val
        attr(X_norm[[col]], "max") <- max_val
      }
    }
  }
  result <- list(X = X_norm, y = y_target, k = k)
  return(result)
}

# Gower distance function
Gower_dist <- function(x, y, df) {
  distance <- numeric(ncol(x))
  for (i in 1:ncol(x)) {
    if (is.factor(x[[i]])) {
      if (is.ordered(x[[i]])) {
        z_x <- (as.numeric(x[[i]]) - 1) / (as.numeric(attributes(df[[i]])$max) - 1)
        z_y <- (as.numeric(y[[i]]) - 1) / (as.numeric(attributes(df[[i]])$max) - 1)
        s <- abs(z_x - z_y)
        distance[i] <- s
      } else {
        s <- as.numeric(x[[i]] != y[[i]])
        distance[i] <- s
      }
    } else {
      s <- abs(x[[i]] - y[[i]]) / (attributes(df[[i]])$max - attributes(df[[i]])$min)
      distance[i] <- s
    }
  }
  return(mean(distance))
}

# Euclidean distance function
Euc_dist <- function(x, y) {
  return(sqrt(sum((x - y)^2)))
}

# KNN prediction function
KNNpred <- function(KNNmodel, X) {
  if (anyNA(X) || anyNA(KNNmodel$y)) {
    stop("Missing data")
  }
  if (any(colnames(X) != colnames(KNNmodel$X))) {
    stop("Missing required columns")
  }
  
  X_norm <- X
  for (col in colnames(X)) {
    if (is.factor(X[[col]]) == FALSE) {
      X_norm[[col]] <- MinMax(X[[col]], newMax = attributes(KNNmodel$X[[col]])$max, newMin = attributes(KNNmodel$X[[col]])$min)
    } else {
      X_norm[[col]] <- X[[col]]
    }
    if (is.numeric(X_norm[[col]])) {
      min_val <- min(X_norm[[col]], na.rm = TRUE)
      max_val <- max(X_norm[[col]], na.rm = TRUE)
      attr(X_norm[[col]], "min") <- min_val
      attr(X_norm[[col]], "max") <- max_val
    } else {
      if (is.ordered(X_norm[[col]])) {
        min_val <- min(as.numeric(X_norm[[col]]), na.rm = TRUE)
        max_val <- max(as.numeric(X_norm[[col]]), na.rm = TRUE)
        attr(X_norm[[col]], "min") <- min_val
        attr(X_norm[[col]], "max") <- max_val
      }
    }
  }
  
  distance_matrix <- matrix(nrow = nrow(X), ncol = nrow(KNNmodel$X))
  for (i in 1:nrow(X)) {
    for (j in 1:nrow(KNNmodel$X)) {
      if (all(sapply(df, is.numeric))) {
        distance_matrix[i, j] <- Euc_dist(X_norm[i,], KNNmodel$X[j,])
      } else {
        distance_matrix[i, j] <- Gower_dist(X_norm[i,], KNNmodel$X[j,], KNNmodel$X)
      }
    }
  }
  
  distance_matrix <- data.frame(distance_matrix)
  index_df <- distance_matrix
  for (row in 1:nrow(distance_matrix)) {
    index_df[row, ] <- rank(distance_matrix[row,], ties.method = "first")
  }
  
  k_neighbors <- which(index_df <= KNNmodel$k, arr.ind = T)
  
  if (is.numeric(KNNmodel$y)) {
    y_pred <- numeric(nrow(X))
    for (i in 1:nrow(k_neighbors)) {
      row <- k_neighbors[i, "row"]
      col <- k_neighbors[i, "col"]
      s <- y_pred[row]
      
      y_pred[row] <- s + KNNmodel$y[col]
    }
    y_pred <- y_pred / KNNmodel$k
  } else {
    y_pred <- data.frame(matrix(0, nrow = nrow(X), ncol = nlevels(KNNmodel$y)))
    colnames(y_pred) <- levels(KNNmodel$y)
    
    for (i in 1:nrow(k_neighbors)) {
      row <- k_neighbors[i, "row"]
      col <- k_neighbors[i, "col"]
      class_ind <- which(colnames(y_pred) == KNNmodel$y[col])
      y_pred[row, class_ind] <- y_pred[row, class_ind] + 1
      if (i == nrow(k_neighbors)) {
        y_hatK <- character(nrow(y_pred))
        for (j in 1:nrow(y_pred)) {
          y_hatK[j] <- colnames(y_pred)[which.max(y_pred[j,])]
        }
        y_pred <- y_pred/(KNNmodel$k)
        y_pred$Predicted <- as.factor(y_hatK)
      }
    }
  }
  
  return(y_pred)
}

# # Załaduj dane z pliku
# boston_data <- read.csv("boston.csv")
# 
# train_indices <- sample(1:nrow(boston_data), size = 0.8 * nrow(boston_data))
# 
# # Tworzymy zbiór uczący i walidacyjny
# trainData <- boston_data[train_indices, ]
# validationData <- boston_data[-train_indices, ]
# 
# # Oddzielamy cechy i cel dla zbiorów uczących i walidacyjnych
# trainFeatures <- trainData[, -ncol(trainData)]  # Cechy dla zbioru uczącego
# trainTarget <- trainData[, ncol(trainData)]    # Cel dla zbioru uczącego
# 
# validationFeatures <- validationData[, -ncol(validationData)]  # Cechy dla zbioru walidacyjnego
# validationTarget <- validationData[, ncol(validationData)] 
# 
# # Parametry KNN
# neighbors <- 5
# minNew <- 0  # minimalna wartość dla normalizacji
# maxNew <- 1  # maksymalna wartość dla normalizacji
# 
# # Trenowanie modelu
# knn_model <- KNNtrain(trainFeatures, trainTarget, neighbors, minNew, maxNew)
# # Predykcja
# predictions <- KNNpred(knn_model, trainFeatures)
# 
# # Wyświetlenie wyników
# print(predictions)
# ModelOcena(trainTarget, predictions)
# 
# 
# 
# #### Klasyfikacja binarna
# data <- read.csv("heart.csv")
# #data <- data[1:500,]
# data$Sex <- as.factor(data$Sex)
# data$ChestPainType <- as.factor(data$ChestPainType)
# data$RestingECG <- as.factor(data$RestingECG)
# data$ExerciseAngina <- as.factor(data$ExerciseAngina)
# data$ST_Slope <- as.factor(data$ST_Slope)
# data$HeartDisease <- as.factor(data$HeartDisease)
# dh <- data$HeartDisease
# 
# onehot_data <- do.call(cbind, lapply(data[, sapply(data, is.factor)], function(x) model.matrix(~ x - 1)))
# onehot_data[,-c(ncol(onehot_data)-1,ncol(onehot_data))]
# numeric_data <- data[, sapply(data, is.numeric)]
# data <- cbind(numeric_data,onehot_data,dh)
# 
# 
# train_indices <- sample(1:nrow(data), size = 0.8 * nrow(data))
# 
# # Tworzymy zbiór uczący i walidacyjny
# trainData <- data[train_indices, ]
# validationData <- data[-train_indices, ]
# 
# # Oddzielamy cechy i cel dla zbiorów uczących i walidacyjnych
# trainFeatures <- trainData[, -ncol(trainData)]  # Cechy dla zbioru uczącego
# trainTarget <- trainData[, ncol(trainData)]    # Cel dla zbioru uczącego
# 
# validationFeatures <- validationData[, -ncol(validationData)]  # Cechy dla zbioru walidacyjnego
# validationTarget <- validationData[, ncol(validationData)] 
# 
# knn_model <- KNNtrain(trainFeatures, trainTarget, 5, 0, 1)
# # Predykcja
# predictions <- KNNpred(knn_model, validationFeatures)
# 
# # Wyświetlenie wyników
# print(predictions)
# ModelEval(validationTarget, predictions[,"0"])
# 
# 
# 
# 
# #####Klasyfikacja wieloklasowa
# 
# data_students <- read.csv("students.csv", sep = ";")
# 
# #data_students <- data_students[-c(1:17)]
# data_students <- data_students[1:2000,]
# 
# data_students$Target <- factor(data_students$Target, levels = c("Dropout", "Enrolled", "Graduate"))
# train_indices <- sample(1:nrow(data_students), size = 0.7 * nrow(data_students))
# 
# 
# 
# # Tworzymy zbiór uczący i walidacyjny
# trainData <- data_students[train_indices, ]
# validationData <- data_students[-train_indices, ]
# 
# 
# # Oddzielamy cechy i cel dla zbiorów uczących i walidacyjnych
# trainFeatures <- trainData[, -ncol(trainData)]  # Cechy dla zbioru uczącego
# trainTarget <- trainData[, ncol(trainData)]    # Cel dla zbioru uczącego
# 
# validationFeatures <- validationData[, -ncol(validationData)]  # Cechy dla zbioru walidacyjnego
# validationTarget <- validationData[, ncol(validationData)] 
# 
# knn_model <- KNNtrain(trainFeatures, trainTarget, 7, 0, 1)
# # Predykcja
# predictions <- KNNpred(knn_model, validationFeatures)
# 
# # Wyświetlenie wyników
# 
# ModelEval(validationTarget, predictions)
# typeof(predictions)

#MultiAUC(validationTarget, predictions[,1:3])
#sum(predictions$Predicted=="Dropout" & validationTarget == "Dropout")


CrossValidTune <- function(dane, kFold, parTune, seed, algorithm= "KNN") {
  set.seed(seed)
  
  # Tworzymy listę indeksów dla k-fold cross-validation
  n <- nrow(dane)
  indices <- lapply(1:kFold, function(i) {
    sample(1:n, size = n/kFold, replace = FALSE)
  })
  
  if(algorithm == "NN") {
    param_grid <- expand.grid(parTune[names(parTune) != "hidden_units"], stringsAsFactors = FALSE)
    if ("hidden_units" %in% names(parTune)) {
      hidden_units_values <- parTune$hidden_units
      param_grid <- param_grid[rep(seq_len(nrow(param_grid)), each = length(hidden_units_values)), ]
      param_grid$hidden_units <- rep(hidden_units_values, times = nrow(param_grid) / length(hidden_units_values))
    }
    
    rownames(param_grid) <- NULL
  }
  else{
    param_grid <- expand.grid(parTune)
  }
  
  results_m <- list()
  results <- data.frame()  
  for (i in 1:nrow(param_grid)) {  
    # Wybieramy zestaw parametrów
    if (algorithm=="KNN"){
      k_value <- param_grid$k[i]
      print(k_value)
    }
    else if (algorithm == "DecisionTree") {
      
      max_depth <- param_grid$max_depth[i]
      min_samples <- param_grid$min_samples[i]
      pruning_method <- param_grid$pruning_method[i]
      cf <- param_grid$cf[i]
      criterion <- param_grid$criterion[i]
      print(paste("Min samples:", min_samples, "max_depth", max_depth, "Pruning_method", pruning_method, "CF:", cf, "Criterion", criterion))
    }
    else if (algorithm == "NN") {
      # Parametry dla sieci neuronowej
      h <- param_grid$h[[i]]  # Liczba neuronów w warstwie ukrytej
      lr <- param_grid$lr[i]  # Współczynnik uczenia
      iter <- param_grid$iter[i]  # Liczba iteracji
      activation_hidden <- param_grid$activation_hidden[i]  # Funkcja aktywacji warstwy ukrytej
      activation_output <- param_grid$activation_output[i]
      #print(paste("Hidden units:", h, "Learning rate:", lr, "Iterations:", iter, "Activation:", activation_hidden, "Activation output:", activation_output))
      print(h)
    }
    
    
    # Pętla k-fold
    for (j in 1:kFold) {
      print(paste("Wyliczanie folda numer:", j, "Kombinacja:", i))
      train_indices <- unlist(indices[-j])
      test_indices <- indices[[j]]
      
      train_data <- dane[train_indices, ]
      test_data <- dane[test_indices, ]
      
      if (is.numeric(train_data[, ncol(train_data)])) {
        y_train <- (train_data[, ncol(train_data)])
        y_test <- (test_data[, ncol(test_data)])
      }
      # Wyciąganie zmiennej zależnej (y) i zmiennych niezależnych (X)
      else {
        y_train <- as.factor(train_data[, ncol(train_data)])
        y_test <- as.factor(test_data[, ncol(test_data)])
      }
      X_train <- train_data[, -ncol(train_data)]
      X_test <- test_data[, -ncol(test_data)]
      #print(colnames(X_train)) # Sprawdź, czy są Inf w X_train
      #print((y_train))
      
      if (algorithm == "KNN"){
        
        model <- KNNtrain(X_train, y_train, k = k_value, 0, 1)  # Przekazujemy pojedynczą wartość k
        
        # Predykcja na zbiorze treningowym
        y_train_pred <- KNNpred(model, X_train)
        
        # Predykcja na zbiorze walidacyjnym
        y_test_pred <- KNNpred(model, X_test)
      }
      else if(algorithm == "DecisionTree") {
        #criterion <- "Gini"
        #pruning_method <- "none"
        #cf <- 0.01
        
        # Tworzymy model drzewa decyzyjnego
        tree <- decisionTree(colnames(train_data)[ncol(train_data)], colnames(train_data)[-ncol(train_data)], 
                             train_data, criterion, max_depth, min_samples, pruning_method, cf)
        
        #print(tree)
        # Predykcja na zbiorze treningowym
        y_train_pred <- predictTree(tree, train_data[-ncol(train_data)])
        
        # Predykcja na zbiorze walidacyjnym
        y_test_pred <- predictTree(tree, test_data[-ncol(test_data)])
      }
      else if (algorithm == "NN") {
        # Trenujemy sieć neuronową
        nn_model <- trainNN(Yname = colnames(train_data)[ncol(train_data)], 
                            Xnames = colnames(train_data)[-ncol(train_data)], 
                            data = train_data, 
                            h = h,
                            lr = lr, 
                            iter = iter, 
                            seed = seed, 
                            activation_hidden = activation_hidden)
        
        
        X_train <- scale(X_train)
        if (is.numeric(y_train) !=T & nlevels(y_train)==2){
          y_train_pred <- 1-predictNN(X_train, nn_model$weights, nn_model$biases, h, activation_output = activation_output, 
                                      min_Y = nn_model$min_Y, max_Y = nn_model$max_Y)
          X_test <- scale(X_test)
          y_test_pred <- 1-predictNN(X_test, nn_model$weights, nn_model$biases, h, activation_output = activation_output, 
                                     min_Y = nn_model$min_Y, max_Y = nn_model$max_Y)
        }
        else if(is.numeric(y_train) !=T & nlevels(y_train)>2){
          print(levels(y_train))
          print(levels(y_test))
          y_train_pred <- predictNN(X_train, nn_model$weights, nn_model$biases, h, activation_output = activation_output, 
                                    min_Y = nn_model$min_Y, max_Y = nn_model$max_Y, Y_levels = levels(y_train))
          X_test <- scale(X_test)
          y_test_pred <- predictNN(X_test, nn_model$weights, nn_model$biases, h, activation_output = activation_output, 
                                   min_Y = nn_model$min_Y, max_Y = nn_model$max_Y, Y_levels = levels(y_test))
        }
        else{
          y_train_pred <- predictNN(X_train, nn_model$weights, nn_model$biases, h, activation_output = activation_output, 
                                      min_Y = nn_model$min_Y, max_Y = nn_model$max_Y)
          X_test <- scale(X_test)
          y_test_pred <- predictNN(X_test, nn_model$weights, nn_model$biases, h, activation_output = activation_output, 
                                     min_Y = nn_model$min_Y, max_Y = nn_model$max_Y)
        }
        
      }
      
      
      # Ocena wyników w zależności od typu problemu
      if (is.numeric(y_train)) {  # Regresja
        train_metrics <- ModelEval(y_train, y_train_pred)
        test_metrics <- ModelEval(y_test, y_test_pred)
        
        # Zbieramy wyniki do tabeli
        results <- rbind(results, data.frame(
          k = ifelse(algorithm == "KNN", k_value, NA),
          max_depth = ifelse(algorithm == "DecisionTree", max_depth, NA),
          min_samples = ifelse(algorithm == "DecisionTree", min_samples, NA),
          pruning_method = ifelse(algorithm == "DecisionTree", pruning_method, NA),
          cf = ifelse(algorithm == "DecisionTree", cf, NA),
          criterion = ifelse(algorithm == "DecisionTree", criterion, NA),
          hidden_units = ifelse(algorithm == "NN", h, NA),  # Liczba neuronów w warstwie ukrytej
          activation_hidden = ifelse(algorithm == "NN", activation_hidden, NA),  # Funkcja aktywacji warstwy ukrytej
          lr = ifelse(algorithm == "NN", lr, NA),  # Współczynnik uczenia
          iter = ifelse(algorithm == "NN", iter, NA),  # Liczba iteracji
          activation_output = ifelse(algorithm == "NN", activation_output, NA),
          AUC_t = NA, 
          Sensitivity_t = NA, 
          Specificity_t = NA, 
          Accuracy_t = NA,
          AUC_w = NA, 
          Sensitivity_w = NA, 
          Specificity_w = NA, 
          Accuracy_w = NA, 
          MAE_t = train_metrics["MAE"],
          MSE_t = train_metrics["MSE"],
          MAPE_t = train_metrics["MAPE"],
          MAE_w = test_metrics["MAE"],
          MSE_w = test_metrics["MSE"],
          MAPE_w = test_metrics["MAPE"],
          fold = j  # Numer folda
        ))
        
      } else if (is.factor(y_train) && nlevels(y_train) == 2) {  # Klasyfikacja binarna
        train_metrics <- ModelEval(y_train, y_train_pred[,1])
        test_metrics <- ModelEval(y_test, y_test_pred[,1])
        
        # Zbieramy wyniki do tabeli
        results <- rbind(results, data.frame(
          k = ifelse(algorithm == "KNN", k_value, NA),
          max_depth = ifelse(algorithm == "DecisionTree", max_depth, NA),
          min_samples = ifelse(algorithm == "DecisionTree", min_samples, NA),
          pruning_method = ifelse(algorithm == "DecisionTree", pruning_method, NA),
          cf = ifelse(algorithm == "DecisionTree", cf, NA),
          criterion = ifelse(algorithm == "DecisionTree", criterion, NA),
          hidden_units = ifelse(algorithm == "NN", h, NA),  # Liczba neuronów w warstwie ukrytej
          activation_hidden = ifelse(algorithm == "NN", activation_hidden, NA),  # Funkcja aktywacji warstwy ukrytej
          lr = ifelse(algorithm == "NN", lr, NA),  # Współczynnik uczenia
          iter = ifelse(algorithm == "NN", iter, NA),  # Liczba iteracji
          activation_output = ifelse(algorithm == "NN", activation_output, NA),
          AUC_t = train_metrics$Metrics["AUC"], 
          Sensitivity_t = train_metrics$Metrics["Sensitivity"], 
          Specificity_t = train_metrics$Metrics["Specificity"], 
          Accuracy_t = train_metrics$Metrics["Accuracy"],
          AUC_w = test_metrics$Metrics["AUC"], 
          Sensitivity_w = test_metrics$Metrics["Sensitivity"], 
          Specificity_w = test_metrics$Metrics["Specificity"], 
          Accuracy_w = test_metrics$Metrics["Accuracy"],
          MAE_t = NA,
          MSE_t = NA,
          MAPE_t = NA,
          MAE_w = NA,
          MSE_w = NA,
          MAPE_w = NA,
          fold = j  # Numer folda
        ))
        
      } else if (is.factor(y_train) && nlevels(y_train) > 2) {  # Klasyfikacja wieloklasowa
        train_metrics <- ModelEval(y_train, y_train_pred)
        test_metrics <- ModelEval(y_test, y_test_pred)
        
        # Zbieramy wyniki do tabeli
        results <- rbind(results, data.frame(
          k = ifelse(algorithm == "KNN", k_value, NA),
          max_depth = ifelse(algorithm == "DecisionTree", max_depth, NA),
          min_samples = ifelse(algorithm == "DecisionTree", min_samples, NA),
          pruning_method = ifelse(algorithm == "DecisionTree", pruning_method, NA),
          cf = ifelse(algorithm == "DecisionTree", cf, NA),
          criterion = ifelse(algorithm == "DecisionTree", criterion, NA),
          hidden_units = ifelse(algorithm == "NN", h, NA),  # Liczba neuronów w warstwie ukrytej
          activation_hidden = ifelse(algorithm == "NN", activation_hidden, NA),  # Funkcja aktywacji warstwy ukrytej
          lr = ifelse(algorithm == "NN", lr, NA),  # Współczynnik uczenia
          iter = ifelse(algorithm == "NN", iter, NA),  # Liczba iteracji
          activation_output = ifelse(algorithm == "NN", activation_output, NA),
          AUC_t = train_metrics$AUC_multi, 
          Sensitivity_t = NA, 
          Specificity_t = NA, 
          Accuracy_t = train_metrics$Accuracy,
          AUC_w = test_metrics$AUC_multi, 
          Sensitivity_w = NA, 
          Specificity_w = NA, 
          Accuracy_w = test_metrics$Accuracy,
          MAE_t = NA,
          MSE_t = NA,
          MAPE_t = NA,
          MAE_w = NA,
          MSE_w = NA,
          MAPE_w = NA,
          fold = j  # Numer folda
        ))
      }
    }
    avg_fold_results <- data.frame(
      AUC_t = mean(results$AUC_t, na.rm = TRUE),
      Sensitivity_t = mean(results$Sensitivity_t, na.rm = TRUE),
      Specificity_t = mean(results$Specificity_t, na.rm = TRUE),
      Accuracy_t = mean(results$Accuracy_t, na.rm = TRUE),
      AUC_w = mean(results$AUC_w, na.rm = TRUE),
      Sensitivity_w = mean(results$Sensitivity_w, na.rm = TRUE),
      Specificity_w = mean(results$Specificity_w, na.rm = TRUE),
      Accuracy_w = mean(results$Accuracy_w, na.rm = TRUE),
      MAE_t = mean(results$MAE_t, na.rm = TRUE),
      MSE_t = mean(results$MSE_t, na.rm = TRUE),
      MAPE_t = mean(results$MAPE_t, na.rm = TRUE),
      MAE_w = mean(results$MAE_w, na.rm = TRUE),
      MSE_w = mean(results$MSE_w, na.rm = TRUE),
      MAPE_w = mean(results$MAPE_w, na.rm = TRUE)
    )
    
    results_m[[i]] <- cbind(param_grid[i,],avg_fold_results)
  }
  final_results <- do.call(rbind, results_m)
  final_results <- final_results[, colSums(is.na(final_results)) < nrow(final_results)]
  #results <- results[, colSums(is.na(results)) < nrow(results)]
  return(final_results)
}
# parTune_example <- list(k = c(3, 5, 7,10))
# results <- CrossValidTune(dane = data[1:200,], kFold = 5, parTune = parTune_example, seed = 123, algorithm = "KNN")
# print(results)#działa, binarna
# 
# parTune <- list(max_depth = c(3, 5, 7, 10), min_samples = c(2, 5, 10,15), pruning_method= c("prune", "none"), criterion = c("Entropy", "Gini"), cf = c(0.1, 0.05, 0.2))  # Lista parametrów dla drzewa
# results <- CrossValidTune(data, kFold = 5, parTune = parTune, seed = 123, algorithm = "DecisionTree")#działa, binarna
# options(max.print = 10000)
# 
# 
# 
# parTune_example <- data.frame(k = c(3, 5, 7, 10))
# results <- CrossValidTune(dane = data_students[1:200,], kFold = 5, parTune = parTune_example, seed = 123, algorithm = "KNN")#działa, wieloklasowa
# 
# print(results)
# 
# parTune <- list(max_depth = c(5,10,15), min_samples = c(5,10,15), pruning_method= c("prune"), criterion = c("Entropy", "Gini"), cf = c(0.1, 0.05))
# results <- CrossValidTune(dane = data_students, kFold = 5, parTune = parTune, seed = 123, algorithm = "DecisionTree")#działa, wieloklasowa
# 
# 
# print(results)
# 
# parTune_example <- data.frame(k = c(3, 5, 7, 10))
# results <- CrossValidTune(dane = boston_data[1:200,], kFold = 5, parTune = parTune_example, seed = 123, algorithm = "KNN")#działa, regresja
# 
# print(results)
# 
# 
# parTune <- list(max_depth = c(1,5, 10, 15), min_samples = c(1,5, 10, 15), pruning_method= c("prune"), criterion = c("SS"), cf = c(0.1, 0.2, 0.3, 0.4))
# results <- CrossValidTune(dane = boston_data, kFold = 5, parTune = parTune, seed = 123, algorithm = "DecisionTree")#działa, regresja
# 
# print(results)
# 
# parTune <- list(hidden_units = list(c(64,32), c(32,16)), activation_hidden = c("sigmoid", "tanh", "linear"),lr=c(0.01, 0.001), iter=c(1000), seed=c(123), activation_output=c("linear"))
# results <- CrossValidTune(dane = boston_data, kFold = 5, parTune = parTune, seed = 123, algorithm = "NN")# regresja, działa
# results
# 
# parTune <- list(hidden_units = list(c(32,16)), activation_hidden = c("sigmoid", "tanh"),lr=c(0.01), iter=c(10000), seed=c(123), activation_output=c("sigmoid"))
# results <- CrossValidTune(dane = data, kFold = 5, parTune = parTune, seed = 123, algorithm = "NN")# diała binarna
# results
# 
# 
# 
# parTune <- list(hidden_units = list(c(32,16)), activation_hidden = c("sigmoid"),lr=c(0.01), iter=c(10000), seed=c(123), activation_output=c("softmax"))
# results <- CrossValidTune(dane = data_students, kFold = 5, parTune = parTune, seed = 123, algorithm = "NN")# działa wieloklasowa
# results
# #results[which.max(results$Accuracy_w),]
# 
# 



# Funkcja aktywacji
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

softmax <- function(x) {
  exp(x) / sum(exp(x))
}

linear <- function(x) {
  return(x)  # Funkcja liniowa
}
relu <- function(x) {
  return(pmax(0, x))  # Zwraca x, jeżeli x > 0, w przeciwnym razie 0
}
leaky_relu <- function(x, alpha = 0.01) {
  return(pmax(alpha * x, x))  # Zwraca x, jeżeli x > 0, w przeciwnym razie alpha * x
}
tanh_activation <- function(x) {
  return(tanh(x))  # Funkcja tanh zwraca wartości z zakresu -1 do 1
}

activation <- function(x, type="relu") {
  if (type == "sigmoid") {
    return(1 / (1 + exp(-x)))  # Sigmoid
  } else if (type == "softmax") {
    exp_x <- exp(x - max(x))  # Stabilność numeryczna
    return(exp_x / rowSums(exp_x)) # Softmax
  } else if (type == "relu") {
    return(relu(x))  # ReLU
  } else if (type == "tanh") {
    return(tanh(x))  # Tanh
  }
  return(x)  # Funkcja liniowa
}

# Funkcja do obliczania pochodnej aktywacji
activation_derivative <- function(A, type="sigmoid") {
  if (type == "sigmoid") {
    return(A * (1 - A))  # Pochodna Sigmoida
  }
  return(rep(1, length(A)))  # Dla funkcji liniowej, pochodna = 1
}

# Funkcja Binary Crossentropy
binary_crossentropy <- function(y_true, y_pred) {
  return(-mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)))
}

# Funkcja Categorical Crossentropy
categorical_crossentropy <- function(y_true, y_pred) {
  return(-mean(rowSums(y_true * log(y_pred))))
}

# Funkcja normalizująca wektor do [0, 1]
normalize <- function(x) {
  min_val <- min(x)
  max_val <- max(x)
  scaled <- (x - min_val) / (max_val - min_val)
  return(list(scaled = scaled, min_val = min_val, max_val = max_val))
}

# Funkcja denormalizująca wektor
denormalize <- function(scaled, min_val, max_val) {
  return(scaled * (max_val - min_val) + min_val)
}

# Funkcja treningowa NN
trainNN <- function(Yname, Xnames, data, h, lr, iter, seed, activation_hidden = "sigmoid", update_method = "backprop") {
  set.seed(seed)
  
  # Przygotowanie danych
  X <- as.matrix(data[, Xnames])
  Y <- data[[Yname]]
  
  # Upewnij się, że X i Y są typu numerycznego
  cat("Czy X jest numeryczne?", is.numeric(X), "\n")
  cat("Czy Y jest numeryczne?", is.numeric(Y), "\n")
  
  # Normalizacja X
  X_norm <- scale(X)  
  #print(X_norm)
  
  
  if (is.factor(Y)) {
    if (nlevels(Y) == 2) {
      Y_scaled <- as.numeric(Y) - 1
      output_size <- 1
      activation_output <- "sigmoid"
      cost_function <- binary_crossentropy
    } else {
      Y_scaled <- model.matrix(~ Y - 1)  # One-hot encoding
      output_size <- ncol(Y_scaled)
      activation_output <- "softmax"
      cost_function <- categorical_crossentropy
    }
    min_Y <- NULL  # Brak normalizacji dla klasyfikacji
    max_Y <- NULL
  } else {
    # Normalizacja Y
    Y_norm <- normalize(Y)
    Y_scaled <- Y_norm$scaled
    min_Y <- Y_norm$min_val
    max_Y <- Y_norm$max_val
    
    output_size <- 1
    activation_output <- "linear"
    cost_function <- function(y_true, y_pred) mean((y_true - y_pred)^2)  # Używamy MSE dla regresji
  }
  
  input_size <- ncol(X)
  layers <- c(input_size, h, output_size)
  
  # Inicjalizacja wag i biasów
  weights <- list()
  biases <- list()
  
  # Inicjalizujemy wagi i biasy
  for (i in 1:(length(layers) - 1)) {
    weights[[i]] <- matrix(rnorm(layers[i] * layers[i+1], mean = 0, sd = sqrt(2 / layers[i])), 
                           nrow = layers[i], ncol = layers[i+1])  # Inicjalizacja He
    biases[[i]] <- runif(layers[i+1], min = -0.1, max = 0.1)
    
    # Debugowanie: Sprawdzamy wymiary wag
    #cat("Wymiary weights[[", i, "]]: ", dim(weights[[i]]), "\n")
  }
  
  # Forward propagation z wyborem funkcji aktywacji
  wprzod <- function(X, weights, biases, activation_hidden, activation_output) {
    A <- list(X)  # Aktywacja dla warstwy wejściowej
    Z <- list()
    
    # Propagacja w przód przez wszystkie warstwy
    for (i in 1:(length(layers) - 1)) {
      #cat("Wymiary A[[", i, "]]:", dim(A[[i]]), "\n")  # Debugowanie wymiarów A
      #cat("Wymiary weights[[", i, "]]:", dim(weights[[i]]), "\n")  # Debugowanie wymiarów weights
      
      Z[[i]] <- A[[i]] %*% weights[[i]] + biases[[i]]  # Z = A * W + b
      #cat("Wymiary Z[[", i, "]]:", dim(Z[[i]]), "\n")  # Debugowanie wymiarów Z
      
      # Aktywacja
      if (i == length(layers) - 1) {
        A[[i + 1]] <- activation(Z[[i]], activation_output)  # Aktywacja wyjściowa
      } else {
        A[[i + 1]] <- activation(Z[[i]], activation_hidden)  # Aktywacja ukryta
      }
      #cat("Wymiary A[[", i + 1, "]]:", dim(A[[i + 1]]), "\n")  # Debugowanie wymiarów A
    }
    
    return(list(A = A, Z = Z))
  }
  
  # Backpropagation
  wstecz <- function(X, Y, A, Z) {
    m <- nrow(X)
    dA <- A[[length(layers)]]  # Ostatnia warstwa jako wynik
    
    # Różnica między przewidywaniami a rzeczywistymi wartościami
    dA <- dA - Y
    
    dW <- list()
    db <- list()
    
    for (i in (length(layers)-1):1) {  # Iteracja po warstwach w odwrotnej kolejności
      dZ <- dA * activation_derivative(A[[i+1]], activation_output)
      dW[[i]] <- t(A[[i]]) %*% dZ / m  # Gradient wag
      db[[i]] <- colSums(dZ) / m  # Gradient biasów
      dA <- dZ %*% t(weights[[i]])  # Backpropagacja
      
      # Debugowanie wymiarów
      #cat("Wymiary dA:", dim(dA), "\n")
      #cat("Wymiary dW[[", i, "]]:", dim(dW[[i]]), "\n")
    }
    
    return(list(dW = dW, db = db))
  }
  
  # Trening sieci
  for (i in 1:iter) {
    # Propagacja w przód
    forward_output <- wprzod(X_norm, weights, biases, activation_hidden, activation_output)
    A <- forward_output$A
    Z <- forward_output$Z
    
    # Obliczanie kosztu
    cost <- cost_function(Y_scaled, A[[length(layers)]])
    
    # Backpropagacja
    grad <- wstecz(X_norm, Y_scaled, A, Z)
    
    # Aktualizacja wag i biasów
    if (update_method == "backprop") {
      for (j in 1:(length(layers) - 1)) {
        weights[[j]] <- weights[[j]] - lr * grad$dW[[j]]
        biases[[j]] <- biases[[j]] - lr * grad$db[[j]]
      }
    }
    
    # Wyświetlanie kosztu co 100 iteracji
    if (i %% 100 == 0) {
      cat("Iteracja:", i, "Koszt:", cost, "\n")
    }
  }
  
  return(list(weights = weights, biases = biases, cost = cost, min_Y = min_Y, max_Y = max_Y))
}


# Funkcja do testowania modelu
predictNN <- function(X, weights, biases, h, activation_output = "sigmoid", min_Y = NULL, max_Y = NULL, class_labels = NULL, Y_levels=NULL) {
  layers <- c(ncol(X), h, length(biases[[length(biases)]]))
  
  A <- list(X)  # Inicjalizacja aktywacji wejściowej
  Z <- list()
  
  for (i in 1:(length(layers) - 1)) {
    Z[[i]] <- A[[i]] %*% weights[[i]] + biases[[i]]
    A[[i + 1]] <- activation(Z[[i]], ifelse(i == length(layers) - 1, activation_output, "sigmoid"))
  }
  Y_pred <- A[[length(A)]]
  
  if (!is.null(min_Y) && !is.null(max_Y)) {
    Y_pred <- denormalize(Y_pred, min_Y, max_Y)
  }
  
  if (activation_output == "softmax") {
    if (is.null(class_labels)) {
      #class_labels <- colnames(weights[[length(weights)]])
      class_labels <- Y_levels
      #print(class_labels)
      #class_labels <- gsub("^Y", "", class_labels)
      if (is.null(class_labels)) {
        stop("Nie udało się automatycznie odczytać nazw klas. Ustaw je ręcznie w trakcie treningu.")
      }
    }
    #print(Y_pred)
    colnames(Y_pred) <- class_labels
    max_class <- colnames(Y_pred)[max.col(Y_pred, ties.method = "first")]
    #print(max_class)
    Y_pred <- as.data.frame(Y_pred)
    Y_pred$Predicted <- factor(max_class, levels = class_labels)
    #print(Y_pred)
  }
  
  return(Y_pred)  # Zwracamy wynik z ostatniej warstwy
}

# boston_data <- boston_data[, sapply(boston_data, is.numeric)]
# 
# 
# parTune <- list(Yname = colnames(boston_data)[ncol(boston_data)], 
#                 Xnames = colnames(boston_data)[-ncol(boston_data)], 
#                 data = boston_data, 
#                 h = c(128, 64),   # 2 warstwy ukryte: 10 neuronów w pierwszej, 5 w drugiej
#                 lr = 0.001, 
#                 iter = 10000, 
#                 seed = 123)
# 
# result <- trainNN(parTune$Yname, parTune$Xnames, parTune$data, 
#                   parTune$h, parTune$lr, parTune$iter, parTune$seed, activation_hidden = "sigmoid")
# 
# X_test <- as.matrix(boston_data[, parTune$Xnames])   # Testowe dane wejściowe
# X_test <- scale(X_test)
# 
# predictions_result <- predictNN(X_test, result$weights, result$biases,
#                                 h = c(128, 64), activation_output = "linear", min_Y = result$min_Y, max_Y = result$max_Y)
# 
# 
# ModelEval(unlist(boston_data[ncol(boston_data)]),predictions_result)# regresja działa
# 
# 
# 
# 
# 
# parTune <- list(Yname = colnames(data)[ncol(data)], 
#                 Xnames = colnames(data)[-ncol(data)], 
#                 data = data, 
#                 h = c(128, 64),   # 2 warstwy ukryte: 10 neuronów w pierwszej, 5 w drugiej
#                 lr = 0.001, 
#                 iter = 100, 
#                 seed = 123)
# 
# result <- trainNN(parTune$Yname, parTune$Xnames, parTune$data, 
#                   parTune$h, parTune$lr, parTune$iter, parTune$seed, activation_hidden = "sigmoid")
# 
# X_test <- as.matrix(data[, parTune$Xnames])   # Testowe dane wejściowe
# X_test <- scale(X_test)
# 
# predictions_result <- predictNN(X_test, result$weights, result$biases,
#                                 h = c(128, 64), activation_output = "sigmoid", min_Y = result$min_Y, max_Y = result$max_Y)
# 
# ModelEval(unlist(data[ncol(data)]),1-predictions_result)# klasyfikacja binarna działa
# predictions_result
# 
# 
# 
# parTune <- list(Yname = colnames(data_students)[ncol(data_students)], 
#                 Xnames = colnames(data_students)[-ncol(data_students)], 
#                 data = data_students[1:2000,], 
#                 h = c(10, 5),   # 2 warstwy ukryte: 10 neuronów w pierwszej, 5 w drugiej
#                 lr = 0.001, 
#                 iter = 1000, 
#                 seed = 123)
# 
# result <- trainNN(parTune$Yname, parTune$Xnames, parTune$data, 
#                   parTune$h, parTune$lr, parTune$iter, parTune$seed, activation_hidden = "sigmoid")
# 
# X_test <- as.matrix(data_students[1:200, parTune$Xnames])   # Testowe dane wejściowe
# X_test <- scale(X_test)
# 
# predictions_result <- predictNN(X_test, result$weights, result$biases,
#                                 h = c(10, 5), activation_output = "softmax", min_Y = result$min_Y, max_Y = result$max_Y, Y_levels=levels(data_students$Target))
# 
# 
# #colnames(predictions_result) <- levels(unlist(data_students[ncol(data_students)]))
# 
# #max_class <- colnames(predictions_result)[max.col(predictions_result, ties.method = "first")]
# #max_class <- data.frame(max_class)
# #colnames(max_class) <- "Predicted"
# 
# #predictions_result <- cbind(predictions_result, max_class)
# 
# #predictions_result$Predicted <- as.factor(predictions_result$Predicted)
# 
# ModelEval(unlist(data_students[ncol(data_students)]),predictions_result)
# predictions_result



# Funkcja obliczająca Gini
Gini <- function(target) {
  unique_classes <- unique(target)
  probs <- sapply(unique_classes, function(c) mean(target == c))
  return(1 - sum(probs^2))
}

# Funkcja obliczająca Entropię
Entropy <- function(target) {
  unique_classes <- unique(target)
  probs <- sapply(unique_classes, function(c) mean(target == c))
  return(-sum(probs * log2(probs + 1e-6)))  # Dodajemy małą wartość do loga dla uniknięcia log(0)
}

# Funkcja obliczająca SS
SS <- function(target) {
  mean_value <- mean(target)
  return(sum((target - mean_value)^2))
}

# Funkcja przypisująca początkowe wartości węzłowi
initializeNode <- function(node, target, data, criterion) {
  node$depth <- 0
  if (is.factor(data[[target]])) {
    # Prawdopodobieństwa dla klasyfikacji
    node$probabilities <- as.numeric(table(data[[target]]) / length(data[[target]]))  
    node$prediction <- names(node$probabilities)[which.max(node$probabilities)]  # Najczęstsza klasa
  } else {
    node$prediction <- mean(data[[target]])  # Średnia dla regresji
  }
  node$size <- length(data[[target]])
  return(node)
}

# Funkcja oceny danych wejściowych przed rozpoczęciem
validateData <- function(target, features, data, criterion, max_depth, min_samples, pruning_method, cf) {
  if (!is.data.frame(data)) {
    print("Data must be a data frame.")
    return(FALSE)
  }
  
  if (!all(c(target, features) %in% names(data))) {
    print("Missing required variables in the data.")
    return(FALSE)
  }
  
  for (var in c(target, features)) {
    if (anyNA(data[[var]])) {
      print("Missing values in the data.")
      return(FALSE)
    }
  }
  
  if (max_depth < 0 || min_samples < 0) {
    print("Maximum depth and minimum samples must be greater than 0.")
    return(FALSE)
  }
  
  if (!(criterion %in% c("Gini", "Entropy", "SS"))) {
    print("Invalid criterion.")
    return(FALSE)
  }
  
  if (!(pruning_method %in% c("none", "prune"))) {
    print("Invalid pruning method.")
    return(FALSE)
  }
  
  if (cf <= 0 || cf > 0.5) {
    print("cf must be between 0 and 0.5.")
    return(FALSE)
  }
  
  if (is.factor(data[[target]]) && criterion == "SS") {
    print("Sum of squares is not valid for classification tasks.")
    return(FALSE)
  }
  
  if (!is.factor(data[[target]]) && criterion %in% c("Gini", "Entropy")) {
    print("Entropy and Gini are not valid for regression tasks.")
    return(F)
  }
  
  return(TRUE)
}

# Funkcja dzielenia danych na najlepszy podział
bestSplit <- function(target, features, data, parent_value, criterion, min_samples) {
  best_split <- list(info_gain = -Inf, point = NULL, feature = NULL)
  
  for (feature in features) {
    if (!(feature %in% colnames(data))) {
      stop(paste("Feature", feature, "does not exist in the data"))
    }
    
    unique_values <- if (is.factor(data[[feature]])) {
      levels(data[[feature]])
    } else {
      sort(unique(data[[feature]]))
    }
    num_samples <- nrow(data)
    
    for (value in unique_values) {
      if (is.factor(data[[feature]])) {
        # Podział na podstawie równości dla zmiennych typu factor
        partition <- data[[feature]] == value
      } else {
        # Podział na podstawie wartości numerycznych
        partition <- data[[feature]] <= value
      }
      
      left_size <- sum(partition, na.rm = TRUE)
      right_size <- num_samples - left_size
      
      # Pomijamy podziały, które są zbyt małe
      if (is.na(left_size) || is.na(right_size) || left_size < min_samples || right_size < min_samples) {
        next
      }
      
      # Obliczenia dla klasyfikacji
      if (is.factor(data[[target]])) {
        left_prob <- as.numeric(table(data[[target]][partition]) / left_size)
        right_prob <- as.numeric(table(data[[target]][!partition]) / right_size)
        
        left_val <- if (criterion == "Gini") {
          1 - sum(left_prob^2)
        } else if (criterion == "Entropy") {
          -sum(left_prob * log2(left_prob + 1e-9))
        } else {
          stop("Unknown criterion for classification")
        }
        
        right_val <- if (criterion == "Gini") {
          1 - sum(right_prob^2)
        } else if (criterion == "Entropy") {
          -sum(right_prob * log2(right_prob + 1e-9))
        } else {
          stop("Unknown criterion for classification")
        }
      } else {
        # Obliczenia dla regresji (bazujące na wariancji)
        left_val <- if (left_size > 1) var(data[[target]][partition], na.rm = TRUE) else 0
        right_val <- if (right_size > 1) var(data[[target]][!partition], na.rm = TRUE) else 0
      }
      
      # Obliczanie zysku informacji
      info_gain <- parent_value - ((left_size * left_val + right_size * right_val) / num_samples)
      
      # Aktualizacja najlepszego podziału
      if (info_gain > best_split$info_gain) {
        best_split$info_gain <- info_gain
        best_split$point <- value
        best_split$left_size <- left_size
        best_split$right_size <- right_size
        best_split$feature <- feature
        if (is.factor(data[[target]])) {
          best_split$left_prob <- left_prob  # Przechowujemy prawdopodobieństwa
          best_split$right_prob <- right_prob  # Przechowujemy prawdopodobieństwa
        }
      }
    }
  }
  
  #if (is.null(best_split$feature)) {
  #  stop("No valid split found")
  #}
  
  return(best_split)
}

# Funkcja budująca drzewo decyzyjne
buildTree <- function(node, target, features, data, criterion, max_depth, min_samples) {
  # Obliczanie wartości węzła nadrzędnego
  if (is.factor(data[[target]])) {
    # Klasyfikacja: Obliczanie rozkładu prawdopodobieństw
    prob <- table(data[[target]]) / length(data[[target]])
    parent_value <- if (criterion == "Gini") {
      1 - sum(prob^2)
    } else if (criterion == "Entropy") {
      -sum(prob * log2(prob + 1e-9))
    } else {
      stop("Unknown criterion for classification")
    }
    # Ustawienie predykcji i błędu w węźle
    node$probabilities <- prob
    node$prediction <- names(prob)[which.max(prob)]
    node$error <- 1 - max(prob)
    # Jeśli węzeł jest czysty (wszystkie dane należą do jednej klasy)
    if (all(prob %in% c(0, 1))) {
      node$leaf <- "*"
      return(node)
    }
  } else {
    # Regresja: Obliczanie średniej wartości
    parent_value <- var(data[[target]])
    node$prediction <- mean(data[[target]])
  }
  
  # Sprawdzenie warunków zakończenia rekurencji
  if (node$depth == max_depth || node$size < 2 * min_samples) {
    node$leaf <- "*"
    return(node)
  }
  
  # Znalezienie najlepszego podziału
  best_split <- bestSplit(target, features, data, parent_value, criterion, min_samples)
  
  if (is.null(best_split$feature)) {
    node$leaf <- "*"
    return(node)
  }
  # Informacja o najlepszym podziale
  feature <- best_split$feature
  split_point <- best_split$point
  node$split_feature <- feature
  node$split_point <- split_point
  node$probabilities <- best_split$left_prob  # Ustawiamy prawdopodobieństwa w węźle
  
  # Podział danych
  if (is.factor(data[[feature]]) && !is.ordered(data[[feature]])) {
    split_idx <- data[[feature]] == split_point
    left_child <- node$AddChild(paste0(feature, " == ", split_point))
  } else {
    split_idx <- data[[feature]] <= split_point
    left_child <- node$AddChild(paste0(feature, " <= ", split_point))
  }
  
  # Lewa gałąź
  left_child$depth <- node$depth + 1
  left_child$prediction <- best_split$left_val
  left_child$size <- best_split$left_size
  left_child$probabilities <- best_split$left_prob  # Ustawiamy prawdopodobieństwa
  buildTree(left_child, target, features, data[split_idx, ], criterion, max_depth, min_samples)
  
  # Prawa gałąź
  right_child <- node$AddChild(paste0(feature, " > ", split_point))
  right_child$depth <- node$depth + 1
  right_child$prediction <- best_split$right_val
  right_child$size <- best_split$right_size
  right_child$probabilities <- best_split$right_prob  # Ustawiamy prawdopodobieństwa
  buildTree(right_child, target, features, data[!split_idx, ], criterion, max_depth, min_samples)
  
  return(node)
}

# Funkcja przycinania drzewa
pruneTree <- function(tree, min_samples) {
  pruneNode <- function(node) {
    if (!is.null(node$leaf)) {
      if (node$size < min_samples) {
        return(TRUE)
      }
      return(FALSE)
    }
    
    prune_flags <- sapply(node$children, pruneNode)
    for (i in seq_along(prune_flags)) {
      if (prune_flags[i]) {
        node$RemoveChild(names(node$children)[i])
      }
    }
    
    if (length(node$children) == 0) {
      node$leaf <- "*"
      return(FALSE)
    }
    
    return(FALSE)
  }
  
  pruneNode(tree)
  return(tree)
}

# Główna funkcja do tworzenia drzewa
decisionTree <- function(target, features, data, criterion, max_depth, min_samples, pruning_method, cf) {
  if (!validateData(target, features, data, criterion, max_depth, min_samples, pruning_method, cf)) {
    stop()
  } else {
    tree <- Node$new("Root")
    initializeNode(tree, target, data, criterion)
    buildTree(tree, target, features, data, criterion, max_depth, min_samples)
    
    if (pruning_method == "prune") {
      tree <- pruneTree(tree, min_samples)
    }
    
    return(tree)
  }
}

# Funkcja przewidywania wyników z drzewa
predictTree <- function(tree, data) {
  required_vars <- unique(sapply(tree$childrenRecursive, function(node) node$split_feature))
  if (!all(required_vars %in% colnames(data))) {
    stop("Variables in the tree do not match variables in the data.")
  }
  
  predictions <- lapply(1:nrow(data), function(i) {
    node <- tree
    while (is.null(node$leaf)) {
      split_condition <- strsplit(node$children[[1]]$name, " ")[[1]]
      feature <- split_condition[1]
      operator <- split_condition[2]
      threshold <- paste(split_condition[-(1:2)], collapse = " ")
      
      if (!(feature %in% colnames(data))) {
        stop(paste("Feature", feature, "not found in data"))
      }
      
      value <- data[i, feature]
      if (is.na(value)) {
        node <- node$children[[2]]
        next
      }
      
      if (operator == "<=") {
        if (as.numeric(value) <= as.numeric(threshold)) {
          node <- node$children[[1]]
        } else {
          node <- node$children[[2]]
        }
      } else if (operator == ">") {
        if (as.numeric(value) > as.numeric(threshold)) {
          node <- node$children[[1]]
        } else {
          node <- node$children[[2]]
        }
      } 
      else if (operator == "==") {  # Dodanie obsługi operatora '=='
        if (as.character(value) == threshold) {
          node <- node$children[[1]]
        } else {
          node <- node$children[[2]]
        }
      } 
      
      else {
        stop(paste("Unknown operator:", operator))
      }
    }
    
    prediction <- node$prediction
    probabilities <- as.numeric(node$probabilities)
    result <- c(probabilities, prediction)
    return(result)
  })
  
  #print("Predictions:")
  #print(predictions)
  
  predictions_matrix <- do.call(rbind, predictions)
  
  #print("Predictions Matrix:")
  #print(predictions_matrix)
  
  results_df <- as.data.frame(predictions_matrix)
  
  #print("Results DataFrame before processing:")
  #print(results_df)
  
  if (ncol(results_df) > 1) {
    results_df[, ncol(results_df)] <- factor(results_df[, ncol(results_df)])
    for (i in 1:(ncol(results_df) - 1)) {
      results_df[[i]] <- as.numeric(results_df[[i]])
    }
    class_names <- levels(results_df[,ncol(results_df)])
    colnames(results_df) <- c(paste0("Class_", 1:(ncol(results_df) - 1)), "Prediction")
    return(results_df)
  }
  else{
    #print("Results DataFrame after processing:")
    #print(results_df)
    
    return(unlist(results_df))
  }
}

# Parametry drzewa decyzyjnego
# criterion <- "Gini"  # Kryterium podziału: Gini, Entropy lub SS
# max_depth <- 3  # Maksymalna głębokość drzewa
# min_samples <- 1  # Minimalna liczba próbek na węzeł
# pruning_method <- "prune"  # Metoda przycinania: "none" lub "prune"
# cf <- 0.1  # Parametr cf (do przycinania, jeśli jest aktywne)
# 
# # Wywołanie funkcji do stworzenia drzewa decyzyjnego
# tree <- decisionTree(colnames(data_students)[ncol(data_students)], colnames(data_students)[-ncol(data_students)], data_students, criterion, max_depth, min_samples, pruning_method, cf)
# 
# # Wyświetlanie drzewa
# print(tree)
# 
# # Przewidywanie wyników na nowych danych (np. na tych samych danych)
# predictions <- predictTree(tree, data_students[-ncol(data_students)])
# ModelEval(data_students$Target, predictions)
# 
# 
# tree <- decisionTree(colnames(data)[ncol(data)], colnames(data)[-ncol(data)], data, criterion, max_depth, min_samples, pruning_method, cf)
# 
# print(tree)
# 
# predictions <- predictTree(tree, data[-ncol(data)])
# ModelEval(data$HeartDisease, predictions[[1]])
# 
# 
# tree <- decisionTree(colnames(boston_data)[ncol(boston_data)], colnames(boston_data)[-ncol(boston_data)], boston_data, "SS", max_depth, min_samples, pruning_method, cf)
# 
# print(tree)
# 
# predictions <- predictTree(tree, boston_data[-ncol(boston_data)])
# ModelEval(boston_data$MEDV, predictions)
# predictions




MAE_func <- function(actual, predicted) {
  return(mean(abs(actual - predicted)))
}

MSE_func <- function(actual, predicted) {
  return(mean((actual - predicted)^2))
}

MAPE_func <- function(actual, predicted) {
  return(mean(abs((actual - predicted) / actual))*100)
}

prepare_data <- function(actual, predicted) {
  sorted_data <- data.frame(actual[order(predicted, decreasing = TRUE)], predicted[order(predicted, decreasing = TRUE)])
  colnames(sorted_data) <- c("actual", "predicted")
  return(sorted_data)
}

sensitivity_calculation <- function(actual, predicted) {
  sorted_df <- prepare_data(actual, predicted)
  sorted_df$sensitivity <- cumsum(as.numeric(sorted_df$actual == levels(actual)[1]))
  sorted_df$sensitivity_pct <- sorted_df$sensitivity / max(sorted_df$sensitivity)
  return(sorted_df$sensitivity_pct)
}

specificity_calculation <- function(actual, predicted) {
  sorted_df <- prepare_data(actual, predicted)
  sorted_df$FP <- cumsum(as.numeric(sorted_df$actual == levels(actual)[2]))
  sorted_df$Spec <- max(sorted_df$FP) - sorted_df$FP
  sorted_df$specificity_pct <- sorted_df$Spec / sum(actual == levels(actual)[2])
  return(sorted_df$specificity_pct)
}

evaluation_df <- function(actual, predicted) {
  sorted_df <- prepare_data(actual, predicted)
  sorted_df$sensitivity <- sensitivity_calculation(actual, predicted)
  sorted_df$specificity <- specificity_calculation(actual, predicted)
  return(sorted_df)
}

quality_score_df <- function(actual, predicted) {
  sorted_df <- evaluation_df(actual, predicted)
  sorted_df$quality_score <- ((sorted_df$sensitivity * sum(actual == levels(actual)[1])) + 
                                (sorted_df$specificity * sum(actual == levels(actual)[2]))) / length(actual)
  return(sorted_df)
}

youden_index_df <- function(actual, predicted) {
  sorted_df <- quality_score_df(actual, predicted)
  sorted_df$Youden_index <- sorted_df$sensitivity + sorted_df$specificity - 1
  return(sorted_df)
}

AUC_func <- function(actual, predicted) {
  sorted_df <- evaluation_df(actual, predicted)
  total_sum <- -sorted_df$sensitivity[1]
  for (i in 2:length(actual)) {
    total_sum <- total_sum + (sorted_df$sensitivity[i - 1] * sorted_df$specificity[i] - (sorted_df$sensitivity[i] * sorted_df$specificity[i - 1]))
  }
  return(total_sum / -2)
}

confusion_matrix_binary <- function(actual, predicted) {
  eval_df <- youden_index_df(actual, predicted)
  max_Youden_index <- which.max(eval_df$Youden_index)
  threshold <- eval_df$predicted[max_Youden_index]
  
  TP <- sum(actual == levels(actual)[1] & predicted >= threshold)
  TN <- sum(actual == levels(actual)[2] & predicted < threshold)
  FP <- sum(actual == levels(actual)[2] & predicted >= threshold)
  FN <- sum(actual == levels(actual)[1] & predicted < threshold)
  
  conf_matrix <- matrix(c(TP, FP, FN, TN), nrow = 2, byrow = FALSE)
  dimnames(conf_matrix) <- list("actual" = levels(actual), "predicted" = levels(actual))
  return(conf_matrix)
}

confusion_matrix_multiclass <- function(actual, predicted) {
  conf_matrix <- matrix(0, nrow = nlevels(actual), ncol = nlevels(actual))
  dimnames(conf_matrix) <- list("actual" = levels(actual), "predicted" = levels(actual))
  
  for (index in 1:length(actual)) {
    row_ind <- which(levels(actual) == actual[index])
    col_ind <- which(levels(predicted) == predicted[index])
    conf_matrix[row_ind, col_ind] <- conf_matrix[row_ind, col_ind] + 1
  }
  return(conf_matrix)
}

MultiAUC <- function(actual, predicted) {
  classes <- levels(actual)
  auc_values <- numeric()
  
  # Iterate over each pair of classes
  for (i in 1:nlevels(actual)) {
    for (j in 1:nlevels(actual)) {
      if (i != j) {
        # Create subsets for class i (positive) vs class j (negative)
        excluded_indices <- which(actual == classes[i])
        actual_sub <- actual[-excluded_indices]  # Remove class i
        predicted_sub <- predicted[-excluded_indices, j]  # Keep only the probabilities for class j
        
        # Relevel classes to make class j the reference (negative)
        actual_sub <- droplevels(actual_sub)
        actual_sub <- relevel(actual_sub, ref = classes[j])  # Relevel so that class j is treated as the negative class
        
        # Calculate AUC for the given pair
        auc_value <- AUC_func(actual_sub, predicted_sub)
        
        # Add the result to the vector
        auc_values <- c(auc_values, auc_value)
      }
    }
  }
  
  # Return the average AUC across all class pairs
  return(mean(auc_values))
}
ModelEval <- function(actual, predicted){
  if (is.numeric(actual)){
    mae_value <- MAE_func(actual, predicted)
    mse_value <- MSE_func(actual, predicted)
    mape_value <- MAPE_func(actual, predicted)
    result <- c(mae_value, mse_value, mape_value)
    names(result) <- c("MAE", "MSE", "MAPE")
    return(result)
  } else if (is.factor(actual)){
    if (nlevels(actual) == 2) {
      auc_value <- AUC_func(actual, predicted)
      eval_df <- youden_index_df(actual, predicted)
      max_Youden_index <- which.max(eval_df$Youden_index)
      threshold <- eval_df$predicted[max_Youden_index]
      last_index <- tail(which(eval_df$predicted == threshold), 1)
      
      confusion_mat <- confusion_matrix_binary(actual, predicted)
      J_index <- c(max(eval_df$Youden_index), threshold)
      metrics <- c(AUC = auc_value, Sensitivity = eval_df$sensitivity[last_index], Specificity = eval_df$specificity[last_index], Accuracy = eval_df$quality_score[last_index])
      return(list(ConfMat = confusion_mat, Youden = J_index, Metrics = metrics))
    } else {
      if (is.factor(predicted)!=T){
        multiclass_conf_mat <- confusion_matrix_multiclass(actual, predicted[,ncol(predicted)])
        auc_multi <- MultiAUC(actual, predicted)
      }
      else{
        multiclass_conf_mat <- confusion_matrix_multiclass(actual, predicted)
        auc_multi <- NA
      }
      
      accuracy <- sum(diag(multiclass_conf_mat)) / length(actual)
      precision <- diag(multiclass_conf_mat) / rowSums(multiclass_conf_mat)
      recall <- diag(multiclass_conf_mat) / colSums(multiclass_conf_mat)
      f1 <- 2 * (precision * recall) / (precision + recall)
      f1_macro <- mean(f1, na.rm = TRUE)
      

      return(list(Matrix = multiclass_conf_mat, Accuracy = accuracy, Precision = precision, Recall = recall, F1_macro = f1_macro,AUC_multi = auc_multi))
    }
  } else {
    stop("Invalid input types")
  }
}

# y_tar_class <- as.factor(c(1, 0, 1, 0, 1, 0))
# y_hat_class <- c(0.9, 0.2, 0.8, 0.4, 0.7, 0.1)
# ModelOcena(y_tar_class,y_hat_class)
# 
# 
# # Dane testowe dla klasyfikacji wieloklasowej
# y_tar_class <- factor(c(1, 0, 2, 1, 0, 2, 1, 2, 0))
# y_hat_class <- factor(c(1, 0, 2, 2, 0, 1, 1, 2, 0), levels = levels(y_tar_class))
# 
# # Wywołanie funkcji
# wynik <- ModelOcena(y_tar_class, y_hat_class)
# 
# # Wynik
# print(wynik)

