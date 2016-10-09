---
title: "Assignment Practical Machine Learning"
author: "Herminio Vazquez"
date: "08/10/2016"
output: html_document
---

## Introduction

This is the final assignment in the __Practical Machine Learning__ Course in Cousera during the Autum (Northern Hemisphere) of 2016. It encloses the final work dedicated to apply the concepts learnt assoaciated to machine learning routines.

The submission includes a prediction to determine the class of exercise realized by a group of individuals, through the metrics collected via wearable devices. Such metrics are result of performing specific classes of exercises and evaluating their form and correctness. The outcome of this essay is the prediction results obtained after composing a model in a training set and obtaining the predictions in 20 examples in the test set.

### Classes
The data set evaluate 5 classes: 

1. sitting-down
2. standing-up
3. standing
4. walking
5. sitting

collected on 8 hours of activities of 4 healthy subjects.


```r
# Loading Libraries
library(knitr)
library(caret)
library(randomForest)
library(RCurl)
library(xtable)
```

## Data Preparation

```r
# Loading data sets
training_url <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
in_train  <- read.csv(text = training_url, header=TRUE, sep=",", na.strings=c("NA","","#DIV/0!"))

testing_url <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
testing  <- read.csv(text = testing_url, header=TRUE, sep=",", na.strings=c("NA","","#DIV/0!"))
```

### EDA

Let's explore the data and understand measurements, dimensions and variability


```r
names(in_train)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

```r
dim(in_train)
```

```
## [1] 19622   160
```

```r
# Removing first column as it has only row ids
in_train$X <- NULL
testing$X <- NULL
```

## Model
Creating a classification model with labels on `training$classe` for the outcome predictor. The split for the training set is **80%** for the training and **20%** for the testing inside the training data set, later on the testing is used for validation.


```r
# Using the training dataset to form an extra validation set
in_train_part <- createDataPartition(y = in_train$classe, p=.60, list=FALSE)
training <- in_train[in_train_part,]
validating <- in_train[-in_train_part,]

# Various columns have multiple NA's therefore we will center in columns with data
Keep <- c((colSums(!is.na(training[,-ncol(training)])) >= 0.6*nrow(training)))
training   <-  training[,Keep]
validating <- validating[,Keep]

# Dimensions of training and testing sets
dim(training)
```

```
## [1] 11776    59
```

```r
dim(validating)
```

```
## [1] 7846   59
```

Generation of a **Random Forest** classifier for the metrics


```r
# Creation a RandomForrest Model
model <- randomForest(classe ~ ., data=training)
```

Print out the confusion matrix for the random forest

```r
confusionMatrix(predict(model,newdata=validating[,-ncol(validating)]),validating$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1516    2    0    0
##          C    0    1 1366    3    0
##          D    0    0    0 1283    1
##          E    0    0    0    0 1441
## 
## Overall Statistics
##                                          
##                Accuracy : 0.999          
##                  95% CI : (0.998, 0.9996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9987         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   0.9985   0.9977   0.9993
## Specificity            0.9998   0.9997   0.9994   0.9998   1.0000
## Pos Pred Value         0.9996   0.9987   0.9971   0.9992   1.0000
## Neg Pred Value         1.0000   0.9997   0.9997   0.9995   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1932   0.1741   0.1635   0.1837
## Detection Prevalence   0.2846   0.1935   0.1746   0.1637   0.1837
## Balanced Accuracy      0.9999   0.9992   0.9990   0.9988   0.9997
```

## Results

The following section presents the accuracy of the predictor, and the evaluation on the validation and test sets.


```r
accuracy <-c(as.numeric(predict(model,newdata=validating[,-ncol(validating)])==validating$classe))
accuracy <-sum(accuracy)*100/nrow(validating)
print(accuracy)
```

```
## [1] 99.89804
```

**Ammending Test Set**

```r
testing <- testing[ , Keep] # Keep the same columns of testing dataset
testing <- testing[,-ncol(testing)] # Remove the problem ID

# Apply the Same Transformations and Coerce Testing Dataset
# Coerce testing dataset to same class and strucuture of training dataset 
testing <- rbind(training[100, -59] , testing) 
# Apply the ID Row to row.names and 100 for dummy row from testing dataset 
row.names(testing) <- c(100, 1:20)
```

## Predictions

This is the outcome of this classification exercise for the 20 observations available in the test set


```r
predictions <- predict(model,newdata=testing[-1,])
print(predictions)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
