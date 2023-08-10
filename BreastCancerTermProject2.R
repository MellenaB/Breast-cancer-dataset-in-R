## Group 2 
## Term project Breast Cancer Data 
## Conducted R-language breast cancer data analysis. 
## Applied supervised learning KNN to predict the best breast cancer attribute to predict diagnosis. 

BiocManager::install("limma", force = TRUE)
BiocManager::install("caret")
BiocManager::install("pROC")
BiocManager::install("ggplot2")
BiocManager::install("tidymodels")
BiocManager::install("recipes")
BiocManager::install("lattic")


library(ggplot2)
library(caret)
library(pROC)         ## was built under 3.6.3 version


## Import data in R and familiarize with the dataset.
setwd("C:/Users/melle/OneDrive/Desktop/ASU Fall 21 Classes/BMI 311/TermProject")

## Load breast cancer data
data<-read.csv("data.cancer.csv")
 

## Familiarizing the dataset
head(data)
dim(data)
str(data)



## (1) Mean and Standard Dev.
summary(data)

sd(data$clump_thickness)
sd(data$shape_uniformity)
sd(data$bland_chromatin)

## (2) Create histogram for 3 numerical attributes
## Histogram plot
## Synatax for histogram: hist(v,main,xlab,xlim,ylim,breaks,col,border)
#hist(data$clump_thickness, main = 'Breast Cancer Histogram', xlab = 'Clump Thickness', ylab = 'Count',
#     xlim = c(0,10), ylim = c(0,200),
  #   col = 'lightblue', breaks =5)

## ggplot 1
## Syntax for ggplot: ggplot(data = NULL, mapping = aes(), ..., environment = parent.frame())
ggplot(data, aes(clump_thickness, fill = as.factor(class))) +
geom_histogram(binwidth = 1) +
theme(legend.position = "bottom") +
ggtitle("Clump Thickness Vs Class")

## ggplot 2
ggplot(data, aes(shape_uniformity, fill = as.factor(class))) +
  geom_histogram(binwidth = 1) +
  theme(legend.position = "bottom") +
  ggtitle("Shape Uniformity Vs Class")

## ggplot 3
ggplot(data, aes(bland_chromatin, fill = as.factor(class))) +
  geom_histogram(binwidth = 1) +
  theme(legend.position = "bottom") +
  ggtitle("Bland Chromatin Vs Class")

#ggplot(data, aes(clump_thickness, fill = mitoses)) +
#geom_histogram(binwidth = 1) +
#theme(legend.position = 'bottom') +
#ggtitle("  Clump Thickness Vs Mitoses")

## (3) Check for NA or zero values
sum(is.na(data))

## Omit na rows
data<- na.omit(data)

## Adjust data for analysis
data.feature = data[, -c(1, 11)]

## log transformation to 
data.logged = data.frame(apply(data.feature, 2, function(x) log10(as.numeric(x))))
hist(data.logged$clump_thickness, 5)

data.good = data.frame(clump_thickness=data.logged$clump_thickness, bare_nucleoli=data.feature$bare_nucleoli, 
                       size_uniformity = data.feature$size_uniformity, shape_uniformity = data.feature$shape_uniformity,
                       marginal_adhesion = data.feature$marginal_adhesion, epithelial_size = data.feature$epithelial_size,
                       bland_chromatin = data.feature$bland_chromatin, normal_nucleoli = data.feature$normal_nucleoli,
                       mitoses = data.feature$mitoses)
                      
## (4) T-test
## epithelial size
## Class: (2 for benign, 4 for malignant)
#epiData = data[which(data$class==4),1]
#epiNoData = data[which(data$class==2),1]


##conduct t-test
#data_ttest = t.test(epiData, epiNoData, paired=F, two.sided = two.sided)
#data_ttest <-t.test(data$epithelial_size~data$class)


data_CTtest <- t.test(data$clump_thickness ~data$class)
benignData = data[which(data$class == 2), 1]
malignantData = data[which(data$class == 4), 1]
data_CTtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_CTtest

data_SUtest <- t.test(data$size_uniformity ~data$class)
benignData = data[which(data$class == 2), 2]
malignantData = data[which(data$class == 4), 2]
data_SUtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_SUtest

data_HUtest <- t.test(data$shape_uniformity ~data$class)
benignData = data[which(data$class == 2), 3]
malignantData = data[which(data$class == 4), 3]
data_HUtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_HUtest

data_MAtest <- t.test(data$marginal_adhesion ~data$class)
benignData = data[which(data$class == 2), 4]
malignantData = data[which(data$class == 4), 4]
data_MAtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_MAtest

data_EStest <- t.test(data$epithelial_size ~data$class)
benignData = data[which(data$class == 2), 5]
malignantData = data[which(data$class == 4), 5]
data_EStest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_EStest

#
data_BNtest <- t.test(data$bare_nucleoli ~data$class)
benignData = data[which(data$class == 2), 6]
malignantData = data[which(data$class == 4), 6]
data_BNtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_BNtest
#
data_BCtest <- t.test(data$bland_chromatin ~data$class)
benignData = data[which(data$class == 2), 7]
malignantData = data[which(data$class == 4), 7]
data_BCtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_BCtest

data_NNtest <- t.test(data$normal_nucleoli ~data$class)
benignData = data[which(data$class == 2), 8]
malignantData = data[which(data$class == 4), 8]
data_NNtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_NNtest

data_Mtest <- t.test(data$mitoses ~data$class)
benignData = data[which(data$class == 2), 9]
malignantData = data[which(data$class == 4), 9]
data_Mtest = t.test(benignData, malignantData, paired = F, two.sided = two.sided)
data_Mtest

head(data)


#incrementally add one more feature and build a new classifier. Repeat this procedure till 5 features are used. 
#For each set of selected features, varying the K and compute the prediction performance of 
#the corresponding classifier. Visualize the results.

gData <- data[c("clump_thickness", "class")]
gData <- data[c("clump_thickness", "shape_uniformity", "class")]
gData <- data[c("clump_thickness", "shape_uniformity", "marginal_adhesion", "class")]
gData <- data[c("clump_thickness", "shape_uniformity", "marginal_adhesion", "epithelial_size", "class")]
gData <- data[c("clump_thickness", "shape_uniformity", "marginal_adhesion", "epithelial_size", "mitoses", "class")]


#Split dataset into train test using Caret
#stratified random sampling

set.seed(101)
# createDataPartition(y, times = 1, p = 0.5, list = TRUE, groups = min(5,length(y)))
# y is vector outcome, times is number of partitions to create, p is % of data that goes into training.
split <- createDataPartition(c(gData$shape_uniformity, gData$class), p = 0.8, list = F)
data.train <- data[split, ]
data.test <- data[-split, ]

str(data.train)
str(data.test)


## Scale train and test separately with out caret (see above for caret)

#train.scale = as.data.frame(scale(data.train[0,-9]))
#train.scale$class = data.train$class
#str(train.scale)

#test.scale = as.data.frame(scale(data.test[0,-9]))
#test.scale$class = data.test$class
#str(test.scale)


## Task 2
## Fit knn grid
## KNN modeling using caret
## Find the best KNN
grid = expand.grid(.k=seq(2,20, by=1))
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed<-100
metric <- "Accuracy"
preProcess=c("center","scale")  ## data will scale around the center


#KNN
# class
set.seed(seed)
fit.knn <- train(factor(class)~., data = data.test, method = "knn", metric=metric,
                preProc = preProcess, trControl = control, tuneGrid = grid)

#plot knn fit
## Plotting yields Number of Neighbors vs accuracy (based on repeated cross val)
plot(fit.knn)
fit.knn



## Testing
knnPredict <- predict(fit.knn, newdata = data.test)

#Get the confusion matrix to see accuracy value and other parameters values
confusionMatrix(knnPredict,factor(data.test$class, levels = c(2,4)))

#plot ROC
knnPredict <- predict(fit.knn, newdata = data.test, type = "prob")
knnROC <- roc(data.test$class, knnPredict[,1])
knnROC
plot(knnROC, type = "S", print.thres = 0.5)

