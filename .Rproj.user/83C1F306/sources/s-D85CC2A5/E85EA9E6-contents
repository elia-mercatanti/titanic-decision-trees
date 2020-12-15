library(VIM)
library(ggplot2)
library(naniar)
library(rpart)
library(plyr)
library(dplyr)
library(timereg)
library(caTools)
library(tree)
library(rpart)
library(rpart.plot)
library(rattle)
library(party)
library(evtree)
library(randomForest)
library(e1071)
library(caret)
library(pROC)

############################################################################
############################ Dataset Setup #################################
############################################################################

# Load Dataset
original_titanic_train <- read.csv("dataset/titanic-train.csv", na.strings="")
original_titanic_test <- read.csv("dataset/titanic-test.csv", na.strings="")

# Print Dataset
head(original_titanic_train)

summary(original_titanic_train)

# Create survived column in test set
original_titanic_test$Survived <- NA

# Bind test and train data to create new variables in both sets
titanic_full <- rbind(original_titanic_train, original_titanic_test)
str(titanic_full)

# Set some feature as factors
titanic_full$Survived <- as.factor(titanic_full$Survived)
titanic_full$Pclass <- as.factor(titanic_full$Pclass)
titanic_full$Sex <- as.factor(titanic_full$Sex)
titanic_full$Embarked <- as.factor(titanic_full$Embarked)

############################################################################
############################ Missing values dataset  #######################
############################################################################

# Missing values
miss <- sapply(titanic_full[-2], function(x) sum(is.na(x)))
miss
aggr(titanic_full[ , -2], sortVars=TRUE, cex.axis=.7)

# Percentage of missing values between male and female
gg_miss_fct(x = titanic_full[ , -2], fct = Sex) + labs(title = "Missing Values in Titanic Dataset Between Male and Female")

# Percentage of missing information across passenger “class”
gg_miss_fct(x = titanic_full[ , -2], fct = Pclass) + labs(title = "Missing Values in Titanic Dataset Across Passenger “class”")

# Percentage of missing information across passenger siblings
gg_miss_fct(x = titanic_full[ , -2], fct = SibSp ) + labs(title = "Missing Values in Titanic Dataset Across Passenger Siblings")

# Delete Cabin
titanic_full <- titanic_full[,-11]

# Replace missing embarked values
titanic_full$Embarked[c(62,830)] = "S"

# Replace missing fare value 
titanic_full$Fare[1044] <- median(titanic_full$Fare, na.rm = TRUE)

# Iterate over Sex (male or female) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.
for (i in 1:nlevels(titanic_full$Sex)) {
  for (j in 1:length(unique(titanic_full$Pclass))) {
    guess_age <- round(median(subset(titanic_full, Sex == titanic_full$Sex[i] & Pclass == j, select=c(Age))$Age, na.rm = TRUE))
    titanic_full$Age[titanic_full$Sex == titanic_full$Sex[i] & titanic_full$Pclass == j & is.na(titanic_full$Age)] <- guess_age
  }
}

############################################################################
##################### Data exploration and visualization ###################
############################################################################

# Sex x Age plot
means <- ddply(titanic_full, "Sex", summarise, Avg.Age=mean(Age))
means
plot <- ggplot(titanic_full, aes(x=Age, fill=Sex)) + geom_density(alpha=0.3) + scale_fill_manual( values = c("red","blue"))
plot + geom_vline(data=means, aes(xintercept=Avg.Age, color=Sex), linetype="dashed") + ggtitle("Density plot for Sex x Age")

# Survival x Age plot
means <- ddply(titanic_full[1:891, ], "Survived", summarise, Avg.Age=mean(Age))
means
plot <- ggplot(titanic_full[1:891, ], aes(x=Age, fill=Survived)) + geom_density(alpha=0.3) + scale_fill_manual( values = c("red","darkgreen"))
plot + geom_vline(data=means, aes(xintercept=Avg.Age, color=Survived), linetype="dashed") + ggtitle("Density plot for Survival x Age")

# Sex x Survival plot
counts <- table(titanic_full[1:891, ]$Survived, titanic_full[1:891, ]$Sex)
bp <- barplot(counts, main="Sex x Survival", xlab="Sex", ylab="Number of Passengers", col=c("red","darkgreen"), legend = c("Not Survived", "Survived"), args.legend = list(x = 'topleft'))
text(bp, counts[1,]-40, labels = counts[1, ])
text(bp, counts[1,]+60, labels = counts[2, ])

# Sex x Age x Survival (Survived) plot
data <- titanic_full[1:891, ][titanic_full[1:891, ]$Survived == 1, ]
means <- ddply(data, "Sex", summarise, Avg.Age=mean(Age))
means
plot <- ggplot(data, aes(x=Age, fill=Sex)) + geom_density(alpha=0.3) + scale_fill_manual( values = c("red","blue"))
plot + geom_vline(data=means, aes(xintercept=Avg.Age, color=Sex), linetype="dashed") + ggtitle("Density plot of Age x Sex for Surviving Passengers")

# Sex x Age x Survival (Not Survived) plot
data <- titanic_full[1:891, ][titanic_full[1:891, ]$Survived == 0, ]
means <- ddply(data, "Sex", summarise, Avg.Age=mean(Age))
means
plot <- ggplot(data, aes(x=Age, fill=Sex)) + geom_density(alpha=0.3) + scale_fill_manual( values = c("red","blue"))
plot + geom_vline(data=means, aes(xintercept=Avg.Age, color=Sex), linetype="dashed") + ggtitle("Density plot of Age x Sex for Non-Survivors Passengers")

# Class x Survival plot
counts <- table(titanic_full[1:891, ]$Survived, titanic_full[1:891, ]$Pclass)
bp <- barplot(counts, main="Class x Survival", xlab="Pclass", ylab="Number of Passengers", col=c("red","darkgreen"), legend = c("Not Survived", "Survived"), args.legend = list(x = 'topleft'))
text(bp, counts[1,]-40, labels = counts[1, ])
text(bp, counts[1,]+60, labels = counts[2, ])

# Class x Age x Survival (Survived) plot
data <- titanic_full[1:891, ][titanic_full[1:891, ]$Survived == 1, ]
means <- ddply(data, "Pclass", summarise, Avg.Age=mean(Age))
means
plot <- ggplot(data, aes(x=Age, fill=Pclass)) + geom_density(alpha=0.3) + scale_fill_manual(values = c("yellow","blue", "red"))
plot + geom_vline(data=means, aes(xintercept=Avg.Age, color=Pclass), linetype="dashed") + ggtitle("Density plot of Age x Pclass for Surviving Passengers")

# Class x Age x Survival (Not Survived) plot
data <- titanic_full[1:891, ][titanic_full[1:891, ]$Survived == 0, ]
means <- ddply(data, "Pclass", summarise, Avg.Age=mean(Age))
means
plot <- ggplot(data, aes(x=Age, fill=Pclass)) + geom_density(alpha=0.3) + scale_fill_manual(values = c("yellow","blue", "red"))
plot + geom_vline(data=means, aes(xintercept=Avg.Age, color=Pclass), linetype="dashed") + ggtitle("Density plot of Age x Pclass for Non-Survivors Passengers")

# Embarked x Survival
counts <- table(titanic_full[1:891, ]$Survived, titanic_full[1:891, ]$Embarked)
bp <- barplot(counts, main="Embarked x Survival", xlab="Pclass", ylab="Number of Passengers", col=c("red","darkgreen"), legend = c("Not Survived", "Survived"), args.legend = list(x = 'topleft'))
text(bp, counts[1,]-20, labels = counts[1, ])
text(bp, counts[1,]+20, labels = counts[2, ])

############################################################################
############################ Feature Extraction ############################
############################################################################

# Take title from name
titanic_full$Title <- gsub('(.*, )|(\\..*)', '', titanic_full$Name)

# Show title counts by sex
table(titanic_full$Sex, titanic_full$Title)

# Re-assign female categories 
titanic_full$Title[titanic_full$Title == 'Mlle' | titanic_full$Title == 'Ms'] <- 'Miss' 
titanic_full$Title[titanic_full$Title == 'Mme']  <- 'Mrs' 

# Concatenate rare titles, potential proxi for high spcoa; standing 
Other <- c('Dona', 'Dr', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir')
titanic_full$Title[titanic_full$Title %in% Other]  <- 'Other'

# Set Title as factor
titanic_full$Title <- as.factor(titanic_full$Title)

# Show title counts by sex again
table(titanic_full$Sex, titanic_full$Title)

# Embarked x Survival
counts <- table(titanic_full[1:891, ]$Survived, titanic_full[1:891, ]$Title)
bp <- barplot(counts, main="Title x Survival", xlab="Title", ylab="Number of Passengers", col=c("red","darkgreen"), legend = c("Not Survived", "Survived"), args.legend = list(x = 'topleft'))
text(bp, counts[1,]-10, labels = counts[1, ])
text(bp, counts[1,]+40, labels = counts[2, ])

# Combine siblings + parents/children 
titanic_full$FamilySize <- titanic_full$SibSp + titanic_full$Parch + 1

#Use ggplot2 to visualize the relationship between family size & survival
ggplot(titanic_full[1:891,], aes(x = FamilySize, fill = Survived)) + geom_bar(stat='count', position='dodge') +
       scale_x_continuous(breaks=c(1:11)) + labs(x = 'Family Size')

# Create new feature IsAlone from FamilySize
titanic_full$IsAlone <- ifelse(titanic_full$FamilySize == 1, 1, 0)

# Set IsAlone as factor
titanic_full$IsAlone <- as.factor(titanic_full$IsAlone)

# Fare Groups
groups = qcut(titanic_full$Fare, 4)
levels(groups)
titanic_full$Fare <- groups
levels(titanic_full$Fare) <- c("1", "2", "3", "4")

# Removing unnecessary features
titanic_full$PassengerId <- NULL
titanic_full$Name <- NULL
titanic_full$SibSp <- NULL
titanic_full$Parch <- NULL
titanic_full$Ticket <- NULL

# Print final dataset
head(titanic_full)

############################################################################
############################ Decision Tree Models ##########################
############################################################################

# Let’s ensure these results can be replicated
set.seed(2020)

titanic <- titanic_full[1:891,] # survived data

# Assess whether any data missing in our training set
sapply(titanic, function(x) sum(is.na(x)))

# Split the training set to get an approximation of the models performance
split = sample.split(titanic$Survived, SplitRatio = 0.8)
train = subset(titanic, split == TRUE)
test = subset(titanic, split == FALSE)

# Outcome variable
dependent_var <- "Survived"
# Predictor variables Model
independent_vars <- c("Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone")

# Create the formula string 
formula <- paste(dependent_var, "~", paste(independent_vars, collapse = " + "))

############################################################################
############################ TREE Package ##################################
############################################################################

# Let’s ensure these results can be replicated
set.seed(2020)

# TREE with entropy
tree_model <- tree(formula, data = train, split = "deviance")
plot(tree_model, type = "uniform", main = "TREE Package with Entropy")
text(tree_model, pretty=5, cex=1)

# Predict Survived
tree_prediction <- predict(tree_model, test, type = "class")
confusion_matrix <- table(tree_prediction, test$Survived)
confusion_matrix
error <- mean(test$Survived != tree_prediction) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# Let’s ensure these results can be replicated
set.seed(2020)

# Search best size (leaf nodes) for the tree
tree_cv <- cv.tree(tree_model, K = 10, FUN = prune.misclass)
tree_cv
plot(tree_cv$size, tree_cv$dev, type="b", xlab="size", ylab="cverror", main="TREE Package with Entropy - Choosing Size of the Tree", cex=2)
best_size <- tail(tree_cv$size[which(tree_cv$dev==min(tree_cv$dev))], n=1)

# Prune tree
pruned_tree <- prune.misclass(tree_model, best=best_size)
plot(pruned_tree, type = "uniform", main="TREE Package with Entropy - Pruned")
text(pruned_tree, pretty=5, cex=1)

# Prune tree predictions
pruned_tree_predictions <- predict(pruned_tree, test, type = "class")
pruned_confusion_matrix <- table(pruned_tree_predictions, test$Survived) #confusion Matrix
pruned_confusion_matrix
error <- mean(test$Survived != pruned_tree_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# Let’s ensure these results can be replicated
set.seed(2077)

# Decision tree with Gini Index
tree_model <- tree(formula, data = train, split = "gini")
plot(tree_model, type = "uniform",  main="TREE Package with Gini")
text(tree_model, pretty=5, cex=0.6)

# Predict Survived
tree_prediction <- predict(tree_model, test, type = "class")
confusion_matrix <- table(tree_prediction, test$Survived)
confusion_matrix
error <- mean(test$Survived != tree_prediction) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# Search best size (leaf nodes) for the tree
tree_cv <- cv.tree(tree_model, K = 10, FUN = prune.misclass)
tree_cv
plot(tree_cv$size, tree_cv$dev, type="b", xlab="size", ylab="cverror", main="TREE Package with Gini - Choosing Size of the Tree", cex=2)
best_size <- tail(tree_cv$size[which(tree_cv$dev==min(tree_cv$dev))], n=1)

# Prune tree
pruned_tree <- prune.misclass(tree_model, best=best_size)
plot(pruned_tree, type = "uniform", main="TREE Package with Gini - Pruned")
text(pruned_tree, pretty=5, cex=1)

# Prune tree predictions
pruned_tree_predictions <- predict(pruned_tree, test, type = "class")
pruned_confusion_matrix <- table(pruned_tree_predictions, test$Survived) #confusion Matrix
pruned_confusion_matrix
error <- mean(test$Survived != pruned_tree_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

############################################################################
############################ RPART Package #################################
############################################################################

# Let’s ensure these results can be replicated
set.seed(2020)

# RPART tree with Gini
rtree_model <- rpart(formula, data = train, method = "class", parms = list(split='gini'))
fancyRpartPlot(rtree_model, main="RPART Package with Gini", sub = "")

# Predict Survived
rtree_model_predictions <- predict(rtree_model, test, type = "class")
confusion_matrix <- table(rtree_model_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != rtree_model_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# Search best index
plotcp(rtree_model, cex=2)

# Pruning the tree
pruned_rtree <- prune.rpart(rtree_model, cp=rtree_model$cptable[which.min(rtree_model$cptable[,"xerror"]),"CP"])
fancyRpartPlot(pruned_rtree, main="RPART Package with Gini - Pruned", sub = "")

# Prune tree predictions
pruned_rtree_predictions <- predict(pruned_rtree, test, type = "class")
confusion_matrix <- table(pruned_rtree_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != pruned_rtree_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# Let’s ensure these results can be replicated
set.seed(2077)

# RPART tree with Entropy
rtree_model <- rpart(formula, data = train, method = "class", parms = list(split='information'))
fancyRpartPlot(rtree_model, main="RPART Package with Entropy", sub = "")

# Predict Survived
rtree_model_predictions <- predict(rtree_model, test, type = "class")
confusion_matrix <- table(rtree_model_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != rtree_model_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# Search best index
plotcp(rtree_model, cex=2)

# Pruning the tree
pruned_rtree <- prune.rpart(rtree_model, cp=rtree_model$cptable[which.min(rtree_model$cptable[,"xerror"]),"CP"])
fancyRpartPlot(pruned_rtree, main="RPART Package with Entropy - Pruned", sub = "")

# Prune tree predictions
pruned_rtree_predictions <- predict(pruned_rtree, test, type = "class")
confusion_matrix <- table(pruned_rtree_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != pruned_rtree_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

############################################################################
############################ PARTY Package #################################
############################################################################

# Let’s ensure these results can be replicated
set.seed(2020)

# Conditional Inference Tree - Party
party_tree_model <- ctree(as.formula(formula), data = train)
plot(party_tree_model, type = "simple", main="Conditional Inference Tree")

# Predict Survived
party_tree_predictions <- predict(party_tree_model, test)
confusion_matrix <- table(party_tree_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != party_tree_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

#############################################################################
############################ EVTREE Package #################################
#############################################################################

# Let’s ensure these results can be replicated
set.seed(2020)

# Evolutionary Learning of Globally Optimal Trees
ev_tree_model <- evtree(formula, data = train)
plot(ev_tree_model, type = "simple", main="Evolutionary Learning of Globally Optimal Tree")

# Predict Survived
ev_tree_predictions <- predict(ev_tree_model, test)
confusion_matrix <- table(ev_tree_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != ev_tree_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

#############################################################################
######################### RANDOMFOREST Package ##############################
#############################################################################

# Let’s ensure these results can be replicated
set.seed(2020)

# Random Forest Model
random_forest_model <- randomForest(as.formula(formula), data = train, importance=TRUE)
print(random_forest_model)

# Plot errors
plot(random_forest_model, main = "Random Forest Model - Errors Plot")

# Plot variables importance
importance(random_forest_model)
varImpPlot(random_forest_model, main = "Random Forest - Variable Importance", cex=1.2)

# Random forest predictions
random_forest_predictions <- predict(random_forest_model, test)
confusion_matrix <- table(random_forest_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != random_forest_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# ROC Curve and print AUROC
random_forest_roc <- roc(train$Survived, random_forest_model$votes[,2])
plot(random_forest_roc, main="Random Forest - ROC Curve")
paste('AUC =', auc(random_forest_roc))

# Let’s ensure these results can be replicated
set.seed(2077)

# Random Forest with tuned parameters
tuned_random_forest <- tune(randomForest, train.x = as.formula(formula), data = train, validation.x = test, ranges = list(mtry = 1:(ncol(train)-1), ntree = seq(500, 2000, 500)), importance=TRUE)
plot(tuned_random_forest)

# Search best model
best_model_random_forest <- tuned_random_forest$best.model
tuned_random_forest$best.parameters
print(best_model_random_forest)

# Plot errors
plot(best_model_random_forest, main = "Random Forest Tunded Model - Errors Plot")

# Variable importance
importance(best_model_random_forest)
varImpPlot(best_model_random_forest, main = "Random Forest Tunded Model - Variable Importance", cex=1.2)

# Accuracy
random_forest_predictions <- predict(best_model_random_forest, test)
confusion_matrix <- table(random_forest_predictions, test$Survived) #confusion Matrix
confusion_matrix
error <- mean(test$Survived != random_forest_predictions) # Misclassification error
paste('Accuracy =', round(1 - error, 5))

# ROC Curve
random_forest_roc <- roc(train$Survived, best_model_random_forest$votes[,2])
plot(random_forest_roc, main="Random Forest Tunded Model - ROC Curve")
paste('AUC =', auc(random_forest_roc))
