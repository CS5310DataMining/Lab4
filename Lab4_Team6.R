# Lab 4
# Team: 6
# Team Members:
#   Melvin Zaldivar - Members contribution: 33.33%
#   Rahim Abdulmalik - Members contribution: 33.33%
#   Raul Beiza - Members contribution: 33.33%

# Due Date: February 23, 2020

#--------------------------------------------
# Step 1 Exploring and preparing the data
#--------------------------------------------

# Read CSV file
sms_raw <- read.csv("sms_spam.csv", na.strings = NA)

str(sms_raw)

# Convert type from character vector to factor
sms_raw$type <- factor(sms_raw$type)

str(sms_raw$type)
table(sms_raw$type)

#---------------------------------------------
# Step 2a Data prepartion - Cleaning and standardizing
# text data
#----------------------------------------------

# Install tm package to r
install.packages("tm")

# Load tm package
library(tm)

# Use VectorSource() reader function to create a source
# object from the data

sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# Print corpus to see containing data
print(sms_corpus)

# Standardizing the messages to only use lowercase 
# characters

sms_corpus_clean <- tm_map(sms_corpus,
                           content_transformer(tolower))

# Strip all numbers from the corpus
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# Remove any words in the text messages defined in stopword() 
# function 
sms_corpus_clean <- tm_map(sms_corpus_clean,
                           removeWords, stopwords())

# Eliminate any punctuation in the text messages
sms_corpus_clean <- tm_map(sms_corpus_clean,
                           removePunctuation)

# Installing SnowballC package
install.packages("SnowballC")

# Loading SnowballC package
library(SnowballC)

# Applying wordStem function to text messages
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

# Removing addtional white spaces from our cleaned text messages
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

# Check our cleaned messges
as.character(sms_corpus_clean[[1]])

# Splitting text documents and creating sparse matrix

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm

#-------------------------------------------------------
# Step 2b - Creating training and test datasets
#------------------------------------------------------

# Dividing data into two portions; Training and test datasets
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]

# Creating labels for training and test matrices
sms_train_labels <- sms_raw[1:4169,]$type
sms_test_labels <- sms_raw[4170:5559,]$type

# Installing wordcloud to help gauge our successfulnes
install.packages("wordcloud")

# Load word cloud package
library(wordcloud)

# Comparing the clouds for SMS spam and ham
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")

# Creat word clouds for spam and ham
wordcloud(spam$text, max.words = 20,  scale = c(3,0.5))
wordcloud(ham$text, max.words = 50, scale=c(3,0.5))

#--------------------------------------------------
# Step 2c - Data prepartion cont., creating indicator 
# features for frequent words
#------------------------------------------------

# Find frequent words to training Naive Bayes
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

# Filtering out DTM to include only the terms appearing 
# in the frequent word vector
sms_dtm_freq_train <- sms_dtm_train[,sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[,sms_freq_words]

# Creating fuction to convert caterofical variable to simples yes or no
convert_counts <- function(x){
  x <- ifelse(x>0, "Yes", "No")
}

# We beed to apply our defined function to each columns in our sparse matrix
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                  convert_counts)

#-------------------------------------------------
# Step 3 - Traing a model on the data
#-------------------------------------------------

# Install Naive Bayes implementation package
install.packages("e1071")

# Load the Naive Bayes implementation package
library(e1071)

# Build our model on the sms_train matrix
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

#-----------------------------------------------------
# Step 4 - Evaluating model performace
#----------------------------------------------------

# Evalute SMS classifier to predict on our test messages
sms_test_pred <- predict(sms_classifier, sms_test)

# Compare predictions to the true values, load CrossTables() 
# function first

library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

#--------------------------------------------------
# Step 5 - Improving model performace
#--------------------------------------------------

# Build a Naive Bayes models with Laplace Estimator
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)

# Make predictions with new Naive Bayes model
sms_test_pred2 <- predict(sms_classifier2, sms_test)

# Compare predicted classes to actual classification
CrossTable(sms_test_pred2, sms_test_labels,
          prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
          dnn = c('predicted', 'actual'))
