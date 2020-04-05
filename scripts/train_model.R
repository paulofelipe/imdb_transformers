library(tidyverse)
library(tensorflow)
library(reticulate)
library(tfdatasets)

source("scripts/prepare_input.R")
source("scripts/model.R")

# Load the data -------------------------------------------------------
imdb_data <- read_csv("data/imdb_data.csv")

train <- imdb_data %>%
  filter(dataset == "train")

test <- imdb_data %>%
  filter(dataset == "test") 

# Prepare inputs and targets ---------------------------------------------
x_train <- fast_encode(
  texts = train$review,
  max_len = 128L
)

x_test <- fast_encode(
  texts = test$review,
  max_len = 128L
)

y_train <- train %>%
  mutate(y = +(sentiment == "pos")) %>%
  select(y) %>%
  data.matrix()

y_test <- test %>%
  mutate(y = +(sentiment == "pos")) %>%
  select(y) %>%
  data.matrix()

# TF datasets --------------------------------------------------------------

train_ds <- tensor_slices_dataset(tensors = list(x_train, y_train)) %>% 
  dataset_shuffle(2048) %>% 
  dataset_batch(32)

test_ds <- tensor_slices_dataset(tensors = list(x_test, y_test)) %>% 
  dataset_shuffle(2048) %>% 
  dataset_batch(32)

# Model --------------------------------------------------------------------

model <- build_model(max_len = 128L)
model$summary()

train_history <- model$fit(
  train_ds,
  validation_data = test_ds,
  epochs = 3L
)
