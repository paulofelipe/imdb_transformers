library(tidyverse)
library(tensorflow)
library(reticulate)
library(tfdatasets)

source("scripts/prepare_input.R")
source("scripts/model.R")

max_len <- 256L
batch_size <- 16L

# Load the data -------------------------------------------------------
imdb_data <- read_csv("data/imdb_data.csv")

train <- imdb_data %>%
  filter(dataset == "train") %>% 
  sample_n(5000)

test <- imdb_data %>%
  filter(dataset == "test") %>% 
  sample_n(1000)

# Prepare inputs and targets ---------------------------------------------
x_train <- fast_encode(
  texts = train$review,
  max_len = max_len
)

x_test <- fast_encode(
  texts = test$review,
  max_len = max_len
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
  #dataset_repeat() %>% 
  dataset_shuffle(512) %>% 
  dataset_batch(batch_size)

test_ds <- tensor_slices_dataset(tensors = list(x_test, y_test)) %>% 
  dataset_batch(batch_size)

# Model --------------------------------------------------------------------

model <- build_model(max_len = max_len)
model$summary()

train_history <- model$fit(
  train_ds,
  #steps_per_epoch = 10L,
  validation_data = test_ds,
  epochs = 30L
)
