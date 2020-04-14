library(tidyverse)
library(tensorflow)
library(reticulate)
library(tfdatasets)

source("scripts/prepare_input.R")
source("scripts/model.R")

tfa <- import("tensorflow_addons")

# Config --------------------------------------------------------------

max_len <- 256L
batch_size <- 16L
n_samples <- 2048 * 4

# Load the data -------------------------------------------------------
set.seed(39839)
imdb_data <- read_csv("data/imdb_data.csv")

train <- imdb_data %>%
  filter(dataset == "train") %>% 
  sample_n(n_samples)

test <- imdb_data %>%
  filter(dataset == "test") %>% 
  sample_n(n_samples)

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
optimizer <- tf$keras$optimizers$Adam(lr = 2e-5)
#optimizer <- tfa$optimizers$LAMB(learning_rate = 1e-5)
model$compile(optimizer, loss = "binary_crossentropy", metrics = list("accuracy"))

# lr_schedule_fn <- function(epoch, lr){

#   lr_start = 1e-5
#   lr_max = 3e-5
#   lr_min = 1e-5
#   lr_rampup_epochs = 3
#   lr_sustain_epochs = 0
#   lr_exp_decay = .87
  
#   if (epoch < lr_rampup_epochs) {
#     lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
#   } else if (epoch < lr_rampup_epochs + lr_sustain_epochs) {
#     lr = lr_max
#   } else {
#     lr = (lr_max - lr_min) * lr_exp_decay^(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
#   }
  
#   return(lr)
  
# }

#lr_schedule <- tf$keras$callbacks$LearningRateScheduler(lr_schedule_fn, verbose = 1L)

train_history <- model$fit(
  train_ds,
  #steps_per_epoch = 10L,
  validation_data = test_ds,
  epochs = 4L
)
