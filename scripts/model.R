library(reticulate)
library(tensorflow)
library(R6)

transformers <- import("transformers")

transformer_layer <- transformers$TFAutoModel$from_pretrained("distilbert-base-cased")
#transformer_layer$trainable <- FALSE

build_model <- function(
  transformer = transformer_layer,
  loss = "binary_crossentropy",
  max_len = 512L,
  steps = 2L
){

  input_word_ids <- tf$keras$Input(
    shape = max_len,
    dtype = tf$int32,
    name = "input_words_ids"
  )
  seq_output <- transformer(input_word_ids)[[1]]
  # pooled_output <- tf$keras$layers$GlobalAvgPool2D(seq_output[[1]])
  # x <- tf$keras$layers$Dropout(0.05)(pooled_output)
  # x <- tf$keras$layers$Dense(units = 1028L, activation = "relu")(pooled_output)
  # x <- tf$keras$layers$Dropout(0.05)(x)
  x <- tf$keras$layers$GlobalAvgPool1D()(seq_output)
  x <- tf$keras$layers$Dropout(0.5)(x)
  out <- tf$keras$layers$Dense(units = 1L, activation = "sigmoid")(x)

  model <- tf$compat$v1$keras$Model(inputs = input_word_ids, outputs = out)

  model
}