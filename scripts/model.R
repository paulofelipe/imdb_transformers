library(reticulate)
library(tensorflow)

transformers <- import("transformers")

transformer_layer <- transformers$TFBertModel$from_pretrained("bert-base-cased")
#transformer_layer$trainable <- FALSE

build_model <- function(
  transformer = transformer_layer,
  loss = "binary_crossentropy",
  max_len = 512L
){

  input_word_ids <- tf$keras$Input(
    shape = max_len,
    dtype = tf$int32,
    name = "input_words_ids"
  )
  pooled_output <- transformer(input_word_ids)[[2]]
  #x <- tf$keras$layers$Dropout(0.05)(pooled_output)
  #x <- tf$keras$layers$Dense(units = 1028L, activation = "relu")(pooled_output)
  #x <- tf$keras$layers$Dropout(0.05)(x)
  out <- tf$keras$layers$Dense(units = 1L, activation = "sigmoid")(pooled_output)

  model <- tf$keras$Model(inputs = input_word_ids, outputs = out)
  model$compile(tf$keras$optimizers$Adam(lr = 1e-5), loss = loss, metrics = list("accuracy"))
  model
}

