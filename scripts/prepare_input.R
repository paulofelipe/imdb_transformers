library(reticulate)

transformers <- import("transformers")
tokenizers <- import("tokenizers")
bert_wordpiece_tokenizer <- tokenizers$BertWordPieceTokenizer

tokenizer <- transformers$BertTokenizer$from_pretrained("bert-base-cased")

if (!dir.exists("tokenizer")) {
  dir.create(path = "tokenizer")
}

tokenizer$save_pretrained("tokenizer/")

fast_tokenizer <- bert_wordpiece_tokenizer(
  "tokenizer/vocab.txt",
  lowercase = FALSE
)

fast_encode <- function(texts, tokenizer = fast_tokenizer, max_len = 512L){
  
  ids <- matrix(0L, nrow = length(texts), ncol = max_len)

  tokenizer$enable_truncation(max_length = max_len)
  tokenizer$enable_padding(max_length = max_len)

  for(i in 1:length(texts)){
    ids[i,] <- tokenizer$encode(texts[i])$ids
  }

  ids

}