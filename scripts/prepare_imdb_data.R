library(tidyverse)

# Download the dataset -----------------------------------------------------
if(!file.exists("data/imdb_data.tar.gz")){
  download.file(
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    destfile = "data/imdb_data.tar.gz"
  )
}

# untar the file -----------------------------------------------------------
untar(
  tarfile = "data/imdb_data.tar.gz",
  exdir = "data"
)

# Combine the texts --------------------------------------------------------

folders <- crossing(
  dataset = c("train", "test"),
  sentiment = c("neg", "pos")
)

prepare_data <- function(dataset, sentiment) {
  arquivos <- list.files(
    file.path("data/aclImdb", dataset, sentiment),
    full.names = TRUE
  )
  map_df(arquivos, ~ {
    tibble(
      review = read_lines(.x)
    ) %>%
      mutate(
        dataset = dataset,
        sentiment = sentiment
      )
  }) %>%
    select(dataset, sentiment, review)
}

imdb_data <- pmap_df(folders, prepare_data)

write_csv(
  x = imdb_data,
  path = "data/imdb_data.csv"
)

