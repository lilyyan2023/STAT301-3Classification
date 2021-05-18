# Load packages
library(tidyverse)
library(tidymodels)

# set seed
set.seed(123)

# load data
loan_train <- read_csv("train.csv")
# split data
