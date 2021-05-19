# Load packages
library(tidyverse)
library(tidymodels)
library(naniar)
library(corrplot)
library(corrr)
library(glmnet)
# set seed
set.seed(123)

# load data
loan_train <- read_csv("stat-301-3-classification-2021-loan-repayment/train.csv")
loan_test <- read_csv("stat-301-3-classification-2021-loan-repayment/test.csv")
dim(loan_train)
dim(loan_test)
loan_train <- loan_train %>% 
  mutate(hi_int_prncp_pd_f = factor(hi_int_prncp_pd))

# split data
loan_folds <- vfold_cv(data = loan_train, v = 5, repeats = 3)
# define recipe
loan_recipe <- 
  recipe(hi_int_prncp_pd_f ~ int_rate + loan_amnt +
           out_prncp_inv + application_type + grade + initial_list_status + term,
         data = loan_train) %>%  
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

# elastic net model
en_model <- logistic_reg(mode = "classification",
                         penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet")

# set-up tuning grid ----
en_params <- parameters(en_model) %>% 
  update(mixture = mixture(range = c(0, 1)))

# define tuning grid
en_grid <- grid_regular(en_params, levels = 5)

# workflow ----
en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(loan_recipe)

# Tuning/fitting ----
en_tuned <- en_workflow %>% 
  tune_grid(loan_folds, grid = en_grid)

write_rds(en_tuned, "en_results.rds")




