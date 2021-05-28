# Load packages
library(tidyverse)
library(tidymodels)
library(naniar)
library(corrplot)
library(corrr)
library(glmnet)
library(ranger)
# set seed
set.seed(123)

# load data
loan_train <- read_csv("stat-301-3-classification-2021-loan-repayment/train.csv")
loan_test <- read_csv("stat-301-3-classification-2021-loan-repayment/test.csv")
dim(loan_train)
dim(loan_test)
miss_var_summary(loan_train)
skimr::skim_without_charts(loan_train)
loan_train <- loan_train %>% 
  mutate(hi_int_prncp_pd_f = factor(hi_int_prncp_pd))

# split data
loan_folds <- vfold_cv(data = loan_train, v = 5, repeats = 3)

# define recipe
# get rid of id. addr_state ? 
#earliest_cr_line -> date conversion, no emp_title, no last_credit_pull_d -> date conversion
# no sub_grade

loan_recipe <- 
  recipe(hi_int_prncp_pd_f ~ int_rate + loan_amnt +
         out_prncp_inv + application_type + grade + initial_list_status + term,
         data = loan_train) %>%  
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

prep(loan_recipe) %>% 
  bake(new_data = NULL)
# Define model
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
#en_tuned <- en_workflow %>% 
  #tune_grid(loan_folds, grid = en_grid)
en_tuned <- read_rds("en_results.rds")

# create another recipe
loan_recipe2 <- 
  recipe(hi_int_prncp_pd_f ~ int_rate + loan_amnt +
           out_prncp_inv + application_type + grade + initial_list_status + term,
         data = loan_train) %>%  
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  step_normalize(all_predictors())

prep(loan_recipe2) %>% 
  bake(new_data = NULL)

# random forest model
rf_model <- rand_forest(
  mode = "classification",
  mtry = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger", importance = "impurity")

# set-up tuning grid ----
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2,6)))

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = c(3,4))

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe2)

# Tuning/fitting
#rf_tuned <- rf_workflow %>% 
  #tune_grid(loan_folds, rf_grid)
rf_tuned <- read_rds("rf_results.rds")

# Metric performance
autoplot(en_tuned, metric = "roc_auc")
autoplot(en_tuned, metric = "accuracy")
select_best(en_tuned, metric = "roc_auc")
select_best(en_tuned, metric = "accuracy")
show_best(en_tuned, metric = "roc_auc")
show_best(en_tuned, metric = "accuracy")

autoplot(rf_tuned, metric = "roc_auc")
autoplot(rf_tuned, metric = "accuracy")
select_best(rf_tuned, metric = "roc_auc")
select_best(rf_tuned, metric = "accuracy")
show_best(rf_tuned, metric = "roc_auc")
show_best(rf_tuned, metric = "accuracy")

# finalize workflow
en_workflow_tuned <- en_workflow %>% 
  finalize_workflow(select_best(en_tuned, metric = "roc_auc"))

en_results <- fit(en_workflow_tuned, loan_train)

en_final_results <- en_results %>% 
  predict(new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(id)) %>% 
  select(id, .pred_class)

write_csv(en_final_results, "en_output.csv")

rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tuned, metric = "roc_auc"))

rf_results <- fit(rf_workflow_tuned, loan_train)

rf_final_results <- rf_results %>% 
  predict(new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(id)) %>% 
  select(id, .pred_class)

write_csv(rf_final_results, "rf_output.csv")
