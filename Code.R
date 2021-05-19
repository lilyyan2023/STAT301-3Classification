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
miss_var_summary(loan_train)
skimr::skim_without_charts(loan_train)
loan_train <- loan_train %>% 
  mutate(hi_int_prncp_pd_f = factor(hi_int_prncp_pd))
loan_train %>%
  select(hi_int_prncp_pd, acc_now_delinq, acc_open_past_24mths, annual_inc,
         avg_cur_bal, bc_util) %>% 
  cor() %>% 
  corrplot()
# based on the corrplot, hi_int_prncp_pd doesn't have high correlation with other variables.

loan_train %>% 
  select(hi_int_prncp_pd, delinq_2yrs, delinq_amnt, dti, int_rate, loan_amnt) %>% 
  cor() %>% 
  corrplot()
# hr_int_prncp_pd has a comparatively high positive correlation with int_rate, loan_amnt
# and int_rate has a comparatively high positive correlation with dti. 

loan_train %>% 
  select(hi_int_prncp_pd, mort_acc, num_sats, num_tl_120dpd_2m, num_tl_90g_dpd_24m, num_tl_30dpd,
         out_prncp_inv, pub_rec, pub_rec_bankruptcies, tot_coll_amt, tot_cur_bal,
         total_rec_late_fee) %>% 
  cor() %>% 
  corrplot()

# hi_int_prncp_pd has a high positive relationship with out_prncp_inv, mort_acc has
# a positive correlation with num_sats, tot_cur_bal has a strong positive correlation with 
# mort_acc, tot_cur_ball has a positive correlation with num_sats, pub_rec has a strong
# positive correlation with pub_rec_bankruptcies

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = application_type)) +
  geom_bar() # application type yes

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = emp_length)) +
  geom_bar(position = "dodge") # emp_length no

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = grade)) +
  geom_bar(position = "dodge") # grade yes

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = home_ownership)) +
  geom_bar(position = "dodge") # home_ownership no

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = initial_list_status)) +
  geom_bar() # initial_list_status yes

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = purpose)) +
  geom_bar(position = "dodge") # purpose no

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = term)) +
  geom_bar() # term yes

ggplot(loan_train, aes(x = hi_int_prncp_pd_f, fill = verification_status)) +
  geom_bar(position = "dodge") # verification_status no

# split data
loan_folds <- vfold_cv(data = loan_train, v = 5, repeats = 3)

# define recipe
# get rid of id. addr_state ? 
#earliest_cr_line -> date conversion, no emp_title, no last_credit_pull_d -> date conversion
# no sub_grade


loan_recipe <- 
  recipe(hi_int_prncp_pd_f ~ int_rate,  loan_amnt, 
         out_prncp_inv, application_type, grade, initial_list_status, term,
         data = loan_train) #%>%  
  #step_dummy(all_nominal(), one_hot = TRUE) %>% 
  #step_normalize(all_predictors())

prep(loan_recipe) %>% 
  bake(new_data = NULL)

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

