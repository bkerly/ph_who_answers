##tidy models
### predictive modeling case study https://www.tidymodels.org/start/case-study/

# Load Libraries ----------------------------------------------------------

library(tidyverse)
library(readxl)
library(tidymodels)
library(lubridate)
library(ggthemes)
library(viridis)
library(vip)
library(missRanger)
library(glmnet)


# READ IN DATA & REFACTOR -------------------------------------------------

modeldata <- read_excel("data/modeldata.xlsx", 
                        col_types = c("text", "numeric", "text", 
                                      "date", "date", "numeric", "numeric", 
                                      "text", "text", "text", "text", "text", 
                                      "text", "text", "text", "text", "text", 
                                      "text", "numeric", "numeric", "numeric", 
                                      "numeric", "text", "text", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "text"))

summary(modeldata)

#View(modeldata)

# There are a lot of variables in modeldata that probably don't help out with the model, and others are coded weird, so I'm gonna do some transmute action and fix them (supercedes a lot of cleanup work commented out below)

data_tidy <- modeldata %>%
  transmute(
    #id = id,
    completed_interview = (completedinterviews == 1) %>% as.factor(),
    interview_initiate_date = ymd(interview_initated_date),
    days_since_covid_start = time_since_covid_start,
    age = age,
    gender = relevel(as.factor(gender), ref = "female") %>% as.numeric(),
    race = relevel(as.factor(race), ref = "black"),
    ethnicity = (ethnicity == "hispanic") %>% as.numeric(), #What if nonwhite, non male was the default? Let's try it once maybe!
    interpreter_required = (interpreter_required == "yes") %>% as.numeric(),
    #county = county,
    tract_poptotal = tract_poptotal,
    tract_popdensitysqmi = tract_popdensitysqmi,
    tract_nhl_wa = tract_nhl_wa,
    tract_white = tract_white,
    tract_minorityrace = tract_minorityrace,
    tract_englishltvw = tract_englishltvw,
    tract_laborforce = tract_laborforce,
    tract_unemployed = tract_unemployed,
    tract_healthsocialfactor = tract_healthsocialfactor,
    tract_pop_under5 = tract_pop_under5,
    tract_pop_over64 = tract_pop_over64,
    tract_lifeexpectancy = tract_lifeexpectancy,
    tract_poc = tract_poc,
    tract_lessthanhs = tract_lessthanhs,
    tract_lowincome = tract_lowincome,
    tract_disability = tract_disability,
    tract_housingcost = tract_housingcost,
    county_positivityrate = county_positivityrate,
    county_rateper100000 = county_rateper100000,
    county_deathsper100000 = 100000 * county_deaths / county_population, # Maybe it's better as a rate?
    completionrate = completionrate # Is this county call completion rate?
  ) %>%
  
  #There are some missing values so we're going to impute them.
  missRanger(num.trees = 100)



####

dim(data_tidy)

mod <- data_tidy %>% 
  count(completed_interview) %>% 
  mutate(prop = n/sum(n))

##92% yes and 8% no
## will need data splitting strategy and resample to balance


# Create data splits ------------------------------------------------------



set.seed(123)
splits      <- initial_split(data_tidy, strata = completed_interview)

# Renaming these so they aren't confusing later
data_non_test <- training(splits)
data_test  <- testing(splits)

# training set proportions by completed_interview
data_non_test %>% 
  count(completed_interview) %>% 
  mutate(prop = n/sum(n))

# test set proportions by completedint
data_test  %>% 
  count(completed_interview) %>% 
  mutate(prop = n/sum(n))


set.seed(234)
val_set <- validation_split(data_non_test, 
                            strata = completed_interview, 
                            prop = 0.80)
val_set




# ## First Model: Penalized Logistic Regression ---------------------------



lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

# Penalty is really "lambda", see here https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/lambda

# If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.
# 
# If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

# mixture is really teh ratio of pure ridge model (alpha = 0) and pure lasso model (alpha = 1). So for this one we are useing a pure lasso model. 

# Limitation of Ridge Regression: Ridge regression decreases the complexity of a model but does not reduce the number of variables since it never leads to a coefficient been zero rather only minimizes it. Hence, this model is not good for feature reduction.

# Lasso regression stands for Least Absolute Shrinkage and Selection Operator. Lasso sometimes struggles with some types of data. If the number of predictors (p) is greater than the number of observations (n), Lasso will pick at most n predictors as non-zero, even if all predictors are relevant (or may be used in the test set).
# If there are two or more highly collinear variables then LASSO regression select one of them randomly which is not good for the interpretation of data


holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <- 
  recipe(completed_interview ~ ., data = data_non_test) %>% 
  step_date(interview_initiate_date) %>% 
  step_holiday(interview_initiate_date, holidays = holidays) %>% 
  step_rm(interview_initiate_date) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())


lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)

##create grid for tuning
lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))
lr_reg_grid %>% top_n(-5) # lowest penalty values
lr_reg_grid %>% top_n(5)  # highest penalty values



# Train and tune LR model -------------------------------------------------




lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE,extract = extract_fit_engine),
            metrics = metric_set(roc_auc))

lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot


# Check out our top performing models

top_models <-
  lr_res %>% 
  show_best("roc_auc", n = 5) %>% 
  arrange(penalty) 
top_models

# Look at our best model
lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(25)
lr_best

# Plot AUC for our best model
lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(completed_interview, .pred_FALSE) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)

# Get variable values for our best model
# cf: https://stackoverflow.com/questions/63087592/how-to-extract-glmnet-coefficients-produced-by-tidymodels
extracted_linear_model <- lr_workflow %>%
  finalize_workflow(lr_best) %>%
  fit(data_non_test) %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  arrange(desc(abs(estimate))) %>%
  filter(estimate != 0)

write_csv(extracted_linear_model,"extracted_linear_model.csv")


# Random forest model -----------------------------------------------------



cores <- parallel::detectCores()
cores


rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
  set_mode("classification")


rf_recipe <- 
  recipe(completed_interview ~ ., data = data_non_test) %>% 
  step_date(interview_initiate_date) %>% 
  step_holiday(interview_initiate_date) %>% 
  step_rm(interview_initiate_date) 


rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

rf_mod

extract_parameter_set_dials(rf_mod)


set.seed(345) ####ERROR for missing values.
#Now there are no more missing values! Instead we imputed them.
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

rf_res %>% 
  show_best(metric = "roc_auc")

autoplot(rf_res)

rf_best <- 
  rf_res %>% 
  select_best(metric = "roc_auc")

rf_best

rf_res %>% 
  collect_predictions()

rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(completed_interview, .pred_FALSE) %>% 
  mutate(model = "Random Forest")

autoplot(rf_auc)

bind_rows(rf_auc, lr_auc) %>% 
  select(-.threshold) %>%
  ggplot() + 
  geom_path(aes(x = 1 - specificity, y = sensitivity, color = model),
            linewidth = 1.5, alpha = 0.8) +
  geom_abline(linetype = 3,linewidth = 1) + 
#coord_equal() + 
  theme_fivethirtyeight() +
  labs(color = "Model")

# extract the best rf model
extracted_rf_model <- rf_workflow %>%
  finalize_workflow(rf_best) %>%
  fit(data_non_test) %>%
  extract_fit_engine() 

extracted_rf_model %>%
  ranger::importance_pvalues(method = "janitza")

# Plot variable importance
feat_imp_df <- importance(extracted_rf_model) %>% 
  data.frame() %>% 
  rownames_to_column() %>%
  `colnames<-`(c("feature","importance"))

# plot dataframe
ggplot(feat_imp_df, aes(x = reorder(feature, importance), 
                        y = importance)) +
  geom_bar(stat='identity') +
  coord_flip() +
  theme_fivethirtyeight() +
  labs(
    x     = "Feature",
    y     = "Importance",
    title = "Feature Importance: Random Forest Model"
  )


# AUC for Test Set --------------------------------------------------------



#From here, mostly https://www.tidymodels.org/start/recipes/#predict-workflow
lr_best_parsnip_model <- lr_workflow %>%
  finalize_workflow(lr_best) %>%
  fit(data_non_test)


# LR Data_non_test --------------------------------------------------------



lr_aug_non_test<- augment(lr_best_parsnip_model,data_non_test)

lr_aug_non_test %>%
  select(completed_interview,.pred_class,.pred_FALSE,.pred_TRUE) %>%
  mutate(correct  = (completed_interview == .pred_class)) %>%
  summarize(pct_correct = sum(correct)/n())

#  91.44% correct

lr_aug_non_test %>%
  roc_curve(truth = completed_interview,.pred_FALSE) %>%
  autoplot()

lr_aug_non_test %>%
  roc_auc(truth = completed_interview,.pred_FALSE)
# Area under the curve = 0.757


# LR_Data_Test ------------------------------------------------------------



lr_aug_test<- augment(lr_best_parsnip_model,data_test)

lr_aug_test %>%
  select(completed_interview,.pred_class,.pred_FALSE,.pred_TRUE) %>%
  mutate(correct  = (completed_interview == .pred_class)) %>%
  summarize(pct_correct = sum(correct)/n())

# 91.8% correct

lr_aug_test_auc <-lr_aug_test %>%
  roc_curve(truth = completed_interview,.pred_FALSE) %>% 
  mutate(model = "Linear Regression")

lr_aug_test_auc %>% autoplot()



lr_aug_test %>%
  roc_auc(truth = completed_interview,.pred_FALSE)


# RF Data Non Test --------------------------------------------------------


rf_best_parsnip_model<- rf_workflow %>%
  finalize_workflow(rf_best) %>%
  fit(data_non_test) 


rf_aug_non_test<- augment(rf_best_parsnip_model,data_non_test)

rf_aug_non_test %>%
  select(completed_interview,.pred_class,.pred_FALSE,.pred_TRUE) %>%
  mutate(correct  = (completed_interview == .pred_class)) %>%
  summarize(pct_correct = sum(correct)/n())

# 91.8% correct

rf_aug_non_test %>%
  roc_curve(truth = completed_interview,.pred_FALSE) %>%
  autoplot()

rf_aug_non_test %>%
  roc_auc(truth = completed_interview,.pred_FALSE)
# Area under the curve = 0.769.


# RF Data Test ------------------------------------------------------------


rf_aug_test<- augment(rf_best_parsnip_model,data_test)

rf_aug_test %>%
  select(completed_interview,.pred_class,.pred_FALSE,.pred_TRUE) %>%
  mutate(correct  = (completed_interview == .pred_class)) %>%
  summarize(pct_correct = sum(correct)/n())

# 91.8% correct

rf_aug_test_auc <- rf_aug_test %>%
  roc_curve(truth = completed_interview,.pred_FALSE) %>% 
  mutate(model = "Random Forest")

rf_aug_test_auc %>% autoplot()

rf_aug_test %>%
  roc_auc(truth = completed_interview,.pred_FALSE)


# Final Graph -------------------------------------------------------------

bind_rows(rf_aug_test_auc, lr_aug_test_auc) %>% 
  select(-.threshold) %>%
  ggplot() + 
  geom_path(aes(x = 1 - specificity, y = sensitivity, color = model),
            linewidth = 1.5, alpha = 0.8) +
  geom_abline(linetype = 3,linewidth = 1) + 
  #coord_equal() + 
  theme_fivethirtyeight() +
  labs(color = "Model")
