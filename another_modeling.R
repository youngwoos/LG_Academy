library(tidymodels)
library(dplyr)
library(ggplot2)
library(readr)

# Regression Tree ----------------------------------------------------------
titanic_raw <- read_csv("data/titanic.csv")

# 복사본 생성
df <- titanic_raw

# 타겟 변수 결측치 제거
df <- df %>% 
  filter(!is.na(age))

# 데이터 탐색
glimpse(df)

# 분할
set.seed(12)
df_split <- df %>%
  initial_split(prop = 0.8, strata = age)

df_train <- training(df_split)
df_test <- testing(df_split)

mean(df_train$age)
mean(df_test$age)

sd(df_train$age)
sd(df_test$age)

summary(df_train$age)
summary(df_test$age)


# 레서피 생성
stack_recipe <-
  recipe(data = df_train, age ~ .) %>%
  update_role(passengerid, name, ticket, cabin, new_role = "id")

# 모델 세팅
tree_mod <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("regression")

# 워크플로우 생성
stack_wf <- workflow() %>%
  add_recipe(stack_recipe) %>%
  add_model(tree_mod)

# 모델 생성
set.seed(12)
fit_tree <- stack_wf %>%
  fit(data = df_train)

fit_tree

## 모델 시각화
library(rattle)
library(tune)

# 노드의 값 = 해당 조건 케이스의 타겟 변수 평균
windows()

fit_tree %>%
  extract_model() %>%
  fancyRpartPlot()

# Variable importance
library(vip)
fit_tree %>%
  extract_model() %>%
  vip()


# 예측값 생성
df_train <- df_train %>%
  bind_cols(predict(fit_tree, df_train)) %>%
  rename(pred = .pred)

glimpse(df_train)

df_test <- df_test %>%
  bind_cols(predict(fit_tree, df_test)) %>%
  rename(pred = .pred)

glimpse(df_test)


# 평가
metrics(data = df_train,
        truth = age,
        estimate = pred)

metrics(data = df_test,
        truth = age,
        estimate = pred)




# Random Forest: Classification ----------------------------------------------------------

df <- read_csv("data/titanic.csv")

# 타겟 변수 전처리
df <- df %>%
  mutate(survived = ifelse(survived == 1, "surv", "die"),
         survived = factor(survived, levels = c("surv", "die")))

# 변수 검토
library(dlookr)
diagnose(df)

# 결측치 제거
library(tidyr)
df <- df %>% 
  drop_na(age, fare, embarked)

# 레서피 생성
stack_recipe <-
  recipe(data = df, survived ~ .) %>%
  update_role(passengerid, name, ticket, cabin, new_role = "id")

stack_recipe

# 모델 세팅
tree_mod <- rand_forest() %>%
  set_engine("ranger", importance = "impurity") %>%  # 변수 중요도
  set_mode("classification")

# 워크플로우 생성
stack_wf <- workflow() %>%
  add_recipe(stack_recipe) %>%
  add_model(tree_mod)

# 모델 생성
set.seed(12)
fit_tree <- stack_wf %>%
  fit(data = df)

# 변수 중요도
library(vip)
fit_tree %>%
  extract_model() %>%
  vip()



# Random Forest: Regression ----------------------------------------------------------
df <- read_csv("data/titanic.csv")

# 변수 검토
library(dlookr)
diagnose(df)

# 결측치 제거
library(tidyr)
df <- df %>% 
  drop_na(age, fare, embarked)

# 레서피 생성
stack_recipe <-
  recipe(data = df, fare ~ .) %>%
  update_role(passengerid, name, ticket, cabin, new_role = "id")

stack_recipe

# 모델 세팅
tree_mod <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("regression")

# 워크플로우 생성
stack_wf <- workflow() %>%
  add_recipe(stack_recipe) %>%
  add_model(tree_mod)

# 모델 생성
set.seed(12)
fit_tree <- stack_wf %>%
  fit(data = df)

# 예측
df_pred <- df %>%
  bind_cols(predict(fit_tree, df)) %>%
  rename(pred = .pred)

# 성능 평가
metrics(data = df_pred,
        truth = fare,
        estimate = pred)


# Target Balancing --------------------------------------------------------
df <- read_csv("data/titanic.csv")
glimpse(df)

# factor로 변환
df <- df %>%
  mutate(survived = ifelse(survived == 1, "surv", "die"),
         survived = factor(survived, levels = c("surv", "die")))

levels(df$survived)

# 분할
set.seed(12)
df_split <- df %>%
  initial_split(prop = 0.8, strata = survived)

df_train <- training(df_split)
df_test <- testing(df_split)

df_train %>%
  count(survived) %>%
  mutate(ratio = n/sum(n)*100)


# 레서피 생성
# 밸런스 맞추기 step_upsample()
stack_recipe <-
  recipe(data = df_train, survived ~ .) %>%
  update_role(passengerid, name, ticket, cabin, new_role = "id") %>% 
  step_upsample(survived)

stack_recipe

# # 추출 확인
# df_train_upsample <- stack_recipe %>% 
#   prep() %>% 
#   juice()
# 
# # 처리 전
# df_train %>% 
#   count(survived) %>% 
#   mutate(ratio = n/sum(n)*100)
# 
# # 처리 후
# df_train_upsample %>% 
#   count(survived) %>% 
#   mutate(ratio = n/sum(n)*100)

# 모델 세팅
tree_mod <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 워크플로우 생성
stack_wf <- workflow() %>%
  add_recipe(stack_recipe) %>%
  add_model(tree_mod)

# 모델 생성
set.seed(12)
fit_tree <- stack_wf %>%
  fit(data = df_train)


