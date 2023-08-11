library(tidyverse)
library(mlr3verse)

task = as_task_regr(diamonds, target = "price", id = "diamonds")

set.seed(1234)

splits = partition(task,ratio=0.8, stratify = TRUE)

train = task$clone()$filter(splits$train)
test = task$clone()$filter(splits$test)

cv10 = rsmp("cv", folds = 10)
cv10$instantiate(train)

measure = msr("regr.rmse")

# Create a graph

learner = lrn("regr.lm")
graph =  po("encode",method = "treatment") %>>% learner
pipeline = GraphLearner$new(graph)

# Cross validation

rr = resample(train,pipeline,cv10)
res1=rr$score(measure)
rr$aggregate(measure)

# Train pipeline with optimal parameters

pipeline$train(train)

# Predict and get scores

resultado = pipeline$predict(test)
resultado$score(measure)
