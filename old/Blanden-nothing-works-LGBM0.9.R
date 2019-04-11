library(data.table)
library(lightgbm)
library(caret)

LGB_CV_Predict <- function(lgb_cv, data, num_iteration = NULL, folds=NULL, type=c("cv", "test")) {
  require(foreach)
  if (is.null(num_iteration)) {
    num_iteration <- lgb_cv$best_iter
  }
  if (type=="cv"){
    print("create cross validation predictions")
    pred_mat <- foreach::foreach(i = seq_along(lgb_cv$boosters), .combine = "c", .packages=c("data.table","lightgbm")) %do% {
      lgb_tree <- lgb_cv$boosters[[i]][[1]]
      predict(lgb_tree, 
              data[folds[[i]],], 
              num_iteration = num_iteration, 
              rawscore = FALSE, predleaf = FALSE, header = FALSE, reshape = FALSE)
    }
    
    as.double(pred_mat)[order(unlist(folds))]
    
  } else if (type=="test"){
    print("create test set predictions")
    pred_mat <- foreach::foreach(i = seq_along(lgb_cv$boosters), .combine = "+", .packages=c("data.table","lightgbm")) %do% {
      lgb_tree <- lgb_cv$boosters[[i]][[1]]
      predict(lgb_tree, 
              data, 
              num_iteration = lgb_cv$best_iter, 
              rawscore = FALSE, predleaf = FALSE, header = FALSE, reshape = FALSE)
    }
    as.double(pred_mat)/length(lgb_cv$boosters)
  }
}


t1 <- fread("../input/train.csv")
s1 <- fread("../input/test.csv")
t1[,filter:=0]
s1[,":="(target=-1,
         filter=2)]

ts1 <- rbind(t1, s1)
set.seed(500)
cvFoldsList <- createFolds(ts1[filter==0, target], k=10)

varnames <- setdiff(colnames(ts1), c("ID_code","target", "filter"))
dtrain <- lgb.Dataset(data.matrix(ts1[filter==0,varnames,with=F]), label=ts1[filter==0, target], free_raw_data = FALSE)

params <- list(objective = "binary", 
               boost="gbdt",
               metric="auc",
               boost_from_average="false",
               num_threads=28,
               learning_rate = 0.01,
               num_leaves = 13,
               max_depth=-1,
               tree_learner = "serial",
               feature_fraction = 0.05,
               bagging_freq = 5,
               bagging_fraction = 0.4,
               min_data_in_leaf = 80,
               min_sum_hessian_in_leaf = 10.0,
               verbosity = 1)

tme <- Sys.time()
lgb1 <- lgb.cv(params,
               dtrain,
               nrounds=1000000,
               folds=cvFoldsList,
               early_stopping_rounds = 3000,
               eval_freq=1000,
               seed=44000)
Sys.time() - tme

test_preds <- LGB_CV_Predict(lgb1, data.matrix(ts1[filter==2, varnames, with=F]), type="test")
dt <- data.table(ID_code=ts1[filter==2, ID_code], target=test_preds)
fwrite(dt, "./submission.csv")




