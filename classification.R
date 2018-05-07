require(quanteda.corpora)
require(quanteda)

corp <- data_corpus_movies
docvars(corp, "manual") <- factor(docvars(corp, "Sentiment"), c("neg", "pos"))

corp_sent <-  corpus_reshape(corp)
mt_sent <- dfm(corp_sent, remove_punct = TRUE) %>% 
           dfm_trim(min_termfreq = 5) %>% 
           dfm_remove(stopwords("en"), min_nchar = 2)
mt <- dfm_group(mt_sent, "id2")

feat <- names(topfeatures(mt, 1000))

data <- data.frame(manual = docvars(mt, "manual"))

## Supervised method ------------

i <- seq(ndoc(mt)) 
l <- i %in% sample(i, 1500) 
mt_train <- mt[l,] 
mt_test <- mt[!l,]

accuracy <- function (x) {
    list(neg_recall =  x[1,1] / sum(x[1,]),
         neg_precision = x[1,1] / sum(x[,1]),
         pos_recall = x[2,2] / sum(x[2,]),
         pos_precision = x[2,2] / sum(x[,2])
    )
}

# ------------------------------

nb <- dfm_select(mt_train, feat) %>% 
    textmodel_nb(docvars(mt_train, "manual"))

data$nb <- NA
#data$nb[!l] <- predict(nb, newdata = mt_test) # should like this
data$nb[!l] <- predict(nb, newdata = dfm_select(mt_test, mt_train))$nb.predicted

tb_nb <- table(data$manual, data$nb, dnn = c("manual", "nb"))
tb_nb
mosaicplot(tb_nb)
accuracy(tb_nb)


# ---------------------------

require(randomForest)
rf <- dfm_select(mt_train, feat) %>% as.matrix() %>% 
      randomForest(docvars(mt_train, "manual"))

data$rf <- NA
data$rf[!l] <- predict(rf, mt_test)

tb_rf <- table(data$manual, data$rf, dnn = c("manual", "rf"))
tb_rf
mosaicplot(tb_rf)
accuracy(tb_rf)

# -----------------------------

ws <- dfm_select(mt_train, feat) %>% 
    textmodel_wordscores(as.numeric(docvars(mt_train, "manual")))

data$ws <- NA
data$ws[!l] <- predict(ws, newdata = mt_test)

plot(data$manual, data$ws)
t.test(ws ~ manual, data)

# ----------------------------


## Unsupervided and semi-supervided -------------------

wf <- dfm_select(mt, feat) %>% textmodel_wordfish()
data$wf <- wf$theta

table(data$manual, data$wf > 0)

plot(data$manual, data$wf)
t.test(wf ~ manual, data)

head(coef(wf, "features")[,"beta"], 30)
tail(coef(wf, "features")[,"beta"], 30)

# ---------------------------

require(LSS)
#feat2 <- feat <- names(topfeatures(mt, 5000))
lss <- textmodel_lss(mt_sent, seedwords("pos-neg"), feat, cache = TRUE, k = 200)
#lss$beta <- lss$beta[lss$beta < quantile(lss$beta, p) | quantile(lss$beta, 1 - p) < lss$beta]
#lss$beta <- (lss$beta ** 2) * sign(lss$beta)

length(lss$beta)
#data$lss <- predict(lss, newdata = t(t(mt) / docfreq(mt)))
data$lss <- predict(lss, newdata = mt)
#data$lss <- predict(lss, newdata = mt, weight_scheme = "boolean")

accuracy(table(data$manual, data$lss > 0))

plot(data$manual, data$lss)
t.test(lss ~ manual, data)

head(coef(lss), 30)
tail(coef(lss), 30)

# how big the training data should be ------------------------------

l2 <- i %in% sample(i, 1000) 
mt_test2 <- mt[!l2,]

data <- data.frame()
for (n in seq(100, 1000, by = 100)) {
    for (m in seq(1:20)) {
        mt_train2 <- mt[sample(i[l2], n),]
        nb <- dfm_select(mt_train2, feat) %>% 
              textmodel_nb(docvars(mt_train2, "manual"))
    
        #docvars(mt_train2, "class") <- predict(nb, newdata = mt_test) # should like this
        docvars(mt_test2, "nb") <- predict(nb, newdata = dfm_select(mt_test2, mt_train2))$nb.predicted
        
        tb_temp <- table(docvars(mt_test2, "manual"), docvars(mt_test2, "nb"))
        
        temp <- as.data.frame(accuracy(tb_temp))
        temp$size <- n
        data <- rbind(data, temp)
    }
}

require(gplots)
par(mfrow = c(2, 2), mar = c(4, 4, 1, 1))
plotmeans(neg_precision ~ size, data, ylim = c(0.65, 0.8), n.label = FALSE)
plotmeans(neg_recall ~ size, data, ylim = c(0.65, 0.8), n.label = FALSE)
plotmeans(pos_precision ~ size, data, ylim = c(0.65, 0.8), n.label = FALSE)
plotmeans(pos_recall ~ size, data, ylim = c(0.65, 0.8), n.label = FALSE)
par(mfrow = c(1, 1))

sort(docfreq(mt)[feat], decreasing = TRUE) %>% 
    plot(type = "b", ylab = "Document frequency of features")

feat_rare <- tail(feat, 10)
feat_rare
docfreq(mt_train2)[feat_rare] * (500 /  ndoc(mt_train2))


plot(head(sort(docfreq(mt_train2), decreasing = TRUE), 1000), type = "l",
     ylab = "Docuent frequency of features", 
     xlab = "Feature ranks")


sparsity(dfm(corp, remove_punct = TRUE)) # this is a dense corpus 

