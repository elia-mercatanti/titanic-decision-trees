# importo dataset
pokemon <- read.csv('datasets/pokemon.csv', sep=',')
pokemon <- data.frame(pokemon)

# inizializzazione data set per analisi descritttiva
pokemonDescr <- subset(pokemon,select=-c(1,2,4,13:15,17:19))
for(i in 1:721){
  x <- as.integer(pokemonDescr[i,"Attack"]) 
  if(x<=45){
    pokemonDescr[i,"Attack_Level"] <- "Low"
  } else if(x>45 && x <=85){
    pokemonDescr[i,"Attack_Level"] <- "Med-Low"
  } else if(x>85 && x <=125){
    pokemonDescr[i,"Attack_Level"] <- "Med-High"
  } else {
    pokemonDescr[i,"Attack_Level"] <- "High"
  }
}

library(dplyr) # funzione %>%
library(fmsb) # radarchart

group <- pokemonDescr %>% group_by(Attack_Level) %>% select(
  HP,Defense,Sp..Atk,Sp..Def,Speed,Height_m, Weight_kg, Catch_Rate) %>% 
  summarise(HP = mean(HP),Defense = mean(Defense),SPAtk = mean(Sp_Atk),
            SPDef = mean(Sp_Def),Height = mean(Height_m),
            Weight = mean(Weight_kg),Catch = mean(Catch_Rate))
max<-c(100,100,100,100,3,200,200)
min<-rep(0,8)
par(mfrow=c(4,4))
par(mar=c(1,1,1,1))
colors_border=c( rgb(0.2,0.5,0.5,0.9), rgb(0.8,0.2,0.5,0.9) , rgb(0.7,0.5,0.1,0.9) )
colors_in=c( rgb(0.2,0.5,0.5,0.4), rgb(0.8,0.2,0.5,0.4) , rgb(0.7,0.5,0.1,0.4) )
radarchart( rbind(max,min,group[,2:8])  , axistype=2,
            #poligoni
            plwd=4 , plty=1,
            #rete
            cglcol="grey", cglty=1, axislabcol="grey", caxislabels=seq(0,100,10), cglwd=0.8,
            #etichette
            vlcex=0.8
)

library(gplots) # heatmaps

pokemonGroup <- as.data.frame(group)
row.names(pokemonGroup) <- pokemonGroup$Attack
pokemonGroup <- pokemonGroup[,2:8]
group_matrix <- data.matrix(pokemonGroup)
heatmap.2(group_matrix, Rowv=FALSE, Colv=FALSE, 
          dendrogram='none', cellnote=round(group_matrix,digits=2),
          notecex = 2.0, notecol="black", trace='none',
          key=FALSE,lwid = c(.01,.99),lhei = c(.01,.99),margins = c(8,16))

#Predizione
pokemonPred <- subset(pokemon,select=-c(2,4,13:19,23))


for(i in 1:721){
  x <- as.integer(pokemonPred[i,"Attack"]) 
  if(x<=45){
    pokemonPred[i,"Attack"] <- "Low"
  } else if(x>45 && x <=85){
    pokemonPred[i,"Attack"] <- "Med-Low"
  } else if(x>85 && x <=125){
    pokemonPred[i,"Attack"] <- "Med-High"
  } else {
    pokemonPred[i,"Attack"] <- "High"
  }
}


Strong <- grepl("High", pokemonPred[,"Attack"], perl=TRUE)
pokemonPred <- data.frame(pokemonPred,Strong)

### preparazione insieme di test e training ###
set.seed(15)
train = sample(1:nrow(pokemon),400) 
pokemonPred.test = pokemonPred[-train,]
Strong.test = Strong[-train]

### Alberi con pacchetto "tree" ###
library(tree)

# albero della DEVIANZA,DELL'ENTROPIA o DELL'INFORMAZIONE 
tree.dev = tree(as.factor(Strong)~.-Attack,pokemonPred.train)#costruzione albero
plot(tree.dev, type = "uniform")
text(tree.dev,pretty=2,cex=0.8)

tree.dev_pred = predict(tree.dev,pokemonPred.test,type = "class")#predizione
tableCM.dev=table(tree.dev_pred,Strong.test) #confusion Matrix
accuracy.dev<- 1-(((tableCM.dev[1,2]+tableCM.dev[2,1]))/321)
accuracy.dev

set.seed(15)
cvpokemon.dev<- cv.tree(tree.dev,FUN = prune.misclass)
cvpokemon.dev
plot(cvpokemon.dev$size,cvpokemon.dev$dev,type="b",
     cex=2,cex.axis=1.5,cex.lab=1.8,cex.main=1.8,
     xlab="size",ylab="cverror",main="Scelta della dimensione")

prunetree.dev <- prune.misclass(tree.dev,best=5) #default:method=deviance
plot(prunetree.dev,type = "uniform")
text(prunetree.dev,pretty=1,cex=2)

prunetree.dev_pred = predict(prunetree.dev,pokemonPred.test,type = "class")#predizione
tableCM.dev.prune=table(prunetree.dev_pred,Strong.test) #confusion Matrix
accuracy.dev.prune<- 1-(((tableCM.dev.prune[1,2]+tableCM.dev.prune[2,1]))/321)
accuracy.dev.prune

# albero con l'indice di Gini
tree.gini = tree(as.factor(Strong)~.-Attack,pokemonPred.train,split='gini')
plot(tree.gini, type = "uniform")
text(tree.gini,pretty=2,cex=0.8)
tree.gini_pred = predict(tree.gini,pokemonPred.test,type = "class")#predizione

tableCM.gini=table(tree.gini_pred,Strong.test) #confusion Matrix
accuracy.gini<- 1-(((tableCM.gini[1,2]+tableCM.gini[2,1]))/321)
accuracy.gini

set.seed(15)
cvpokemon.gini<- cv.tree(tree.gini,FUN = prune.misclass)
cvpokemon.gini
plot(cvpokemon.gini$size,cvpokemon.gini$dev,type="b",
     cex=2,cex.axis=1.5,cex.lab=1.8,cex.main=1.8,
     xlab="size",ylab="cverror",main="Scelta della dimensione")
prunetree.gini <- prune.misclass(tree.gini,best=8) #default:method=deviance
plot(prunetree.gini, type = "uniform")
text(prunetree.gini,pretty=1,cex=1.3)

prunetree.gini_pred = predict(prunetree.gini,pokemonPred.test,type = "class")#predizione
tableCM.gini.prune=table(prunetree.gini_pred,Strong.test) #confusion Matrix
accuracy.gini.prune<- 1-(((tableCM.gini.prune[1,2]+tableCM.gini.prune[2,1]))/321)
accuracy.gini.prune

############################################################################
############################################################################
############################################################################

### Alberi con pacchetto RPART ###

library(rpart)
library(rpart.plot)

# tolgo la colonna attacco perché non mi riesce 
# a farlo dentro alla formula
pokemonPredRpart <- subset(pokemonPred,select=-c(5))
pokemonPredRpart.train <- pokemonPredRpart[1:400,]
pokemonPredRpart.test <- pokemonPredRpart[401:721,]

# albero con Gini
rtree.gini = rpart(Strong ~ .,data = pokemonPredRpart.train,method = "class",parms = list(split='gini'))
rpart.plot(rtree.gini, cex = 0.55)
rtree.gini_pred <- predict(rtree.gini,pokemonPred.test,type = "class")
tableCM.gini2=table(rtree.gini_pred,Strong.test) #confusion Matrix
accuracy.gini2<- 1-(((tableCM.gini2[1,2]+tableCM.gini2[2,1]))/321)
accuracy.gini2

#cerchiamo indice migliore
plotcp(rtree.gini,cex.lab=1.5,cex.axis=1.3,cex=2)
#prunaggio
rtree.gini.prune<- prune.rpart(rtree.gini,cp=0.032)
rpart.plot(rtree.gini.prune,tweak = 1.2)
rtree.gini_pred.prune <- predict(rtree.gini.prune,pokemonPred.test,type = "class")
tableCM.gini2.prune=table(rtree.gini_pred.prune,Strong.test) #confusion Matrix
accuracy.gini2.prune<- 1-(((tableCM.gini2.prune[1,2]+tableCM.gini2.prune[2,1]))/321)
accuracy.gini2.prune

# albero entropia
rtree.entr = rpart(Strong ~ .,data = pokemonPredRpart.train,method = "class",parms = list(split='information'))
rpart.plot(rtree.entr, cex = 0.55)
rtree.entr_pred <- predict(rtree.entr,pokemonPred.test,type = "class")
tableMC.dev2=table(rtree.entr_pred,Strong.test)
accuracy.dev2<- 1-(((tableMC.dev2[1,2]+tableMC.dev2[2,1]))/321)
accuracy.dev2

#indice migliore
plotcp(rtree.entr)
#prunaggio
rtree.entr.prune<- prune.rpart(rtree.entr,cp=0.11)
rpart.plot(rtree.entr.prune,tweak = 0.7)

rtree.dev_pred.prune <- predict(rtree.entr.prune,pokemonPred.test,type = "class")
tableCM.dev2.prune=table(rtree.dev_pred.prune,Strong.test) #confusion Matrix
accuracy.dev2.prune<- 1-(((tableCM.dev2.prune[1,2]+tableCM.dev2.prune[2,1]))/321)
accuracy.dev2.prune


############################################################################
##########################################################################
#############################################################################

### Alberi con pacchetto PARTY ###

library(party)

#Conditional Inference Tree - Party
#ho corretto il comando ed esce la matrice di dispersione a modino

partytree=ctree(as.factor(Strong)~.,pokemonPredRpart.train)
plot(partytree, type = "simple", main="Conditional Inference Tree")
party_pred<- predict(partytree,pokemonPredRpart.test)
tableMC.party<-table(party_pred,Strong.test)
accuracy.party<- 1-(((tableMC.party[1,2]+tableMC.party[2,1]))/321)
accuracy.party

#partytree = ctree(Strong~.,data= pokemonPredRpart.train)
#summary(predict(partytree))
#plot(partytree, type = "extended", main="Conditional Inference Tree")

#cMatrix = predict(partytree)
# for(i in 1:400){
#   x <- cMatrix[i] 
# if(x<=0.5){
#   cMatrix[i] <- "FALSE"
# } else {
#   cMatrix[i] <- "TRUE"
# } 
# }


### Alberi con pacchetto EVTREE ###

library(evtree)

ev_tree <- evtree(as.factor(Strong)~., data = pokemonPredRpart.train)
plot(ev_tree,type = "simple")
evtree_pred<- predict(ev_tree,pokemonPredRpart.test)

tableMC.evtree<-table(evtree_pred,Strong.test)
accuracy.evtree<- 1-(((tableMC.evtree[1,2]+tableMC.evtree[2,1]))/321)
accuracy.evtree