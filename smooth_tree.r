# R code partially modified from the BooST repositorie: https://github.com/gabrielrvsc/BooST

grow_tree=function(x, y, p, d_max, gamma, node_obs){
  bf=0

  N=length(y)

  t0 = grow0(x,y,p,N,gamma,node_obs)
  tree = t0$tree
  Pmat = t0$Pmat

  iter=1
  while(iter<=d_max){
    titer=grow(x,y,p,N,gamma,node_obs,tree,Pmat,d_max,iter,bf)
    tree = titer$tree
    Pmat = titer$Pmat
    iter = titer$iter
    bf = titer$bf
  }
  Pmat[is.na(Pmat)]=0
  fitted=Pmat%*%tree$b
  result=list(tree=tree,fitted.values=fitted)
  return(result)
}

grow0 = function(x,y,p,N,gamma,node_obs){
  variables = sample(ncol(x), round(p*ncol(x)))
  gammai=gamma[sample(1:length(gamma),1)]
  fit=list()
  for(i in 1:length(variables)){
    xtest=x[,variables[i]]
    xtest=sample(xtest,min(20,N))
    gammascale=max(stats::sd(x[,variables[i]]),0.1)
    ssr=sapply(xtest,initial_node_var_test,x=x[,variables[i]],y=y,gamma=gammai/gammascale,node_obs=node_obs)
    ssr=t(ssr)
    best=which.min(ssr[,1]) # ACHA O MENOR P-VALUE?
    res0=c(xtest[best],gammai/gammascale,ssr[best,])
    names(res0)=c("c0","gamma","val","b0","b1")
    fit[[i]]=res0

  }
  best=which.min(lapply(fit,function(x)x["val"]))
  node=fit[[best]]

  nodeleft=data.frame("side" = 1,"b" = node["b0"],"c0" = node["c0"],gamma = node["gamma"]
                      , "parent" = 0, "terminal" = "yes", variable = variables[best], id = 1,deep=1)
  noderight=data.frame(side = 2,  "b" = node["b1"],"c0" = node["c0"],gamma = node["gamma"]
                       , "parent" = 0, "terminal" = "yes", variable = variables[best], id = 2,deep=1)
  tree=rbind(nodeleft,noderight)
  tree$terminal=as.character(tree$terminal)

  Pmat=1/(1+exp(-node["gamma"]*(x[,variables[best]]-node["c0"])))
  Pmat=cbind(Pmat,1-Pmat)
  return(list(tree = tree, Pmat=Pmat))
}

grow = function(x,y,p,N,gamma,node_obs,tree,Pmat,d_max,iter,bf){
  gammai=gamma[sample(1:length(gamma),1)]
  terminal=which(tree$terminal=="yes")
  variables=sample(ncol(x), round(p*ncol(x)))
  test=expand.grid(variables,terminal)
  colnames(test)=c("variable","terminal")

  t1=Sys.time()
  fit=list()
  for(i in 1:nrow(test)){
    xt=x[,test[i,"variable"]]
    #xtest=stats::runif(10,min(xt)-0.1*stats::sd(xt),max(xt)+0.1*stats::sd(xt))
    xtest=sample(xt,min(20,N),prob = Pmat[,test$terminal[i]]+0.01)

    gammascale=max(stats::sd(xt),0.1)
    middlenodes=which(is.na(colSums(Pmat)))

    fit[[i]]=c(val=Inf)
    ssr=sapply(xtest,node_var_test,x=x[,test[i,"variable"]],y=y,gamma=gammai/gammascale,Pmat=Pmat,
               terminal=test$terminal[i],middlenodes=middlenodes,deep=tree$deep[test$terminal[i]]+1,node_obs=node_obs)
    ssr=t(ssr)
    ssr[is.nan(ssr)]=Inf
    best=which.min(ssr[,1])
    res=c(xtest[best],gammai/gammascale,ssr[best,])
    names(res)=c("c0","gamma","val","b0","b1")
    fit[[i]]=res

  }
  Sys.time()-t1
  best=which.min(lapply(fit,function(x)x["val"]))
  node=fit[[best]]
  if(bf==5){
    iter=d_max+1
    return(list(tree = tree, Pmat = Pmat, iter = iter, bf = bf))
  }
  if(is.infinite(node["val"])){
    bf=bf+1
    return(list(tree = tree, Pmat = Pmat, iter = iter, bf = bf))
  }

  nodeleft=data.frame("side" = 1,"b" = node["b0"],"c0" = node["c0"],gamma = node["gamma"]
                      , "parent" = tree$id[test$terminal[best]], "terminal" = "yes", variable = test[best,"variable"],id=nrow(tree)+1,
                      deep=tree$deep[test$terminal[best]]+1)
  noderight=data.frame(side = 2,  "b" = node["b1"],"c0" = node["c0"],gamma = node["gamma"]
                       , "parent" = tree$id[test$terminal[best]], "terminal" = "yes", variable = test[best,"variable"],id=nrow(tree)+2,
                       deep=tree$deep[test$terminal[best]]+1)

  tree$terminal[test$terminal[best]]="no"

  tree=rbind(tree,nodeleft,noderight)

  p0=(1/(1+exp(-node["gamma"]*(x[,nodeleft$variable]-node["c0"]))))*Pmat[,test$terminal[best]]
  p1=(1-(1/(1+exp(-node["gamma"]*(x[,noderight$variable]-node["c0"])))))*Pmat[,test$terminal[best]]
  Pmat=cbind(Pmat,p0,p1)
  Pmat[,test$terminal[best]]=NA
  tree$b[tree$terminal=="yes"] =  node[ c(6:length(node),4,5)]
  iter=iter+1
  return(list(tree = tree, Pmat = Pmat, iter = iter, bf = bf))
}

initial_node_var_test=function(c0,x,y,gamma,node_obs){
#  set.seed(0)
  logit=1/(1+exp(-gamma*(x-c0)))
  b0=logit;b1=1-logit
  X=cbind(b0,b1)
  l0=length(which(b0>=0.5))
  l1=length(which(b1>=0.5))
  if(l0<node_obs | l1<node_obs){
    return(c(Inf,rep(NA,ncol(X))))
  }
  b=tryCatch(stats::coef(stats::.lm.fit(X,y)),error=function(e)Inf) # regressao normal, retorna os coeficientes
  if(is.infinite(b[1])){
    return(c(b[1],rep(NA,ncol(X))))
  }
  c(sum((y-X%*%b)^2),b) # retorna soma dos residuos ao quadrado e os coeficientes da regressao
}

node_var_test=function(c0,x,y,gamma,Pmat,terminal,middlenodes,deep,node_obs){
  logit=1/(1+exp(-gamma*(x-c0)))
  b0=logit*Pmat[,terminal]
  b1=(1-logit)*Pmat[,terminal]
  X=cbind(b0,b1,Pmat[,-c(terminal,middlenodes)])
  l0=length(which(b0>=0.5^deep))
  l1=length(which(b1>=0.5^deep))
  if(l0<node_obs | l1<node_obs){
    return(c(Inf,rep(NA,ncol(X))))
  }
  b=tryCatch(stats::coef(stats::.lm.fit(X,y)),error=function(e)Inf)
  if(is.infinite(b[1])){
    return(c(b[1],rep(NA,ncol(X))))
  }
  e=sum((y-X%*%b)^2)
  c(e,b)
}

predict.SmoothTree=function(object,newx,...){
#  set.seed(0)
  if(is.null(newx)){
    return(stats::fitted(object))
  }

  if(is.vector(newx)){newx=matrix(newx,nrow=1)}

#  newx=matrix(newx,nrow=1)
  model=object$tree
  fitted.values=eval_tree(newx,model)

  return(fitted.values)
}

smooth_tree=function(x, y, p = 1, d_max = 4, gamma = seq(0.5,5,0.01),node_obs=nrow(x)/200, random = FALSE){
  if(random==TRUE){
    grow_tree = grow_tree_random
  }
  tree=grow_tree(x,y, p = p, d_max = d_max, gamma = gamma, node_obs=node_obs)
  fitted.values=tree$fitted.values
  result=list(tree=tree$tree, fitted.values=fitted.values, nvar = ncol(x) , varnames=colnames(x) ,call=match.call())
  class(result)="SmoothTree"
  return(result)
}

eval_tree=function(x,tree){
#  set.seed(0)
  terminal=tree[which(tree$terminal=="yes"),]
  logimat=matrix(NA,nrow(x),nrow(terminal))
  for(i in 1:nrow(terminal)){
    node=terminal[i,]
    logit=1/(1+exp(-node$gamma*(x[,node$variable]-node$c0)))
    if(node$side==2){logit=1-logit}
    parent=node$parent
    while(parent!=0){
      node=tree[parent,]
      logitaux=1/(1+exp(-node$gamma*(x[,node$variable]-node$c0)))
      if(node$side==2){logitaux=1-logitaux}
      logit=logit*logitaux
      parent=node$parent
    }
    logimat[,i]=logit
  }

  fitted=logimat%*%terminal$b
  return(fitted)
}

BooST = function(x, y, v=0.2, p = 2/3, d_max = 4, gamma = seq(0.5,5,0.01),
                 M = 300, display=FALSE,
                 stochastic=FALSE,s_prop=0.5, node_obs=nrow(x)/200, random = FALSE) {

  params = list(v=v,p=p,d_max=d_max,gamma=gamma,M=M,stochastic=stochastic,
                s_prop=s_prop,node_obs=node_obs, random = random)

  d_max=d_max-1
  N=length(y)
  phi=rep(mean(y),length(y))

  brmse=rep(NA,M)
  savetree=vector(mode = "list", length = M)
  save_rho=rep(NA,M)
  if(random==TRUE){
    grow_tree = grow_tree_random
  }

  if(stochastic==TRUE){
    for(i in 1:M){
      s=sample(1:N,floor(N*s_prop),replace = FALSE)
      u=y-phi

      step=grow_tree(x=x[s,],y=u[s],p=p,d_max=d_max,gamma=gamma,node_obs=node_obs)
      fitstep=eval_tree(x,step$tree)
      rho=stats::coef(stats::lm(y[s]-phi[s]~-1+fitstep[s]))

      phitest=phi+v*rho*fitstep
      savetree[[i]]=step
      brmse[i]=sqrt(mean((y-phitest)^2))

      if(i>1){
        if(brmse[i]/brmse[i-1]>1.02){
          rho=0
          phitest=phi+v*rho*fitstep
          savetree[[i]]=step
          brmse[i]=sqrt(mean((y-phitest)^2))
          cat("stag")
        }
      }
      phi=phitest
      save_rho[i]=rho
      if(display==TRUE){
        cat(i," RMSE = ",brmse[i],"\n")
      }

    }

  }else{

    for(i in 1:M){
      u=y-phi
      step=grow_tree(x=x,y=u,p=p,d_max=d_max,gamma=gamma,node_obs=node_obs)
      fitstep=stats::fitted(step)
      rho=stats::coef(stats::lm(y-phi~-1+fitstep))
      phitest=phi+v*rho*fitstep
      savetree[[i]]=step
      brmse[i]=sqrt(mean((y-phitest)^2))
      phi=phitest
      save_rho[i]=rho
      if(display==TRUE){
        cat(i," RMSE = ",brmse[i],"\n")
      }
    }

  }

  result=list(Model=savetree,fitted.values=phi,brmse=brmse,ybar=mean(y),v=v,rho=save_rho,nvar=ncol(x),varnames=colnames(x),params = params ,call=match.call())
  class(result)="BooST"
  return(result)
}

predict.BooST=function(object,newx=NULL,...){

#  set.seed(0)
  if(is.null(newx)){
    return(stats::fitted(object))
  }

  if(is.vector(newx)){newx=matrix(newx,nrow=1)}

  v=object$v
  y0=object$ybar
  rho=object$rho
  model=object$Model
  rhov=rho*v
  fitaux=t(t(Reduce("cbind",lapply(model,function(t)eval_tree(newx,t$tree))))*rhov)
  fitted.values=y0+rowSums(fitaux)
  return(fitted.values)
}
