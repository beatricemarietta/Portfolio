#Training & Testing
mydata=read.delim2("my.txt")
PPM=as.numeric(mydata$PPM)
TPT=as.numeric(mydata$TPT)
RLS=as.numeric(mydata$RLS)
JP=as.numeric(mydata$JP)
Kemiskinan=matrix(c(TPT,RLS,JP,PPM),ncol = 4)
colnames(Kemiskinan)=c("TPT","RLS","JP","PPM")

n=nrow(Kemiskinan)
ntrain=round(0.75*n, 0)
trainidx=sample(1:35,ntrain)
traindata=Kemiskinan[trainidx,]
testdata=Kemiskinan[-trainidx,]
Xtrain=traindata[,1:3]
Ytrain=traindata[,4]
Xtest=testdata[,1:3]
Ytest=testdata[,4]

#REPLIKASI INPUT
minEReg<-function (x,tg1,eta,eps,epc){
  n=nrow(x)
  ni=ncol(x)
  minx=matrix(rep(apply(x, 2, min), n), nrow=n, byrow=T)
  maxx=matrix(rep(apply(x, 2, max), n), nrow=n, byrow=T)
  rangex=maxx-minx
  x=cbind((x-minx)/rangex, rep(1,n))
  tg=0.1+0.8*(tg1-min(tg1))/(max(tg1)-min(tg1))
  set.seed(1)
  w=matrix(runif(8, 0.01, 1.01), ncol=2)
  v=runif(3, .01, 1.01)
  MSE=NULL
  ncount=1
  #langkah maju
  repeat {
    E=NULL #batch square error
    GV=NULL #batch gradien v
    GW1=NULL #batch gradien w node 1
    GW2=NULL #batch gradien w node 2
    OO=NULL #batch output jaringan
    for(k in 1:n){
      nh=c(0,0)
      oh=c(0,0)
      for (j in 1:2){
        nh[j]=sum(x[k,]*w[,j])
        oh[j]=1/(1+exp(-nh[j]))
      }
      x2=c(oh, 1)
      no=sum(x2*v)
      oo=1/(1+exp(-no)) #output jaringan
      OO=c(OO,oo)
      E=c(E, (tg[k]-oo)^2)
      gv=c(0,0,-(tg[k]-oo)*oo*(1-oo))
      for (j in 1:2){
        gv[j]=-(tg[k]-oo)*oo*(1-oo)*oh[j]
        gw=matrix(0,nrow=4, ncol=2)
        for(i in 1:4){
          gw[i,j]=-(tg[k]-oo)*oo*(1-oo)*v[j]*oh[j]*x[k,i]
        }
      }
      GV=cbind(GV, gv)
      GW1=cbind(GW1, gw[,1])
      GW2=cbind(GW2, gw[,2])
    }
    MSE=c(MSE, mean(E))
    if (mean(E)<eps || ncount>=epc){
      break
    }
    else { #update bobot
      gv=apply(GV, 1, mean)
      gw1=apply(GW1, 1, mean)
      gw2=apply(GW2, 1, mean)
      gw=cbind(gw1, gw2)
      w=w-eta*gw
      v=v-eta*gv
    }
    ncount=ncount+1
  }
  plot(c(1:length(MSE)), MSE, xlab="Epoch", type='l', lwd=3)
  return(list(w=w, v=v, MSE=mean(E), nIter=ncount,
              Output=min(tg1)+1.25*(OO-0.1)*(max(tg1)-min(tg1))))
}
my=minEReg(Xtrain,Ytrain,eta = 0.5,eps = 0.01,epc = 5000)

#REGRESI LINIER BERGANDA
lm.fit <- glm(Ytrain~., data = as.data.frame(Xtrain))
summary_reg=summary(lm.fit)
summary_reg

#NEURALNET
maxs=apply(traindata, 2, max)
mins=apply(traindata, 2, min)
scaled=as.data.frame(scale(traindata,
                              center = mins,
                              scale = maxs - mins))
library(neuralnet)
set.seed(7)
n=names(as.data.frame(traindata))
f=as.formula(paste("PPM ~", paste(n[!n %in% "PPM"], collapse = " + ")))
nn=neuralnet(f, data = scaled, hidden = 3, linear.output =T)
plot(nn)
nn
nn2 <- neuralnet(f, data = scaled, hidden = c(3,2), linear.output =T)
plot(nn2)
nn3 <- neuralnet(f, data = scaled, hidden = c(3,2,2), linear.output =T)
plot(nn3)
nn.2<- neuralnet(f, data = scaled, hidden = 4, linear.output =T)
plot(nn.2)
nn.3<- neuralnet(f, data = scaled, hidden = 5, linear.output =T)
plot(nn.3)

#PERBANDINGAN
#SSE
SST=sum((Ytrain-mean(Ytrain))^2)
SST
SSE.my=sum((Ytrain-my$Output)^2)
SSE.my
SSE.reg=sum((Ytrain-lm.fit$fitted.values)^2)
SSE.reg
SSE.nn=sum((c(nn$net.result[[1]])*(20.32-5.07)+5.07-Ytrain)^2)
SSE.nn
SSE.nn2=sum((c(nn2$net.result[[1]])*(20.32-5.07)+5.07-Ytrain)^2)
SSE.nn2
SSE.nn3=sum((c(nn3$net.result[[1]])*(20.32-5.07)+5.07-Ytrain)^2)
SSE.nn3
SSE.nn.2=sum((c(nn.2$net.result[[1]])*(20.32-5.07)+5.07-Ytrain)^2)
SSE.nn.2
SSE.nn.3=sum((c(nn.3$net.result[[1]])*(20.32-5.07)+5.07-Ytrain)^2)
SSE.nn.3

#RSQUARE
Rsq.my=1-(SSE.my/SST)
Rsq.my
Rsq.reg=1-(SSE.reg/SST)
Rsq.reg
Rsq.nn=1-(SSE.nn/SST)
Rsq.nn
Rsq.nn2=1-(SSE.nn2/SST)
Rsq.nn2
Rsq.nn3=1-(SSE.nn3/SST)
Rsq.nn3
Rsq.nn.2=1-(SSE.nn.2/SST)
Rsq.nn.2
Rsq.nn.3=1-(SSE.nn.3/SST)
Rsq.nn.3

#MSE
MSE.my=SSE.my/ntrain
MSE.my
MSE.reg=SSE.reg/ntrain
MSE.reg
MSE.nn=SSE.nn/ntrain
MSE.nn
MSE.nn2=SSE.nn2/ntrain
MSE.nn2
MSE.nn3=SSE.nn3/ntrain
MSE.nn3
MSE.nn.2=SSE.nn.2/ntrain
MSE.nn.2
MSE.nn.3=SSE.nn.3/ntrain
MSE.nn.3

#W DAN V
w.nn=matrix(nn$result.matrix[4:15],ncol = 3)
v.nn=matrix(nn$result.matrix[16:19],ncol = 1)

w.nn.2=matrix(nn.2$result.matrix[4:19],ncol = 4)
v.nn.2=matrix(nn.2$result.matrix[20:24],ncol = 1)

w.nn.3=matrix(nn.3$result.matrix[4:23],ncol = 5)
v.nn.3=matrix(nn.3$result.matrix[24:29],ncol = 1)

w1.nn2=matrix(nn2$result.matrix[4:15],ncol = 3)
w2.nn2=matrix(nn2$result.matrix[16:23],ncol = 2)
v.nn2=matrix(nn2$result.matrix[24:26],ncol = 1)

w1.nn3=matrix(nn3$result.matrix[4:15],ncol = 3)
w2.nn3=matrix(nn3$result.matrix[16:23],ncol = 2)
w3.nn3=matrix(nn3$result.matrix[24:29],ncol = 2)
v.nn3=matrix(nn3$result.matrix[30:32],ncol = 1)

#MAPE
minEReg.test<-function (x,tg1,w,v){
  n=nrow(x)
  minx=matrix(rep(apply(x, 2, min), n), nrow=n, byrow=T)
  maxx=matrix(rep(apply(x, 2, max), n), nrow=n, byrow=T)
  rangex=maxx-minx
  x=cbind((x-minx)/rangex, rep(1,n))
  tg=0.1+0.8*(tg1-min(tg1))/(max(tg1)-min(tg1))
  OO=NULL
  for(i in 1:n){
    nh=x%*%w
    oh=1/(1+exp(-nh))
    x2=cbind(oh,rep(1,n))
    no=x2%*%v
    oo=1/(1+exp(-no))
  }
  return(list(w=w, v=v, Output=min(tg1)+1.25*(oo-0.1)*(max(tg1)-min(tg1))))
}
my_=minEReg.test(Xtest,Ytest,my$w,my$v)
MAPE.my_=mean(abs((Ytest-my.$Output)/Ytest))*100
MAPE.my_

for (i in 1:nrow(Xtest)) {
  Yhatest_reg[j]=summary_reg$coefficients[[1]]+Xtest[i,1]*summary_reg$coefficients[[(2)]]+Xtest[i,2]*summary_reg$coefficients[[(3)]]+Xtest[i,3]*summary_reg$coefficients[[(4)]]
}
MAPE.reg_=mean(abs((Ytest-Yhatest_reg)/Ytest))*100
MAPE.reg_

NN_1<-function (x,tg1,w,v){
  n=nrow(x)
  minx=matrix(rep(apply(x, 2, min), n), nrow=n, byrow=T)
  maxx=matrix(rep(apply(x, 2, max), n), nrow=n, byrow=T)
  rangex=maxx-minx
  x=cbind(rep(1,n),(x-minx)/rangex)
  tg=0.1+0.8*(tg1-min(tg1))/(max(tg1)-min(tg1))
  for(i in 1:n){
    nh=x%*%w
    oh=1/(1+exp(-nh))
    x2=cbind(rep(1,n),oh)
    no=x2%*%v
    oo=1/(1+exp(-no))
  }
  return(list(x=x,w=w,v=v,Output=min(tg1)+1.25*(oo-0.1)*(max(tg1)-min(tg1))))
}
nn_=NN_1(Xtest,Ytest,w.nn,v.nn)
MAPE.nn_=mean(abs((Ytest-nn_$Output)/Ytest))*100
MAPE.nn_

NN_2<-function (x,tg1,w1,w2,v){
  n=nrow(x)
  minx=matrix(rep(apply(x, 2, min), n), nrow=n, byrow=T)
  maxx=matrix(rep(apply(x, 2, max), n), nrow=n, byrow=T)
  rangex=maxx-minx
  x=cbind(rep(1,n),(x-minx)/rangex)
  tg=0.1+0.8*(tg1-min(tg1))/(max(tg1)-min(tg1))
  for(i in 1:n){
    nh1=x%*%w1
    oh1=1/(1+exp(-nh1))
    x2=cbind(rep(1,n),oh1)
    nh2=x2%*%w2
    oh2=1/(1+exp(-nh2))
    x3=cbind(rep(1,n),oh2)
    no=x3%*%v
    oo=1/(1+exp(-no))
  }
  return(list(x=x,w1=w1,w2=w2,v=v,Output=min(tg1)+1.25*(oo-0.1)*(max(tg1)-min(tg1))))
}
nn_2=NN_2(Xtest,Ytest,w1.nn2,w2.nn2,v.nn2)
MAPE.nn_2=mean(abs((Ytest-nn_2$Output)/Ytest))*100
MAPE.nn_2

NN_3<-function (x,tg1,w1,w2,w3,v){
  n=nrow(x)
  minx=matrix(rep(apply(x, 2, min), n), nrow=n, byrow=T)
  maxx=matrix(rep(apply(x, 2, max), n), nrow=n, byrow=T)
  rangex=maxx-minx
  x=cbind(rep(1,n),(x-minx)/rangex)
  tg=0.1+0.8*(tg1-min(tg1))/(max(tg1)-min(tg1))
  for(i in 1:n){
    nh1=x%*%w1
    oh1=1/(1+exp(-nh1))
    x2=cbind(rep(1,n),oh1)
    nh2=x2%*%w2
    oh2=1/(1+exp(-nh2))
    x3=cbind(rep(1,n),oh2)
    nh3=x3%*%w3
    oh3=1/(1+exp(-nh3))
    x4=cbind(rep(1,n),oh3)
    no=x4%*%v
    oo=1/(1+exp(-no))
  }
  return(list(x=x,w1=w1,w2=w2,w3=w3,v=v,Output=min(tg1)+1.25*(oo-0.1)*(max(tg1)-min(tg1))))
}
nn_3=NN_3(Xtest,Ytest,w1.nn3,w2.nn3,w3.nn3,v.nn3)
MAPE.nn_3=mean(abs((Ytest-nn_3$Output)/Ytest))*100
MAPE.nn_3

nn_.2=NN_1(Xtest,Ytest,w.nn.2,v.nn.2)
MAPE.nn_.2=mean(abs((Ytest-nn_.2$Output)/Ytest))*100
MAPE.nn_.2

nn_.3=NN_1(Xtest,Ytest,w.nn.3,v.nn.3)
MAPE.nn_.3=mean(abs((Ytest-nn_.3$Output)/Ytest))*100
MAPE.nn_.3
