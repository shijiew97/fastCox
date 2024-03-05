library(survival)
library(reticulate)

df = na.omit(lung)
covariates = c("age", "sex",  "ph.karno", "ph.ecog", "wt.loss")
p = length(covariates)
X = df[covariates]
N = 20
size = 1000

file = "C:\\Users\\Shijie Wang\\Desktop\\Research Notes\\GBS and NPMLE code\\GBS_Cox_proportional_hazard_model.py"
reticulate::source_python(file,envir = NULL,convert = FALSE)
Beta = py$B

boot_cox = matrix(NA, nrow=size, ncol=p)
for(i in 1:size){
  set.seed(2021+i^2)
  b_index = sample(nrow(df),size=nrow(df),replace=T)
  res.cox = coxph(Surv(time, status) ~ age + sex + ph.karno + ph.ecog + wt.loss , data = df[b_index,])
  boot_cox[i,] = res.cox$coefficients
}

B = matrix(unlist(Beta),nrow=1)
B_WLB = matrix(boot_cox,nrow=1)
Theta_dist = data.frame("value"=t(cbind(B,B_WLB)))
Theta_dist["Theta"] = factor(rep(rep(paste("b",1:p,sep=""),each=size),times=2),levels=paste("b",1:10,sep=""))
Theta_dist["Method"] = c(rep("GBS", size*p),rep("WLB", size*p))

library(ggplot2)
ggplot(data=Theta_dist)+geom_boxplot(aes(x=Theta,y=value,color=Method))+ylim(-2,2)+theme_minimal()
