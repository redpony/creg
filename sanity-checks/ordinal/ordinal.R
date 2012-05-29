library(rms)
d<-read.table("shuttle.txt",header=T)
oreg<-lrm(Resp ~ Temp + Date, data=d, na.action=na.pass)
coef(oreg)
