d<-read.table("mpg-train.txt",header=T)
lreg = lm(d$MPG ~ d$HP + d$Acc + d$Cyl + d$Dis + d$Weight)
coef(lreg)
