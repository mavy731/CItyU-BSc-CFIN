setwd("C:/Users/USER/Documents/A_study/2018-2019 sem B/EF4822/PS1")
library(quantmod)

##download stock data
getSymbols('AMZN', from='2009-01-02',to='2009-02-01')
getSymbols('ADBE', from='2009-01-02',to='2009-02-01')
getSymbols('JNJ', from='2009-01-02',to='2009-02-01')

##compute simple return
AMZNreturn=diff(AMZN$AMZN.Close)/lag(AMZN$AMZN.Close)
ADBEreturn=diff(ADBE$ADBE.Close)/lag(ADBE$ADBE.Close)
JNJreturn=diff(JNJ$JNJ.Close)/lag(JNJ$JNJ.Close)

##rename simple return & delete missing value
colnames(AMZNreturn)[1]="Simple return"
colnames(ADBEreturn)[1]="Simple return"
colnames(JNJreturn)[1]="Simple return"
AMZNreturn=AMZNreturn[-1,]
ADBEreturn=ADBEreturn[-1,]
JNJreturn=JNJreturn[-1,]

##compute log return & rename
lnAMZN=log(AMZNreturn+1)
lnADBE=log(ADBEreturn+1)
lnJNJ=log(JNJreturn+1)
colnames(lnAMZN)[1]="log return"
colnames(lnADBE)[1]="log return"
colnames(lnJNJ)[1]="log return"

##compute sample mean, variance, max, min of simple return
SimpleMeanAMZN=mean(AMZNreturn)
SimpleVarAMZN=var(AMZNreturn)
SimpleMaxAMZN=max(AMZNreturn)
SimpleMinAMZN=min(AMZNreturn)
SimpleMeanAMZN
SimpleVarAMZN
SimpleMaxAMZN
SimpleMinAMZN

SimpleMeanADBE=mean(ADBEreturn)
SimpleVarADBE=var(ADBEreturn)
SimpleMaxADBE=max(ADBEreturn)
SimpleMinADBE=min(ADBEreturn)
SimpleMeanADBE
SimpleVarADBE
SimpleMaxADBE
SimpleMinADBE

SimpleMeanJNJ=mean(JNJreturn)
SimpleVarJNJ=var(JNJreturn)
SimpleMaxJNJ=max(JNJreturn)
SimpleMinJNJ=min(JNJreturn)
SimpleMeanJNJ
SimpleVarJNJ
SimpleMaxJNJ
SimpleMinJNJ

##compute sample mean, variance, max, min of log return
LogMeanAMZN=mean(lnAMZN)
LogVarAMZN=var(lnAMZN)
LogMaxAMZN=max(lnAMZN)
LogMinAMZN=min(lnAMZN)
LogMeanAMZN
LogVarAMZN
LogMaxAMZN
LogMinAMZN

LogMeanADBE=mean(lnADBE)
LogVarADBE=var(lnADBE)
LogMaxADBE=max(lnADBE)
LogMinADBE=min(lnADBE)
LogMeanADBE
LogVarADBE
LogMaxADBE
LogMinADBE

LogMeanJNJ=mean(lnJNJ)
LogVarJNJ=var(lnJNJ)
LogMaxJNJ=max(lnJNJ)
LogMinJNJ=min(lnJNJ)
LogMeanJNJ
LogVarJNJ
LogMaxJNJ
LogMinJNJ

##compute covariance 
cor(lnAMZN,lnADBE)
cor(lnAMZN,lnJNJ)
cor(lnADBE,lnJNJ)

##plot densities of each stock
par(mfrow=c(2,2))
hist(AMZNreturn, main="Histogram of AMZN Daily Stock Return")
hist(ADBEreturn, main="Histogram of ADBE Daily Stock Return")
hist(JNJreturn, main="Histogram of JNJ Daily Stock Return")

##hypothesis testing
t.test(AMZNreturn)
t.test(ADBEreturn)
t.test(JNJreturn)

