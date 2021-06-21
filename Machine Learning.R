#Machine Learning

library(rpart)
library(rpart.plot)
library(nnet)
library(randomForest)
library(mlbench)
library(caret)
library(kernlab)

# 1. 위스콘신 유방암 데이터셋을 대상으로 분류기법 3개를 적용하여 기법별 결과를 비교하시오.

# 종속변수는diagnosis: Benign(양성), Malignancy(악성)
# 사용기법은 1) 의사결정트리 2)인공신경망분석 3)랜덤포레스트

# 데이터 샘플링
rawData <- read.csv("wisc_bc_data.csv", header = T)
diagnosis.df <- rawData[-1] #필요없는 id부분 삭제
head(diagnosis.df);str(diagnosis.df) #데이터 확인

# 훈련-검정 데이터셋으로 분리
set.seed(1)
idx <- sample(1:nrow(diagnosis.df), 0.7*nrow(diagnosis.df))
train <- diagnosis.df[idx,]
test <- diagnosis.df[-idx,]
dim(train);dim(test)


# 1 - 1. 의사결정트리방식 사용 - rpart{rpart}와 rpart.plot{rpart.plot}

rpart.diag <- rpart(diagnosis ~ ., data = train) #의사결정트리 생성

summary(rpart.diag) #398개 트리와 중요변수들 확인가능

rpart.diag$variable.importance #각 변수가 암진단에 미치는 영향력 확인가능

rpart.plot(rpart.diag) #시각화.

# 예측치 산출
rpart.pred <- predict(rpart.diag, newdata = test)
rpart.pred2 <- ifelse(rpart.pred[,1] >= 0.5, "B", "M") #범주화

table(rpart.pred2, test$diagnosis) #혼돈매트릭스를 통해 예측치와 검정데이터의 실제값 확인
(100+64) / 171 #분류정확도 95% 확인

# 1 - 2. 인공신경망 사용 - nnet{nnet}

# 2차 데이터 샘플링
# 인공신경망에 사용할 수 있도록 diagnosis의 인자값을 문자에서 숫자로 변환
train$diagnosis[train$diagnosis == 'B'] <- 1
train$diagnosis[train$diagnosis == 'M'] <- 2
test$diagnosis[test$diagnosis == 'B'] <- 1
test$diagnosis[test$diagnosis == 'M'] <- 2
train$diagnosis <- as.numeric(train$diagnosis) #캐릭터값에서 변경
test$diagnosis <- as.numeric(test$diagnosis)

# 추가로, 데이터를 인공 신경망에 사용하기 위해 정규화과정 필요

# 정규화 함수
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))}
train.norm <- as.data.frame(sapply(train, normalize))
test.norm <- as.data.frame(sapply(test, normalize))

nnet.diag1 <- nnet(diagnosis ~ ., data = train.norm, size = 1) #33개 가중치 확인
nnet.diag2 <- nnet(diagnosis ~ ., data = train.norm, size = 2) #65개 가중치
nnet.diag3 <- nnet(diagnosis ~ ., data = train.norm, size = 3) #97개 가중치

summary(nnet.diag1) #가중치 수치 확인 가능
summary(nnet.diag2)
summary(nnet.diag3)

# 가중치가 너무 많을경우 연산량이 증가하므로 여기서는 은닉노드층이 2인 쪽을 사용해 작업 수행

# 예측치 산출과 분류정확도 확인
nnet.pred <- predict(nnet.diag2, newdata = test.norm, type = "raw") #예측값 산출.
nnet.pred <- round(nnet.pred, 0) #소수점 정리
table(nnet.pred, test.norm$diagnosis)
(103+61) / 171 #분류정확도 95% 확인. 매번 예측값이 다르지만 정확도는 1%내외로 차이남

# 1 - 3. 랜덤포레스트 - randomForest{randomForest}

#분류용 랜덤포레스트 사용을 위해 목적변수인 diagnosis를 요인(factor)화 할것
train$diagnosis <- as.factor(train$diagnosis)
test$diagnosis <- as.factor(test$diagnosis)

# 분류 목적일 때는 m=sqrt(p) 개의 설명변수를 사용하는것이 일반적임. 
str(train) #31개 변수중 목적변수를 제외하면 30.
sqrt(30) #5.4개이므로 변수는 5개로 사용
set.seed(1)

rf.diag <- randomForest(diagnosis ~ ., data = train, mtry = 5, importance = T) 
rf.diag # OOB 5.03확인가능

plot(rf.diag) #시각화자료로 확인. 트리 수가 약 50을 넘어간 이후로는 오류가 어느 정도 안정화되며 250을 넘으면 더욱 안정화됨 

importance(rf.diag)
#중요 변수가 무엇인지 파악 가능. 상위 3개 확인

varImpPlot(rf.diag) 
#중요변수 시각화를 통해 가장 중요한 변수는 perimeter worst와 points worst, points_mean임을 확인할 수 있음.

# 예측치 산출과 분류정확도 확인
rf.pred <- predict(rf.diag, newdata = test)
table(rf.pred, test$diagnosis)
(103+66) / 171 #98.8 확인 

# 세가지 방법으로 분류결과 가장 분류정확도가 높게 산출되는것은 RandomForest 방식이므로 이쪽을 사용해 분류하는것이 효과적임

# 2. mlbench 패키지 내 BostonHousing 데이터셋을 대상으로 예측기법 3개를 적용하여 기법별 결과를 비교하시오.
# 종속변수는MEDV 또는CMEDV를사용

# 사용 기법은 1)다중회귀분석, 2)SVM, 3)랜덤포레스트

# 데이터 샘플링
data("BostonHousing") #
rawData.Boston <- BostonHousing
str(rawData.Boston) #총 14개 변수를 가진 데이터셋임을 확인. 목적변수는 medv로 사용

# 훈련-검정 데이터셋으로 분리
set.seed(22)
idx <- sample(1:nrow(rawData.Boston), 0.7*nrow(rawData.Boston))
train.Boston <- rawData.Boston[idx,]
test.Boston <- rawData.Boston[-idx,]
dim(train.Boston);dim(test.Boston)

# 2 - 1. 다중회귀분석 stats{lm}

lm.Boston <- lm(medv ~ ., data = train.Boston)
summary(lm.Boston)

#0.71 설명력확인. P값은 유의수준보다 낮지만 검정통계량 F값이 65이며  몇몇 회귀계수들에 문제가 있음을 알 수 있음.
#때문에 전진후진법을 모두 실시해서 교정을 실시함

lm.Boston.fit <- step(lm.Boston, method = "both")
summary(lm.Boston.fit) 
#설명력은 0.71로 동일함.
#검정통계량 F값이 77.7로 크게 상승했으며 P 밸류값은 여전히< 2.2e-16으로 나타남. 회귀계수들도 안정화되었음을 확인 가능

#MSE(평균제곱오차) 산출

#수치 예측을 실시하므로 혼돈매트릭스를 통한 분류정확도 비교는 실시하지 않고, 예측값에서 MSE 산출을 수행함
lm.pred <- predict(lm.Boston.fit, newdata = test.Boston)
lm.MSE <- sqrt(mean((lm.pred - test.Boston$medv)^2)) #평균제곱오차(MSE) 계산하기.
#약 3.966으로 산출

# 2 - 2. SVM - kernlab{ksvm} 

#커널트릭을 이용해서 가장 많은 데이터점을 포함하는 튜브를 찾을것

#세 종류의 커널트릭을 사용한 뒤 더 많은 서포트벡터를 포함하는 쪽을 수치 예측에 사용할것

svm.Boston.G <- ksvm(medv ~ ., data = train.Boston) #기본 가우시안 함수 사용.
svm.Boston.G #서포트벡터 243개 확인
svm.Boston.L <- ksvm(medv ~ ., data = train.Boston, kernal = "vanila dot") #리니어 함수 사용
svm.Boston.L #서포트벡터 237개 확인
svm.Boston.P <- ksvm(medv ~ ., data = train.Boston, kernal = "poly dot") #polynomial 함수 사용
svm.Boston.P # 서포트벡터 238개 확인

#미세하지만 가우시안을 사용한쪽이 우세하게 나타나므로 이쪽을 수치 예측에 사용
svm.pred <- predict(svm.Boston.G, newdata = test.Boston)
svm.MSE <- sqrt(mean((svm.pred - test.Boston$medv)^2))
#약 3.036으로 나타남

# 2 - 3. 랜덤 포레스트 - randomForest{randomForest}

# 여기서는 수치 추정을 목적으로 실행하므로 목적변수는 따로 요인화를 거치지않음

# 분류 목적일 때는 m=p/3 개의 설명변수를 사용하는것이 일반적
str(train.Boston) #14개 변수중 목적변수를 제외하면 13.
#4.3333...이므로 4개로 사용

set.seed(1)
rf.Boston <- randomForest(medv ~ ., data = train.Boston, mtry = 4, importance = T) 
rf.Boston # 잔차 11정도로 나타남

plot(rf.Boston) #시각화자료로 확인. 트리수가 약 50을 넘어간 이후로는 오류가 매우 안정화됨.

importance(rf.Boston)
#중요 변수가 무엇인지 파악 가능. 가장 큰 상위 2개 확인

varImpPlot(rf.Boston) 
#중요변수 시각화를 통해 가장 중요한 변수는 rm와 istat임을 확인할수있음.

# 예측치 산출과 분류정확도 확인
rf.pred <- predict(rf.Boston, newdata = test.Boston)
rf.MSE <- sqrt(mean((rf.pred - test.Boston$medv)^2))
#2.886으로 산출됨. 

#세 기법의 결과를 놓고 비교했을때 평균제곱오차(MSE)값이 가장 작은 기법은 랜덤포레스트로, 실제값과 오차가 2.886정도로 가장 작다.
#때문에 세 기법중 랜덤포레스트 기법를 사용해서 수치를 예측하는것이 가장 적절하다.