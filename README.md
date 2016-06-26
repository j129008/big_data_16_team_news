數據科學與大數據分析學期計畫
========

## 第八組成員介紹

###### 104753032 資科碩一 張逸
###### 105753013 資科碩一 姚德謙
###### 100401048 新聞四 李家華


## 報告簡介

###### 主題:手機廣告點擊率預測 (Click-Through Rate Prediction)<br/>
###### 資料集來源: <https://www.kaggle.com/c/avazu-ctr-prediction/data> <br/>


## 報告目的  
   
    
    本學期計畫目的在於以團隊的方式體驗大數據分析的流程，組員以課堂所學習到的機器學習演算
    法為基礎，從網路上搜索感興趣的資料集，並利用現今最流行的大數據運算框架Spark以及內建
    的MLlib進行巨量資料運算，去實作分類或分群的預測。


## 報告摘要  
   

    
    本組實際參與了於知名數據競賽平台kaggle上的一個已結案的手機廣告點擊率比賽，在經過特徵
    值挑選、各種模型的試驗、參數調整後，最終所選擇的分類器是隨機森林，我們擷取的是它善於
    處理高維度的特性，以及不會過度擬合的兩大優點來進行建模。在預測階段，於本機端執行交叉
    驗證後實驗的最佳結果AUC為0.68，而在kaggle平台上經測試資料驗證後的準確率，本組的機器
    學習模型所獲得的最低損失函數log-loss的數值則為0.76。
    
## 預測最佳成果
#### * Area Under Curve

![image](https://github.com/chiahualee/temp/blob/master/AUC.png)
#### * 提交至kaggle所得到log loss的值 

![image](https://github.com/chiahualee/temp/blob/master/losslog.png)

