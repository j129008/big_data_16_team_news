# This is the repository of our project

### news crawler info (by Chia Hua Lee)
這是我目前找到可能最接近做爬蟲新聞主題可以參考的資料

主題: 他們是蒐集某個新聞網站兩年來的資料，目標是成功去預測某則新聞被分享的次數

這裡提供兩個連結:

1. [dataset連結](http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) (可以參考他們所選用的attribute)

2. [研究連結](http://cs229.stanford.edu/proj2015/328_report.pdf) (5頁pdf，可以參考他們的machine learning，全都是上課老師提過的)


### 小小summary:

他們最好的情況會有70%的成功預測率(使用random forest)，也有提到model要再improve的空間小，要improve的話會是feature的選擇，最後還有提出一點點可能可以改善的方法，如果沒理解錯的話感覺就是語意分析。

我之後還會繼續看別的~~~


### news crawler (by張逸)
[Code Link] (https://github.com/j129008/newsCrawler)

### 情境

現在的媒體產業，其商業模式乃是透過網路瀏覽量轉換為廣告收益。

故，每篇文章的瀏覽量與其創造之收益息息相關。

而Facebook的讚數代表該文章於網路擴散的可能性。

本計劃嘗試透過機器學習方式，將既有資料餵入Spark，製作出預測文章讚數的應用
