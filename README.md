# This is the repository of our project

### Motivation

現在的媒體產業，其商業模式乃是透過網路瀏覽量轉換為廣告收益。

故，每篇文章的瀏覽量與其創造之收益息息相關。

而Facebook的讚數代表該文章於網路擴散的可能性。

本計劃嘗試透過機器學習方式，將既有資料餵入Spark，製作出預測文章讚數的應用

### News crawler info (by Chia Hua Lee)
這是我目前找到可能最接近做爬蟲新聞主題可以參考的資料

主題: 他們是蒐集某個新聞網站兩年來的資料，目標是成功去預測某則新聞被分享的次數

#### 參考資料
1. [Attributes](http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)

2. [Machine learning](http://cs229.stanford.edu/proj2015/328_report.pdf)

### News crawler (by張逸)
用來爬資料(新頭殼) 的程式碼，最初的版本能夠抓下 標題/記者/內文/讚數
[Source Code] (https://github.com/j129008/newsCrawler)

### Summary:

他們最好的情況會有70%的成功預測率(使用random forest)，也有提到model要再improve的空間小，要improve的話會是feature的選擇，最後還有提出一點點可能可以改善的方法，如果沒理解錯的話感覺就是語意分析。

我之後還會繼續看別的~~~
