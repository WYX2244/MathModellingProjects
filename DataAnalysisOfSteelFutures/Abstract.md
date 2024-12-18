### Research on the Interaction Patterns of Futures Prices Based on the Characteristics of Data Geometry

#### Abstract
With the continuous development of the global steel industry, the trading market for ferrous industry products such as coking coal, iron ore, and steel futures continues to expand. Due to the varying and dynamic prices of these commodities, they attract numerous domestic and international investors. Exploring the interconnection and mutual influence of futures price changes has profound implications for economic development and investment decisions.

This study selects 9 types of futures—rebar, iron ore, stainless steel, hot-rolled coil, ferrosilicon, coke, coking coal, silicon manganese, and wire rod—to analyze their price fluctuations and establish relevant numerical models. Daily closing price data of major contracts from 2020 to 2022 were extracted from public futures exchange databases for analysis after preprocessing.

From the perspective of one of machine learning’s hottest topics—low-dimensional manifold embedding in high-dimensional spaces—the dataset used in this issue represents a low-dimensional flow within a high-dimensional space. Using the concept of potential fields and a relatively stable sliding window size, a secondary quadratic manifold embedding is proposed to determine the optimal investment strategy.


#### Research Objectives

##### Objective 1
The study initially conducted simple linear regression analysis on two futures categories, calculating their **Pearson correlation coefficient** to preliminarily determine linearity and correlation. Subsequently, an **ADF test** was performed on the time series of futures prices, confirming their long-term equilibrium relationships. Using factor analysis, variables with good linear relationships were identified, and **multiple linear regression analysis** was conducted on all futures prices, resulting in a correlation coefficient \( R^2 = 0.988 \), demonstrating strong linear correlations among the nine futures categories.

##### Objective 2
Nonlinear regression analysis was applied, introducing price increments into the regression model. The quadratic relationships between rebar prices and other futures prices revealed a correlation coefficient \( R^2 = 0.991 \), indicating a robust nonlinear correlation. A quadratic nonlinear model \( x^T A x = 1 \) was established to optimize the model for high-dimensional geometric flow-based correlations. After standardizing futures prices, a specific matrix \( A \) was derived for further calculations.

##### Objective 3
Using the quadratic nonlinear model as a benchmark, predicted values were compared to actual values for different futures categories. When the difference was less than 0.1, it was considered within the quadratic nonlinear manifold window; otherwise, it was excluded. Machine learning methods were employed to visualize the spatial distribution of data points, analyzing trends in two-dimensional projection. A localized weighted interpolation method was used to refine predictions within the manifold windows.

##### Objective 4
Using the quadratic nonlinear model established in Objective 2, standardization of price data was applied to high-dimensional spaces, forming a multi-item laddering curve. By setting step sizes and utilizing step-wise back-calculations, changes in actual price variations were obtained. Based on these predictions, an investment strategy was formulated: buying during high-growth periods and selling during declines, yielding a maximum return rate of **161.27%** among the strategies provided.



#### Here's the problem 
Problem B: Correlation of Futures Prices

Many financial instruments exhibit intrinsic correlations. Identifying these relationships from price and volume data is an interesting challenge. For example, in the "Coal-Coke-Steel-Mining" sector on Wind, there are nine categories: rebar, iron ore, stainless steel, hot-rolled coil, ferrosilicon, coking coal, coke, silicon manganese, and wire rod. Some of these products have upstream-downstream relationships. For instance, a certain quantity of coking coal, coke, and iron ore can be used to produce rebar and other products. Exploring the relationships between different products through their prices has always been of interest to investors. Your task is to uncover the potential relationships among these products based on their daily price data.

Problems

1. Develop an algorithm to establish linear relationships among these products using daily price data and model these linear relationships.

2. Can you identify nonlinear relationships among these products?

3. These relationships are typically valid for a certain period. Can you identify a criterion to determine the valid time window for these relationships?

4. Based on these relationships, construct an investment portfolio strategy.

Data Sources

- Rebar, Hot-Rolled Coil, Stainless Steel, Wire Rod**  
  *(Shanghai Futures Exchange)*  
  [Link to data](https://www.shfe.com.cn/statements/dataview.html?paramid=delaymarket_rb)

- Iron Ore, Coking Coal, Coke**  
  *(Dalian Commodity Exchange)*  
  [Link to data](http://www.dce.com.cn/dalianshangpin/xqsj/lssj/index.html)

- Silicon Manganese, Ferrosilicon**  
  *(Zhengzhou Commodity Exchange)*  
  [Link to data](http://www.czce.com.cn/cn/jysj/lshqxz/H770319index_1.htm)
