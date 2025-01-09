# Introduction
This project aims to analyze customer behavior through clustering techniques, focusing on key metrics like monetary value, purchase frequency, and recency. By segmenting customers into distinct groups, the goal is to uncover actionable insights that drive personalized marketing strategies. This approach ensures better customer engagement, improves retention, and supports data-driven business decisions.
# Data
Data used in the project is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

**Background**

*This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.*

**Key Variable Information**
- **InvoiceNo:** Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation. 
- **StockCode:** Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product. 
- **Quantity:** The quantities of each product (item) per transaction. Numeric.	
- **InvoiceDate:** Invice date and time. Numeric. The day and time when a transaction was generated. 
- **UnitPrice:** Unit price. Numeric. Product price per unit in sterling (Â£). 
- **CustomerID:** Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer

# Analysis
I started my analysis with importing libraries. For the analysis, i used **pandas** for data manipulation and visualization , **seaborn** for data visualization  and **scikit-learn** for clustering and scaling data. 
```python
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

pd.options.display.float_format = '{:20.2f}'.format
df = pd.read_excel('C:/Users/mertk/Data Science/Python/Datasets/Excel/online_retail_II.xlsx' , sheet_name= 0 )
```
### First three rows of data looks like this :
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Invoice</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>Price</th>
      <th>Customer ID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>489434</td>
      <td>85048</td>
      <td>15CM CHRISTMAS GLASS BALL 20 LIGHTS</td>
      <td>12</td>
      <td>2009-12-01 07:45:00</td>
      <td>6.95</td>
      <td>13085.00</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>489434</td>
      <td>79323P</td>
      <td>PINK CHERRY LIGHTS</td>
      <td>12</td>
      <td>2009-12-01 07:45:00</td>
      <td>6.75</td>
      <td>13085.00</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>489434</td>
      <td>79323W</td>
      <td>WHITE CHERRY LIGHTS</td>
      <td>12</td>
      <td>2009-12-01 07:45:00</td>
      <td>6.75</td>
      <td>13085.00</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>

It looks like there are NaN values for **Customer ID** .

```python
df['Customer ID'].isna().sum()
```
 107927

 I will be removing all rows with **NaN CustomerID** values since they will not be beneficial to my analysis.

According to data source;
-  **InvoiceNo** is 6-digit integral number , values starting with *'c'* indicate cancelation therefore i will remove them from the data set.
- **StockCode** is 5-digit integral number.After exploring it a bit , i found that there are bunch of different stock codes that don't fit the description. Further investigation revealed that only **SP1002** serves my analysis since others are mostly for postal expanses, bank deposits and test inputs.
    ```python
    df[df['StockCode'].str.match(pat='^\\d{5}[a-zA-Z]*$')==False]['StockCode'].unique()
    ```
    ['POST', 'C2', 'M', 'BANK CHARGES', 'TEST001', 'TEST002', 'PADS',
       'ADJUST', 'D', 'ADJUST2', 'SP1002']

**Code below processes data and prepares it for further analysis:**
```python
df_nona = df.dropna(subset=['Customer ID']).copy()
df_nona[['Invoice','StockCode']] = df_nona[['Invoice','StockCode']].astype('str')
mask1 = (df_nona['Invoice'].str.match(pat='^\\d{6}$') == True)        
df_nona[df_nona['StockCode'].str.match(pat='^\\d{5}[a-zA-Z]*$')==False]['StockCode'].unique()
mask2 = (df_nona['StockCode'].str.match(pat='^\\d{5}[a-zA-Z]*$')==True) | (df_nona['StockCode'].str.match(pat='^SP1002$')==True)
df_nona = df_nona[mask1]
df_stock= df_nona[mask2].copy()
```
Now dataset has no missing values and both "Invoice" and "StockCode" are in correct form according to the data source.

### Let's get to the  **RFM Analysis**. I start of by creating the necessary features ;
- Multiplied 'Quantity' with 'Price' column to create 'Amount Spent' column.
- Grouped dataset by 'Customer ID' to find out total amount spent , total number of unique purchases and date of the last purchase made.
- To create 'Recency' feature , subtracted each last invoice date from final date.
    ```python
    df_stock['Amount Spent'] =df_stock['Quantity']*df_stock['Price']
    dfstock_feat = df_stock.groupby('Customer ID',as_index=False).agg(
    MonetaryValue = ('Amount Spent','sum'),
    Frequency = ('Invoice','nunique'),
    LastInvo = ('InvoiceDate','max')
    )

    last_invoice = df_stock['InvoiceDate'].max()
    dfstock_feat['Recency'] = (last_invoice -dfstock_feat['LastInvo']).dt.days
    dfstock_feat.drop('LastInvo',axis=1,inplace=True)
  ```

### Let's check distribution of the features:
```python
plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
sns.histplot(dfstock_feat['MonetaryValue'] , color='salmon',bins=40)
plt.ylim((0,4500))
plt.title('Monetary Value Distribution')
plt.ylabel('')

plt.subplot(1,3,2)
sns.histplot(dfstock_feat['Frequency'] , color='skyblue',bins=100)
plt.title('Frequency Distribution')
plt.ylabel('')

plt.subplot(1,3,3)
sns.histplot(dfstock_feat['Recency'] , color='lightgreen',bins=20)
plt.title('Recency Distribution')
plt.ylabel('')

plt.tight_layout()
plt.grid(False)
plt.show()
```
![Histrogram](/assets/hist1.png)
```python
plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
sns.boxplot(y=dfstock_feat['MonetaryValue'] , color='salmon')
plt.title('Monetary Value Distribution')
plt.ylabel('Monetary Value')

plt.subplot(1,3,2)
sns.boxplot(y=dfstock_feat['Frequency'] , color='skyblue')
plt.title('Frequency Distribution')
plt.ylabel('Frequency')

plt.subplot(1,3,3)
sns.boxplot(y=dfstock_feat['Recency'] , color='lightgreen')
plt.title('Recency Distribution')
plt.ylabel('Recency')

plt.tight_layout()
plt.show()
```
![Boxplot](/assets/boxplot1.png)

Looks like there are outliers for both 'MonetaryValue' and 'Frequency'. To detect outliers i will be using **Interquartile Range (IQR) Method** . Method suggests to calculate middle %50 of the data and points that fall significantly below or above the range are outliers.
```python
#Calculating middle %50 percent portion.
mon_Q3 = dfstock_feat['MonetaryValue'].quantile(0.75)
mon_Q1 = dfstock_feat['MonetaryValue'].quantile(0.25)
mon_Q3_Q1 = mon_Q3-mon_Q1 

fre_Q3 = dfstock_feat['Frequency'].quantile(0.75)
fre_Q1 = dfstock_feat['Frequency'].quantile(0.25)
fre_Q3_Q1 = fre_Q3-fre_Q1
```
```python
# Detecting Monetary Value outliers
mon_upper_bound = mon_Q3 + 1.5 * mon_Q3_Q1
mon_lower_bound = mon_Q1 - 1.5 * mon_Q3_Q1
mon_outliers = dfstock_feat[
    (dfstock_feat['MonetaryValue'] > mon_upper_bound) | 
    (dfstock_feat['MonetaryValue'] < mon_lower_bound)
].copy()

# Detecting Frequency outliers
freq_upper_bound = fre_Q3 + 1.5 * fre_Q3_Q1
freq_lower_bound = fre_Q1 - 1.5 * fre_Q3_Q1
freq_outliers = dfstock_feat[
    (dfstock_feat['Frequency'] > freq_upper_bound) | 
    (dfstock_feat['Frequency'] < freq_lower_bound)
].copy()
# Create a DataFrame excluding outliers
non_outlier_df = dfstock_feat[
    ~dfstock_feat.index.isin(mon_outliers.index) & 
    ~dfstock_feat.index.isin(freq_outliers.index)
].copy()
```
### Plot features again.
![Boxplot2](/assets/boxplot2.png)
*Much better.*
### Let's project our data in 3d plane.
```python
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection = '3d')

scatter = ax.scatter(non_outlier_df['MonetaryValue'], non_outlier_df['Frequency'] , non_outlier_df['Recency'])
plt.title('3D Projection of Customer Data')
ax.set_ylabel('Frequency')
ax.set_xlabel('Monetary Value')
ax.set_zlabel('Recency')
ax.zaxis.labelpad= -30

plt.tight_layout()
plt.show()
```
![3d](/assets/3dcust.png)

Last step before clustering process is to scale our data . Clustering algorithm works better when features are in the same range of values . I will use 'standard scaler' from scikit-learn library.

```python
scaler = StandardScaler()
scaled_l = scaler.fit_transform(non_outlier_df[['MonetaryValue','Frequency','Recency']])
scaled_df = pd.DataFrame(scaled_l , index = non_outlier_df.index , 
                         columns = ['MonetaryValue','Frequency','Recency'])
```
### Now every preprocessing step is completed . We can start 'Clustering'.
First , i will run clustering algorithm in a for loop to determine optimal number of clusters . For loop will run **Kmeans** algorithm 11 times with each time increasing the number of clusters by 1 . For finding optimal number of clusters, i will be using the help of **Inertia** and **Silhouette Score** .
- *Inertia* measures how tightly the data points are grouped within their respective clusters.
- *Silhouette score* evaluates the quality of clustering by measuring how similar a data point is to points in its own cluster compared to points in other clusters.

**Long story short I am looking for low inertia , high silhouette scores.**

```python
max_k = 12 
k_val = range(2,max_k+1)
sil_score = []
inertia = []

for k in k_val : 
    kmean = KMeans(n_clusters=k , max_iter=1000 , random_state= 42 )
    cluster_labels = kmean.fit_predict(scaled_df)
    silho = silhouette_score(scaled_df,cluster_labels)
    sil_score.append(silho)
    inertia.append(kmean.inertia_)
```
![Sil_inertia](/assets/sil-inertia.png)
*Graph above represents Inertia and Silhouette Score collected for each individual for loop run.*

**Looks like 4 clusters works best for our dataset.**
```python
# Running algorithm with 4 clusters and mapping colors to them
kmean = KMeans(n_clusters=4, max_iter=1000, random_state=42)

cluster_label = kmean.fit_predict(scaled_df)
non_outlier_df['Clusters'] = cluster_label

cluster_colors = {0: '#1f77b4',  # Blue
                  1: '#ff7f0e',  # Orange
                  2: '#2ca02c',  # Green
                  3: '#d62728'}  # Red

col = non_outlier_df['Clusters'].map(cluster_colors)
```
### Let's see our clustered data in 3d plane.
![3dclusterd](/assets/clustered3d.png)


### First 3 rows of the clustered dataset looks like this .
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>MonetaryValue</th>
      <th>Frequency</th>
      <th>Recency</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346.00</td>
      <td>169.36</td>
      <td>2</td>
      <td>164</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12347.00</td>
      <td>1323.32</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12348.00</td>
      <td>221.16</td>
      <td>1</td>
      <td>73</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

### Plot each feature for all clusters.  (Distribution of whole clustered dataset at the end)

![Violinnonoutlier](/assets/violinnoncluster.png)

### Characteristics of each cluster: 

- **Cluster 0:** Active, medium-spending frequent buyers.
- **Cluster 1:** Dormant, low-spending infrequent buyers.
- **Cluster 2:** Infrequent, consistent lower spenders.
- **Cluster 3:** High-value, recent big spenders.
##

Now that clustering of non-outliers data is done , i will cluster outliers data manually by clustering them into 3 seperate groups.

```python
# Cluster -3 : Both monetary and frequency outliers
# Cluster -2 : Frequency only outliers.
# Cluster -1 : Monetary only outliers 
mon_freq = freq_outliers.index.intersection(mon_outliers.index)
mon_only_outlier = mon_outliers.loc[~mon_outliers.index.isin(mon_freq)]
freq_only_outlier = freq_outliers.loc[~freq_outliers.index.isin(mon_freq)]
mon_freq_outlier = mon_outliers.loc[mon_freq]

mon_only_outlier['Clusters'] = -1 
freq_only_outlier['Clusters'] = -2 
mon_freq_outlier['Clusters'] = -3 

full_outlier = pd.concat([mon_only_outlier,freq_only_outlier,mon_freq_outlier])
```

### Plot each feature for all outlier clusters.
![Violinoutlier](/assets/violinoutlier.png)

### Characteristics of outlier clusters: 
- **Cluster -1:** Moderate outliers with low to medium spending, infrequent transactions, and low engagement.

- **Cluster -2:** Rare outliers with extremely low monetary value, frequency, and recency, likely inactive customers.

- **Cluster -3:** Extreme outliers with very high spending and transaction frequency but inconsistent recency patterns.
##
### Next step of my analysis is joining non-outlier and outliers data together and naming each cluster.

```python
#Joining data
full_df = pd.concat([non_outlier_df,full_outlier])

cluster_names = {
    -3: "Whales",         # High spending and frequency, inconsistent recency
    -2: "Dormants",       # Extremely low spending and activity
    -1: "Lurkers",        # Low to moderate spending and engagement
    0: "Regulars",        # Moderate spending, consistent activity
    1: "Sleepers",        # Low spending, infrequent activity
    2: "Occasionals",     # Predictable low spending and rare activity
    3: "Elites"           # High spending, frequent, and recent activity
}
```
### Plot full clustered dataset  based on number of members in each cluster and clusters average values.
![Fulldata](/assets/full.png)
##

# Targeted Strategies
Here are my vision of targeted strategies to maximize income , increase traffic and retain loyalty.
### Cluster -3: Whales
**Retain VIPs**:
These are inconsistent but exceptionally high-value customers. Offer exclusive, personalized deals to retain these high-spending customers. Provide VIP programs, early access to new products, or premium services to enhance loyalty. Address inconsistent behavior by identifying specific needs or preferences through surveys or data insights. Ensure consistent communication through tailored emails or account managers.

### Cluster -2: Dormants
**Reactivate:**
These customers are largely inactive and disengaged. Launch reactivation campaigns with highly attractive offers, such as significant discounts or free trials. Send reminders of past purchases to rekindle interest and demonstrate relevance. Use email or SMS campaigns to highlight limited-time offers and create urgency. Consider a "win-back" survey to understand their inactivity and adjust strategies accordingly.

### Cluster -1: Lurkers
**Encourage Growth**:
These are moderate spenders with irregular buying patterns. Encourage these moderate spenders to engage more frequently by offering targeted discounts or personalized recommendations. Highlight upsell opportunities or bundling options that align with their preferences. Promote loyalty programs that reward incremental spending or frequency. Educate them on additional product benefits to expand their usage.

### Cluster 0: Regulars
**Maintain Loyalty**:
These customers exhibit balanced and consistent behavior. Focus on maintaining engagement through consistent loyalty rewards, like discounts or points systems. Use newsletters or content to keep them updated and interested in your brand. Personalize offers based on their purchase history to make them feel valued. Avoid overloading with promotions, and instead, focus on consistent quality and experience.

### Cluster 1: Sleepers
**Wake Up**:
These customers are dormant with low spending and engagement. Reignite interest with attention-grabbing campaigns, such as "We Miss You!" offers or time-sensitive discounts. Highlight new products or services they haven't explored yet. Make communication more engaging by using interactive or gamified content. Provide incentives for small actions, such as referrals or signing up for newsletters.

### Cluster 2: Occasionals
**Nurture Interest**:
These are infrequent buyers with predictable, low spending habits. Build trust and familiarity by offering small rewards for occasional purchases, such as "Buy 2, Get 1 Free." Send educational or value-driven content to keep them informed about your brand's benefits. Leverage reminders for seasonal offers or related products. Gradually increase engagement with promotions that align with their past behavior.

### Cluster 3: Elites
**Reward**:
These are the most loyal and highly active customers. Strengthen their loyalty with exclusive perks, like early access to sales, premium support, or surprise gifts. Highlight their status as a top buyer to make them feel appreciated and valued. Create referral programs where they can earn rewards for bringing in new customers. Keep their experience seamless by prioritizing customer service and proactive engagement.

# Conclusion
This project effectively segmented customers into meaningful clusters based on their spending habits, purchase frequency, and recency of engagement, enabling tailored marketing strategies for each group. Through clustering algorithms, I uncovered valuable insights into customer behavior, from high-value loyalists to inactive buyers in need of re-engagement. The personalized strategies developed for each segment ensure more focused and efficient marketing efforts, driving both customer satisfaction and long-term business growth. By placing data-driven insights at the heart of the approach, this segmentation provides a solid framework for building customer-centric and sustainable marketing initiatives.






