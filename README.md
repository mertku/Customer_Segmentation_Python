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

