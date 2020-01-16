# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data=pd.read_csv('../input/amazon_alexa.tsv',delimiter='\t',quoting=3)
data.head()
data.columns
data['verified_reviews'].head(50)
import re
def clean(texts):
    texts=re.sub(r'[^\w\s]','',texts)
    texts=re.sub('\s+',' ',texts)
    texts=texts.lower()
    return texts
data['cleaned_text']=data.verified_reviews.apply(lambda x:clean(x))
data['cleaned_text'].head(50)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sc=SentimentIntensityAnalyzer()
data['sentiment_polarity']=data.cleaned_text.apply(lambda x:sc.polarity_scores(x)['compound'])
data['sentiment_polarity'].head()
data['sentiment_polarity negative']=data.cleaned_text.apply(lambda x:sc.polarity_scores(x)['neg'])
data['sentiment_polarity positive']=data.cleaned_text.apply(lambda x:sc.polarity_scores(x)['pos'])
data['sentiment_polarity neutral']=data.cleaned_text.apply(lambda x:sc.polarity_scores(x)['neu']) 
data['sentiment_type']=''
data.loc[data.sentiment_polarity>0,'sentiment_type']='Positive Sentiment'

data.loc[data.sentiment_polarity==0,'sentiment_type']='Neutral Sentiment'
data.loc[data.sentiment_polarity<0,'sentiment_type']='Negative Sentiment'
data['sentiment_type'].head()
import seaborn as sns
sns.countplot('sentiment_type',data=data)
import matplotlib.pyplot as plt
fig1,ax1=plt.subplots(1,1)
ax1.pie(data.sentiment_type.value_counts(),autopct='%1.1f%%',labels=['Positive','Neutral','Negative'])
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
reviews=[]
len(data['cleaned_text'])
stopword=set(stopwords.words('english'))
for i in range(0,len(data['verified_reviews'])):
    review=re.sub('[^a-zA-Z]','',data['verified_reviews'][i])
    #review=data['cleaned_text'][i]
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopword]
    review=' '.join(review)
    reviews.append(review)
reviews[2]           
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(reviews).toarray()
x.shape
y=data['feedback']
y.shape
from nltk.tokenize import word_tokenize
totalwords=[]
cleaned_reviews=''
for i in range(1,len(data['cleaned_text'])):
    review=data['cleaned_text'][i]
    cleaned_reviews=cleaned_reviews+review
    tokens=word_tokenize(review)
    for word in tokens:
        if word not in stopword:
            totalwords.append(word)
wordfreq=nltk.FreqDist(totalwords)
common_words=wordfreq.most_common(50)
wc=WordCloud().generate(cleaned_reviews)
plt.imshow(wc)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)
train_x.shape
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(train_x,train_y)
pred=dt.predict(test_x)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(pred,test_y)
accuracy_score(pred,test_y)
#Accuracy of 92%
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(train_x,train_y)
pred1=rf.predict(test_x)
confusion_matrix(pred1,test_y)
accuracy_score(pred1,test_y)


        
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.