#!/usr/bin/env python
# coding: utf-8

# In[1]:


#########################################################################################################################
#
#
##################################          Importing Dataset and Libraries               ##############################
#
#
#########################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from email.utils import parsedate_tz
import plotly
import plotly.express as px
import cufflinks as cf
import re
from nltk.corpus import stopwords
import string

pd.set_option('display.max_colwidth', 400)
df=pd.read_csv("2_and_half_months.csv")
print("Total # of tweets",df.shape[0])


# In[2]:


#########################################################################################################################
#
#
##################################          Data Preprocessing and Cleaning               ##############################
#
#
#########################################################################################################################

df.drop(['hashtags','user_description','user_created_at','user_followers_count','media','urls','user_default_profile_image','user_statuses_count','source','favorite_count','in_reply_to_user_id','retweet_count','retweet_id','user_favourites_count','in_reply_to_screen_name','in_reply_to_status_id','possibly_sensitive','retweet_screen_name','tweet_url','user_friends_count','user_listed_count','user_name','user_time_zone','user_verified','user_screen_name.1','user_urls'],axis=1,inplace=True)

def cleanTweets(text):
    text = re.sub('@[A-Za-z0-9]+', '', str(text))  #removed @mentions
    text = re.sub('#', '',str(text) ) #removing the hash symbols
    text = re.sub('RT', '', str(text)) #removing RT
    text = re.sub('https?:\/\/\S+', '', str(text)) #remove the hyper links
    text = re.sub('\d+', '', str(text)) #remove the hyper links
    return text

df['text'] = df['text'].apply(cleanTweets)

additional  = ['rt','rts','retweet'] #additional stopwords 

swords = set().union(stopwords.words('english'),additional) #big list of stopwords + additional ones

df.drop_duplicates(subset='text',inplace=True) #Removing Duplicate Tweets

#Removing Null Coordiates and splitting them into Longitude And Latitude for Visualization

df = df[pd.notnull(df['coordinates'])]
df[['A','B']] = df['coordinates'].str.split(',',expand=True)

df.head()

df.shape

string.punctuation

def remove_pun(text):
    txt_nopunt ="".join([c for c in text if c not in string.punctuation])
    return txt_nopunt

df['text'] = df['text'].apply(remove_pun)
df.drop(df.index[(df["lang"] != "en")], axis = 0, inplace = True)
df = df.replace(r'\n\n',' ', regex=True) 


# In[3]:



df['processed_text'] = df['text'].str.lower()          .str.replace('(@[a-z0-9]+)\w+',' ')          .str.replace('(http\S+)', ' ')          .str.replace('([^0-9a-z \t])',' ')          .str.replace(' +',' ')          .apply(lambda x: [i for i in x.split() if not i in swords])


# In[4]:


#########################################################################################################################
#
#
##################################           Feature Extraction(Porter Stemmer)            ##############################
#
#
#########################################################################################################################

from nltk.stem import PorterStemmer
ps = PorterStemmer()
df['stemed'] = df['processed_text'].apply(lambda x: [ps.stem(i) for i in x if i != ''])
df.head(2)


# In[5]:


#########################################################################################################################
#
#
##################################        Sentiment Analysis using TextBlob and Vader      ##############################
#
#
#########################################################################################################################

from textblob import TextBlob

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df['Subjectivity']= df['text'].apply(getSubjectivity)
df['Polarity']= df['text'].apply(getPolarity)
#create a function to compute negative, neutral and positive analysis
def get_analysis(score):
    if score < 0:
        return 'Negative'
    if score == 0:
        return 'Neutral'
    else:
        return 'Positive'
df['Analysis'] = df['Polarity'].apply(get_analysis)

df.tail(20)

import nltk.sentiment.vader as vd
from nltk import download
download('vader_lexicon')


# In[8]:


#Calculating Sentiment Score

sia = vd.SentimentIntensityAnalyzer()

from nltk import download
download('punkt')
from nltk.tokenize import word_tokenize
df['sentiment_score'] = df['processed_text'].apply(lambda x: sum([ sia.polarity_scores(i)['compound'] for i in word_tokenize( ' '.join(x) )]) )
df[['processed_text','sentiment_score']].head(n=10)


# In[9]:


#########################################################################################################################
#
#
##################################            Text Visualization/ WordCloud                ##############################
#
#
#########################################################################################################################

from wordcloud import WordCloud 

all_words = ' '.join( twts for twts in df['text'] )
wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 120).generate(all_words)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()


# In[10]:


#########################################################################################################################
#
#
##################################Topic Modelling Using  LDA ( Latent Dirichlet Allocation)##############################
#
#
#########################################################################################################################

from sklearn.feature_extraction.text import CountVectorizer
# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(max_df=0.9, min_df=100, token_pattern='\w+|\$[\d\.]+|\S+')
# apply transformation
tf = vectorizer.fit_transform(df['text']) #.toarray()
# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()
tf.shape 


# In[11]:


from sklearn.decomposition import LatentDirichletAllocation
number_of_topics = 10
model = LatentDirichletAllocation(n_components=number_of_topics, random_state=45) # random state for reproducibility
# Fit data to model
model.fit(tf)


# In[12]:


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# In[13]:


no_top_words = 10
display_topics(model, tf_feature_names, no_top_words)


# In[14]:


#########################################################################################################################
#
#
##################################          Sentiment Analysis Visualization               ##############################
#
#
#########################################################################################################################

df['sentiment_score'].apply(lambda x: round(x,)).value_counts()


# In[15]:


sent_clasification = pd.cut(df['sentiment_score'],          [-3,-1.2, 0, 1.2 , 3],          right=True,          include_lowest=True,          labels=['strongly negative', 'negative', 'positive', 'strongly positive'])


# In[16]:


sent_clasification.value_counts()


# In[17]:


sent_clasification.value_counts().plot(kind='bar')


# In[18]:


#########################################################################################################################
#
#
##################################          Sentiment Polarity Visualization               ##############################
#
#
#########################################################################################################################
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
df['Polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='Polarity',
    linecolor='black',
    yTitle='Tweets',
    title='Sentiment Polarity Distribution')


# In[19]:


df=df.reset_index()
print(df.shape[0])


# In[21]:


#########################################################################################################################
#
#
##################################            GeoModelling using MAp representation        ##############################
#
#
#########################################################################################################################

import folium
map1 = folium.Map(
    location=[59.338315,18.089960],
    tiles='cartodbpositron',
    zoom_start=1,
)
df.apply(lambda row:folium.CircleMarker(location=[row["B"], row["A"]]).add_to(map1), axis=1)
map1


# In[22]:


#########################################################################################################################
#
#
##################################                  Covid tweets Representation            ##############################
#
#
#########################################################################################################################

df['hour'] = pd.DatetimeIndex(df['created_at']).hour
df['date'] = pd.DatetimeIndex(df['created_at']).date
df['date'] = pd.to_datetime(df['date']) - pd.to_timedelta(7, unit='d')
df['count'] = 1
data_filtered = df[['hour', 'date', 'count']]
data_filtered.head(5)


# In[23]:


df_tweets_daily = data_filtered.groupby(["date"]).sum().reset_index()
df_tweets_daily.tail(5)


# In[24]:


plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
df_tweets_daily["count"].plot.bar(color='#BC0AF3')
plt.xlabel('Days')
plt.ylabel('No# of Tweets')
df_tweets_daily["count"].plot()
plt.show()


# In[26]:


#########################################################################################################################
#
#
##################################             Actual Covid19 Cases Daily/Weekly           ##############################
#
#
#########################################################################################################################

dff=pd.read_csv("COVID19-DAILY.csv")


# In[27]:


#Summing up the Cases by daily Worldwide counts
#ALL countries's where summed up by daily counts
dfft = dff[['dateRep', 'cases', 'deaths']]
df_covid_daily = dfft.groupby(["dateRep"]).sum().reset_index()
dfft["dateRep"] =pd.to_datetime(dfft.dateRep)
dfft.sort_values(by='dateRep')
df_covid_daily.head(40)

df_covid_daily.tail(5)


# In[28]:


#########################################################################################################################
#
#
###############################   Visualization of Total confimed daily Cases worldwide    ##############################
#
#
#########################################################################################################################

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_covid_daily["dateRep"], y=df_covid_daily["cases"], name='Covid Confirms',
                         line = dict(color='red', width=2)))


# Edit the layout
fig.update_layout(title='Correlation Between Daily Tweets and Covid Cases',
                   xaxis_title='Month',
                   yaxis_title='Confirmed Reports')

fig.show()


# In[29]:


#########################################################################################################################
#
#
#################################     Visualization of frequenxy of daily Tweets           ##############################
#
#
#########################################################################################################################

import plotly.graph_objects as go
fig = go.Figure()


fig.add_trace(go.Scatter(x=df_tweets_daily["date"], y=df_tweets_daily["count"], name='Tweets',
                         line = dict(color='royalblue', width=2)))
# Edit the layout
fig.update_layout(title='Correlation Between Daily Tweets and Covid Cases',
                   xaxis_title='Month',
                   yaxis_title='Tweets')

fig.show()


# In[ ]:




