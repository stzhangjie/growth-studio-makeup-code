#==============================================================================

# -*- coding: utf-8 -*-

"""
Created on Mon Dec 2 11:30:13 2016

@author: jie.i.zhang
"""

#==============================================================================
#==============================================================================
#==============================================================================



#==============================================================================
#packages

import os
from os import listdir
from os.path import isfile, join
    
import tweepy
import time
import datetime
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    

from collections import Counter
from six.moves import cPickle as pickle

#==============================================================================
#connect

root_fldr='C:/Users/jie.i.zhang/Desktop/tw'
mid_fldr=root_fldr+'/output'
if not os.path.exists(mid_fldr): os.makedirs(mid_fldr)
os.chdir(mid_fldr)
    
consumer_key='tXxOcUQBWZJdz4NmBY0ygQzqw'
consumer_secret='Q6A7yPI9nIFOjvXXHfkFmdyNCbeEjDDIQwX07svaFGJ6quWywW'
auth_key='150748731-OmC3fhplkd6xT6bQvxOf8CEOficiLakEwMw6eGme'
auth_secret='Pxs5b70MDvLZ9fvORtvLQAq0dfGnRt4Yd230WXHwYIg45'    

auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(auth_key, auth_secret)

api2 = tweepy.API(auth, wait_on_rate_limit=True, \
                wait_on_rate_limit_notify=True, \
                compression=True)
                
#==============================================================================
#
##================
##scrape list of followers
#
#ids2 = []
#for page in tweepy.Cursor(api2.followers_ids, \
#                        screen_name="bareMinerals").pages():
#    if (len(ids2))%10000==0:
#        print(len(ids2))
#        print(datetime.datetime.now())
#    ids2.extend(page)
#    time.sleep(60)
#
##================    
## check counts
#print(len(ids2))
##200530
#
##================    
## output
#
#out_ids2=pd.DataFrame({ \
#        'seq_no':np.arange(len(ids2)), \
#        'ids':ids2
#})
#
#out_ids2=out_ids2[[ \
#        'seq_no', \
#        'ids'
#        ]]
#
#follower_list_fdlr=mid_fldr+'/00_follower_list'
#if not os.path.exists(follower_list_fdlr): os.makedirs(follower_list_fdlr)
#os.chdir(follower_list_fdlr)
#    
#out_ids2.to_csv('followers_id_list.csv',index=False)
#
#ids=ids2.ids
#                
#==============================================================================
#follower_count + tweet_count
                
#================
#input (for continue) 

follower_list_fdlr=mid_fldr+'/00_follower_list'
os.chdir(follower_list_fdlr)

ids2=pd.read_csv('followers_id_list.csv')
ids=ids2.ids

influencer_candidate=len(ids)
#
##================
##prepare loop
#
#count_tbl=pd.DataFrame({ \
#        'seq_no':np.arange(len(ids)), \
#        'twitter_ids':ids, \
#        'screen_name':'', \
#        'follower_count':0, 
#        'tweet_count':0 
#})
#
#count_tbl=count_tbl[[ \
#        'seq_no', \
#        'twitter_ids', \
#        'screen_name', \
#        'follower_count',
#        'tweet_count'
#        ]]
#        
##================
##scrape follower_count + tweet_count
#
#candidate_list_fdlr=mid_fldr+'/01_followers'
#if not os.path.exists(candidate_list_fdlr): os.makedirs(candidate_list_fdlr)
#os.chdir(candidate_list_fdlr)
#        
#for j in range(0,(round(influencer_candidate*.1/500))):
#    for i in range((j*500),(500*j+500)):
#        if i%50==0: 
#             print(i,'-',(i+50),'/',(j*500),'-',(j*500+500),'/',datetime.datetime.now())
#        sid=count_tbl.at[count_tbl.index[i],'twitter_ids']
##        count_tbl.iloc[[i]]
#        user2 = api2.get_user(sid)
##        user2.id    
##        user2.screen_name    
##        user2.followers_count    
#        count_tbl.at[count_tbl.index[i],'screen_name']=user2.screen_name
#        count_tbl.at[count_tbl.index[i],'follower_count']=user2.followers_count
#        count_tbl.at[count_tbl.index[i],'tweet_count']=user2.statuses_count
#    count_tbl.iloc[500*j:(500*j+500)].to_csv( \
#                        'list_'+str(j*500)+'_'+str(j*500+500)+'.csv' \
#                        ,index=False)
#    time.sleep(120)
##    count_tbl.iloc[[i]]
#    
###process the tail
##for i in range(xxxx,len(ids)):
##    sid=count_tbl.at[count_tbl.index[i],'twitter_ids']
##    if i%100==0: print(i, '  ', datetime.datetime.now())
##    user2 = api2.get_user(sid)
##    count_tbl.at[count_tbl.index[i],'screen_name']=user2.screen_name
##    count_tbl.at[count_tbl.index[i],'follower_count']=user2.followers_count
##    count_tbl.at[count_tbl.index[i],'tweet_count']=user2.statuses_count
##    count_tbl.iloc[xxxx:len(ids)].to_csv('tail.csv',index=False)
#    

#==============================================================================
#influencer list (rank)

###================
###merge candidate data
#
#candidate_list_fdlr=mid_fldr+'/01_followers'
#os.chdir(candidate_list_fdlr)
#all_csvs = [f for f in listdir(candidate_list_fdlr) \
#                if isfile(join(candidate_list_fdlr, f))]
#
#all_followers_tbl=pd.DataFrame()
#for file in all_csvs:
#    all_followers_tbl=pd.concat([all_followers_tbl,pd.read_csv(file)])
#
##all_followers_tbl.columns
#all_followers_tbl.drop('Unnamed: 0',axis=1, inplace=True)
#all_followers_tbl.reset_index(level=0, drop=True, inplace=True)
#
##================
##sort by followers        
#
#all_followers_tbl=all_followers_tbl.sort_values( \
#                ['follower_count'],ascending=False)
#
#influencer_fldr=mid_fldr+'/02_influencer'
#if not os.path.exists(influencer_fldr): os.makedirs(influencer_fldr)
#os.chdir(influencer_fldr)
#all_followers_tbl.to_csv('all_followers_tbl_fllwr_rank.csv' \
#                        ,index=False)
#
##================
##sort by tweets
#
#all_followers_tbl=all_followers_tbl.sort_values( \
#                ['tweet_count'],ascending=False)
#
#influencer_fldr=mid_fldr+'/02_influencer'
#if not os.path.exists(influencer_fldr): os.makedirs(influencer_fldr)
#os.chdir(influencer_fldr)
#all_followers_tbl.to_csv('all_followers_tbl_tw_rank.csv' \
#                        ,index=False)

#==============================================================================

##function to scrape tweets
#
##screen_name='boxticker'
##prd=92
#
#def get_all_tweets(screen_name,prd):
#    
##    print('=============================')
#    print('screen_name: '+screen_name)
#    
#    now = datetime.datetime.now()
#    since = now + datetime.timedelta(days=-1*int(prd))
#    print('... scraping since '+since.strftime('%Y-%m-%d'))
#    
#    alltweets=[]
#    new_tweets=api2.user_timeline(screen_name = screen_name,count=200)
#    alltweets.extend(new_tweets)
#    oldest = alltweets[-1].id - 1
##    print('...... batchs from '+str(alltweets[-1].created_at))
#    
#    since_flag=0
#    while ( (len(new_tweets)>0) & (since_flag==0) ):
#        
#        new_tweets = api2.user_timeline( \
#                                  screen_name = screen_name, \
#                                  count=200, \
#                                  max_id=oldest)
#        alltweets.extend(new_tweets)
##        print('...... batchs from '+str(alltweets[-1].created_at))
#        
#        #update the id of the oldest tweet less one
#        oldest = alltweets[-1].id - 1
#        if alltweets[-1].created_at<since: since_flag=1
#
#
#    twt_count=len(alltweets)
#    tw_df=pd.DataFrame({ \
#            'seq_no':range(twt_count), \
#            'screen_name':'', \
#            'create_time':'', \
#            'tweet_txt':''
#            })
#    tw_df=tw_df[[ \
#            'seq_no', \
#            'screen_name', \
#            'create_time', \
#            'tweet_txt'
#            ]]
#    for i in range(twt_count):
#        tweet=alltweets[i]
#        if tweet.created_at>=since:
#            tw_df.at[tw_df.index[i],'screen_name']= \
#                    screen_name
#            tw_df.at[tw_df.index[i],'create_time']= \
#                    tweet.created_at.strftime('%Y-%m-%d')
#            tw_df.at[tw_df.index[i],'tweet_txt']= \
#                    tweet.text.encode("utf-8")
#    tw_df=tw_df.query('tweet_txt != ""')  
#    
#    print('... number of tweets: '+str(len(tw_df)))
#    print('... first tweet date: '+str(tw_df.at[tw_df.index[-1],'create_time']))
##    print('=============================')
#    return(tw_df)
#    
##================
##input (for continue) 
#    
#influencer_fldr=mid_fldr+'/02_influencer'
#if not os.path.exists(influencer_fldr): os.makedirs(influencer_fldr)
#os.chdir(influencer_fldr)
#all_followers_tbl=pd.read_csv('all_followers_tbl_fllwr_rank.csv') ####
##all_followers_tbl=pd.read_csv('all_followers_tbl_tw_rank.csv')
#    
##================
##scrape tweets
#    
#influencer_count=round(all_followers_tbl.shape[0]*.05)    
#eff=0  
#i=0
##twt_fdlr=mid_fldr+'/03_tweet_1yr_fllwr_cnt' ####
#twt_fdlr=mid_fldr+'/03_tweet_1yr_fllwr_rank' ####
#if not os.path.exists(twt_fdlr): os.makedirs(twt_fdlr)
#os.chdir(twt_fdlr)
#
##while i<influencer_count:
#while i<2:
#    
#    print('==========================================================')
#    print('process at: '+str(datetime.datetime.now()))
#    print('... seq: '+str(i)+' / '+ \
#            '... effective: '+str(eff)+' / '+ \
#            '... total: '+str(influencer_count))
#    
#    screen_name=all_followers_tbl.screen_name[ \
#                all_followers_tbl.screen_name.index[i]]
#    try: 
##        all_twt=get_all_tweets(screen_name,prd=366)
#        all_twt=get_all_tweets(screen_name,prd=92)
#        i+=1
#        eff+=1
#        all_twt.to_csv(str(eff)+'_'+screen_name+'.csv',index=False)
#    except:
#        print('wrong: '+str(i))
#        pass
#        i+=1

#==============================================================================
##merge candidate data

#src_fldr='/03_tweet_1yr_tw_cnt'

#def all_twt_func(src_fldr):
#    tweet_pool_fldr=mid_fldr+src_fldr
#    os.chdir(tweet_pool_fldr)
#    all_csvs = [f for f in listdir(tweet_pool_fldr) \
#                    if isfile(join(tweet_pool_fldr, f))]
#    all_tweet_tbl=pd.DataFrame()
#    for file in all_csvs:
#        print(file)
#        all_tweet_tbl=pd.concat([all_tweet_tbl,pd.read_csv(file)])
#    
#    #all_followers_tbl.columns
#    #all_tweet_tbl.drop('Unnamed: 0',axis=1, inplace=True)
#    all_tweet_tbl.reset_index(level=0, drop=True, inplace=True)
#    return(all_tweet_tbl)
#
#clean_fdlr=mid_fldr+'/04_clean'
#if not os.path.exists(clean_fdlr): os.makedirs(clean_fdlr)
#os.chdir(clean_fdlr)
#
#all_tweet_tbl=all_twt_func('/03_tweet_1yr_fllwr_cnt')
#all_tweet_tbl.to_csv('1yr_fllwr_cnt.csv',index=False)     ######
#
#all_tweet_tbl2=all_twt_func('/03_tweet_1yr_tw_cnt')
#all_tweet_tbl2.to_csv('1yr_tw_cnt.csv',index=False)     ######

#==============================================================================
##hashtag at

#================
#input (for continue) 

clean_fdlr=mid_fldr+'/04_clean'
if not os.path.exists(clean_fdlr): os.makedirs(clean_fdlr)
os.chdir(clean_fdlr)

all_tweet_tbl=pd.read_csv('1yr_fllwr_cnt.csv')     ######
all_tweet_tbl2=pd.read_csv('1yr_tw_cnt.csv')     ######

#all_tweet_tbl.drop('Unnamed: 0',axis=1, inplace=True)
#all_tweet_tbl.drop('Unnamed: 0.1',axis=1, inplace=True)
all_tweet_tbl.reset_index(level=0, drop=True, inplace=True)

#all_tweet_tbl2.drop('Unnamed: 0',axis=1, inplace=True)
#all_tweet_tbl2.drop('Unnamed: 0.1',axis=1, inplace=True)
all_tweet_tbl2.reset_index(level=0, drop=True, inplace=True)

#================
#functions
def emoticon_re_define():
    emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [:D] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )"""
     
    regex_str = [
        emoticons_str,
        r'<[^>]+>', # HTML tags
        r'(?:@[\w_]+)', # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', 
        # URLs
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        r'(?:[\w_]+)', # other words
        r'(?:\S)' # anything else
    ]
    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
    return(tokens_re,emoticon_re)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    tokens = [token if emoticon_re.search(token) else token.lower()  \
                    for token in tokens]
    return tokens
tokens_re,emoticon_re=emoticon_re_define()

#================
#functions

def hashtag_at(all_tweet):
#    hash_txt=''
    hash_lst=[]
#    at_txt=''
    at_lst=[]
    #print('processing: hasttag & at')
    for i in range(len(all_tweet)):
        if i%50000==0:
            print('Process: '+str(i)+' / '+ \
                str(len(all_tweet))+ ' = ' + \
                str(i/len(all_tweet)))
        twt=all_tweet[i]
        terms_hash = [ \
            term for term in preprocess(twt) \
                if ( term.startswith('#')  ) \
            ] 
        terms_at = [ \
            term for term in preprocess(twt) \
                if ( term.startswith('@')  ) \
            ]   
        if (terms_hash!=[]) or (terms_hash!=['#']):
            for th in terms_hash:
                hash_lst.append(th)
#                hash_txt = hash_txt+' '+th 
        if (terms_at!=[]) or (terms_at!=['@']):
            for ta in terms_at:
                at_lst.append(ta)
#                at_txt = at_txt+' '+ta 
#    return(hash_txt,hash_lst,at_txt,at_lst)
    return(hash_lst,at_lst)

hash_lst_fl,at_lst_fl=hashtag_at(all_tweet_tbl.tweet_txt)
hash_lst_tw,at_lst_tw=hashtag_at(all_tweet_tbl2.tweet_txt)

hash_txt_fl = " ".join(hash_lst_fl)
at_txt_fl = " ".join(at_lst_fl)
hash_txt_tw = " ".join(hash_lst_tw)
at_txt_tw = " ".join(at_lst_tw)

#=================
#output

clean_fldr=mid_fldr+'/04_clean/' 
if not os.path.exists(clean_fldr): os.makedirs(clean_fldr)
os.chdir(clean_fldr)    

with open('hash_lst_fl', 'wb') as f:
    pickle.dump(hash_lst_fl, f, pickle.HIGHEST_PROTOCOL)
    
with open('at_lst_fl', 'wb') as f:
    pickle.dump(at_lst_fl, f, pickle.HIGHEST_PROTOCOL)
    
with open('hash_lst_tw', 'wb') as f:
    pickle.dump(hash_lst_tw, f, pickle.HIGHEST_PROTOCOL)
    
with open('at_lst_tw', 'wb') as f:
    pickle.dump(at_lst_tw, f, pickle.HIGHEST_PROTOCOL)
    
with open('hash_txt_fl', 'wb') as f:
    pickle.dump(hash_txt_fl, f, pickle.HIGHEST_PROTOCOL)
    
with open('at_txt_fl', 'wb') as f:
    pickle.dump(at_txt_fl, f, pickle.HIGHEST_PROTOCOL)
    
with open('hash_txt_tw', 'wb') as f:
    pickle.dump(hash_txt_tw, f, pickle.HIGHEST_PROTOCOL)
    
with open('at_txt_tw', 'wb') as f:
    pickle.dump(at_txt_tw, f, pickle.HIGHEST_PROTOCOL)

#==============================================================================
#term frequency

##================
##input (for continue) 

out='C:/Users/jie.i.zhang/Desktop/tw/mid01/04_clean/' 
if not os.path.exists(out): os.makedirs(out)
os.chdir(out)

with open('hash_lst_fl', 'rb') as f:
    hash_lst_fl = pickle.load(f)

with open('at_lst_fl', 'rb') as f:
    at_lst_fl = pickle.load(f)

with open('hash_lst_tw', 'rb') as f:
    hash_lst_tw = pickle.load(f)

with open('at_lst_tw', 'rb') as f:
    at_lst_tw = pickle.load(f)

with open('at_txt_fl', 'rb') as f:
    at_txt_fl = pickle.load(f)

with open('hash_txt_tw', 'rb') as f:
    hash_txt_tw = pickle.load(f)

with open('at_txt_tw', 'rb') as f:
    at_txt_tw = pickle.load(f)

##================


def word_freq(word_collect):
    counter = pd.DataFrame.from_dict( \
        Counter(word_collect), \
        orient='index').reset_index()
    counter.columns=['word', 'frequency']
    counter=counter.sort_values(['frequency'],ascending=False)
    counter.query('word!= "#"',inplace=True)  
    counter.query('word!= "@"',inplace=True)  
    counter.reset_index(level=0, drop=True, inplace=True)
    return(counter)

#==============
freq_fldr=mid_fldr+'/05_frequency/' 
if not os.path.exists(freq_fldr): os.makedirs(freq_fldr)
os.chdir(freq_fldr)    

terms_hash_all_count=word_freq(hash_lst_fl)
terms_hash_all_count.to_csv('1yr_hash_freq_fllwr_rank.csv')
terms_at_all_count=word_freq(at_lst_fl)
terms_at_all_count.to_csv('1yr_at_freq_fllwr_rank.csv')

#==============
freq_fldr=mid_fldr+'/05_frequency/' 
if not os.path.exists(freq_fldr): os.makedirs(freq_fldr)
os.chdir(freq_fldr)    

terms_hash_all_count2=word_freq(hash_lst_tw)
terms_hash_all_count2.to_csv('1yr_hash_freq_tw_rank.csv')
terms_at_all_count2=word_freq(at_lst_tw)
terms_at_all_count2.to_csv('1yr_at_freq_tw_rank.csv')

#==============================================================================
#bar_frequency

def word_freq_plot(counter,file_name):    
    counter_top=counter.iloc[0:5]
    indexes = np.arange(counter_top.shape[0])
    width = 0.7
    plt.figure(figsize=(7, 5))
    plt.bar(indexes, counter_top.frequency, width)
    plt.xticks(indexes + width * 0.5, \
                counter_top.word, \
                rotation='vertical', fontsize=15)
#    plt.savefig(file_name+'.png',dpi=100)
    plt.savefig(file_name+'.png',dpi=70)
#    plt.show()

#==============
freq_fldr=mid_fldr+'/05_frequency/' 
if not os.path.exists(freq_fldr): os.makedirs(freq_fldr)
os.chdir(freq_fldr)    

word_freq_plot(terms_hash_all_count,'1yr_hash_freq_fl_rank')
word_freq_plot(terms_at_all_count,'1yr_at_freq_fl_rank')

#==============
freq_fldr=mid_fldr+'/05_frequency/' 
if not os.path.exists(freq_fldr): os.makedirs(freq_fldr)
os.chdir(freq_fldr)    

word_freq_plot(terms_hash_all_count2,'1yr_hash_freq_tw_rank')
word_freq_plot(terms_at_all_count2,'1yr_at_freq_tw_rank')

#==============================================================================

from wordcloud import WordCloud

import os
word_cloud_fldr=mid_fldr+'/06_word_cloud')
os.chdir(word_cloud_fldr)

# Generate a word cloud imag
def wc_plot(terms_txt,file_name):

        wordcloud = WordCloud().generate(terms_txt)
        plt.figure(figsize=(30, 12))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(file_name+'.png',dpi=400)
#        image  =  wordcloud.to_image()
#        image.show()



with open('at_txt_fl', 'rb') as f:
    txt = pickle.load(f)
wc_plot(txt,'at_txt_fl')


with open('at_txt_tw', 'rb') as f:
    txt = pickle.load(f)
wc_plot(txt,'at_txt_tw')


with open('hash_txt_fl', 'rb') as f:
    txt = pickle.load(f)
wc_plot(txt,'hash_txt_fl')


with open('hash_txt_tw', 'rb') as f:
    txt = pickle.load(f)
wc_plot(txt,'hash_txt_tw')