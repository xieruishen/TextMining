"""Analyze the tweets similarity for different hashtags from twitter
@author: Sherrie Shen
"""

import re
import string
import math
from twython import *
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


def get_tweet_wordslist(hashtag):
    """
    Read 1100 tweets referenced to a specific hashtag from twitter.

    Username, Reply-to, website links, punctuation, and whitespace are stripped away.
    Returns a list of words used in each tweet as a list. All words are
    converted to lower case.

    :param hashtag: [str]
    A hashtag string without #.
    :return: word_list [list]
    """
    api = Twython(app_key='J2kQncld9iudZl1siXvyrZH01',
                  app_secret='lDwGit8XkGio0p15OwjWXNaprhLWMI9vhKu7LGHKKwjLxL9KGk',
                  oauth_token='2403785147-wukDst1jsWvfrcqb1uS4n2oTloKqh5HeembM7NR',
                  oauth_token_secret='1XGGuTlxRw0Fa4V6J8vMwvs53GFtknuRBN4gSC0r4w06o')
    # api.GetUserTimeline(screen_name='textmiining')
    search = api.search(q='#%s' % hashtag,  # **supply whatever query you want here**
                        count=1100)

    tweets = search['statuses']

    words_list = []
    for tweet in tweets:
        text = ' '.join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", " ", tweet['text']).split()).lower().replace(
            hashtag, '')
        # print(text, '\n\n\n')
        words = text.split()
        stripables = string.whitespace + string.punctuation
        for word in words:
            word = word.strip(stripables)
            word = word.lower()
            words_list.append(word)
    return words_list


def histogram(word_list):
    """
    Return a word histogram of all tweets.

    :param word_list: [list]
    :return: tweet_his [dic]
    """

    tweet_his = {}
    for word in word_list:
        if word == '':
            continue
        tweet_his[word] = tweet_his.get(word, 0) + 1
    return tweet_his


def most_frequent_list(word_list):
    """ Return a sorted list of words based on frequency
    """
    tweet_his = histogram(word_list)
    ordered_by_frequency = sorted(tweet_his, key=tweet_his.get, reverse=True)
    return ordered_by_frequency


def get_top_n_words(word_list, n):
    """Take a list of words as input and return a list of the n most
    frequently-occurring words ordered from most- to least-frequent.

    Parameters
    ----------
    word_list: [str]
        A list of words. These are assumed to all be in lower case, with no
        punctuation.
    n: int
        The number of words to return.

    Returns
    ---------
    The n most frequently occurring words ordered from most to least.
    frequently to least frequently occurring
    """
    top_n_list = []
    tweet_his = histogram(word_list)
    ordered_by_frequency = most_frequent_list(word_list)
    for i in ordered_by_frequency[:n + 1]:  # extract the top 100 common ones
        word, frequency = i, tweet_his[i]
        top_n_list.append((word, frequency))
    return top_n_list


def print_top_n_words(word_list, n):
    """
    Print the most frequently appeared words from tweets of a certain hashtag and the corresponding frequencies.

    :param word_list:[list]
    :param n: [int] number of most frequent words
    """
    top_n_list = get_top_n_words(word_list, n)
    for words in top_n_list:
        word = words[0]
        freq = words[1]
        print(word + ':' + '\t' + '%g' % freq)


def find_word_frequency(hashtag, n):
    """
    Retrieve 1100 tweets of a specific hashtag and print the n most frequently appeard words from the tweets.
    :param hashtag: [str]
    :param n: [int]
    """
    word_list = get_tweet_wordslist(hashtag)
    print_top_n_words(word_list, n)


def get_cosine(histogram1, histogram2):
    """
    Return cosine similarity of two histograms. The key of the dictionary is a unique word and the value is its corresponding frequency.
    :param histogram1: [dictionary]
    :param histogram2: [dictionary]
    :return: [int] cosine similarity
    """
    intersection = set(histogram1.keys()) & set(histogram2.keys())
    numerator = sum([histogram1[x] * histogram2[x] for x in intersection])

    sum1 = sum([histogram1[x] ** 2 for x in histogram1.keys()])
    sum2 = sum([histogram2[x] ** 2 for x in histogram2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def find_cosine_similarity(hashtag1, hashtag2):
    """
    Calculate and print cosine similarity of tweets between two hashtags.
    :param hashtag1: [str]
    :param hashtag2: [str]
    """
    word_list1 = get_tweet_wordslist(hashtag1)
    histogram1 = histogram(word_list1)
    word_list2 = get_tweet_wordslist(hashtag2)
    histogram2 = histogram(word_list2)
    cosine = get_cosine(histogram1, histogram2)
    print('Cosine:', cosine)


def find_cosine_similarity_multiple_hashtags_nonduplicates(hashtags):
    """
    Calculate pairwise cosine similarity of multiple hashtags and store each nonduplicated and nonidentical pair in a dictionary.
    The dictionary is saved in hashtags_similarity_dictionary.txt
    :param hashtags:list of hashtags. Each hashtag is a str.
    :return:hashtags_similarity_dictionary.txt
    """
    allhashtags = []
    cosines = {}
    for hashtag in hashtags:
        allhashtags.append(hashtag)
    for i in range(0, len(allhashtags) - 1):
        for j in range(i + 1, len(allhashtags)):
            word_list1 = get_tweet_wordslist(allhashtags[i])
            histogram1 = histogram(word_list1)
            word_list2 = get_tweet_wordslist(allhashtags[j])
            histogram2 = histogram(word_list2)
            cosine = get_cosine(histogram1, histogram2)
            cosines[(allhashtags[i] + ' ' + allhashtags[j])] = cosine

    # save cosine similarity dictionary to txt
    with open("hashtags_similarity_dictionary.txt", "w") as output:
        for key, cosine in cosines.items():
            output.write(str(key) + ':' + str(cosine) + '\t' + '\n\n')


def find_cosine_similarity_multiple_hashtags(hashtags):
    """
    Calculate pairwise cosine similarity of multiple hashtags. Each pair of cosine similarity is stored in a 2D symmetric list.
    :param hashtags: List of hashtags. Each hashtag is a str.
    :return: A 2D list saved in twitter_similarity_list.txt
    """
    cosines_list = []
    allhashtags = []
    for hashtag in hashtags:
        allhashtags.append(hashtag)
    # print(allhashtags)
    for i in range(len(allhashtags)):
        cosine_sublist = []
        for j in range(len(allhashtags)):
            # print(i,j)
            word_list1 = get_tweet_wordslist(allhashtags[i])
            histogram1 = histogram(word_list1)
            word_list2 = get_tweet_wordslist(allhashtags[j])
            histogram2 = histogram(word_list2)
            cosine = get_cosine(histogram1, histogram2)
            cosine_sublist.append(cosine)
        cosines_list.append(cosine_sublist)
    # print(cosines_list)

    # save cosine similarity list for plotting to txt as a string
    with open("twitter_similarity_list.txt", "w") as output:
        output.write(str(cosines_list))
    return cosines_list


def plot_similarity(hashtags, similarity_array=None):
    """
    Plot Metric Multi-dimensional Scaling (MDS) to visualize the hashtags in a two dimensional space.
    :param hashtags: List of hashtags. Each hashtag is a str.
    :param similarity_array: A 2D symmetric array.
    :return: A matplotlib plot.
    """
    if similarity_array is None:  # Plot from tweets of hashtags retrieved live.
        S = np.asarray(find_cosine_similarity_multiple_hashtags(hashtags))
        # print(S)
    else:
        S = np.asarray(similarity_array)  # Plot from saved 2D list in twitter_similarity_list.txt
    for i in range(len(S)):
        for j in range(len(S[i])):
            S[i][j] = round(S[i][j], 1)

    # dissimilarity is 1 minus similarity
    dissimilarities = 1 - S

    # compute the embedding
    coord = MDS(dissimilarity='precomputed').fit_transform(dissimilarities)

    plt.scatter(coord[:, 0], coord[:, 1])

    # Label the points

    for i in range(coord.shape[0]):
        plt.annotate('#%s' % hashtags[i], (coord[i, :]), horizontalalignment='left', verticalalignment='top')

    plt.show()


if __name__ == "__main__":
    # print("The top 100 frequent words from this hashtag are:")
    # find_word_frequency('guncontrolnow',100)

    """Find cosine similarity for hashtags of political topics. """
    hashtags_politics = ['guncontrol', 'blacklivesmatter', 'election2016', 'MakeAmericaGreatAgain', 'NotMyPresident',
                         '2ndamendment', 'Resist']

    # Uncomment this line to directly retrieve tweets of hashtags from the list above and compute pairwise cosine simlarity.
    # Result will be stored the cosine similiarty in hashtags_similarity_dictionary.txt
    # find_cosine_similarity_multiple_hashtags_nonduplicates(hashtags_politics)

    similarity_array_politics = [
        [0.9999999999999999, 0.5158714831476968, 0.6044504315037563, 0.412853524343512, 0.13818139599483806,
         0.7979342944537227, 0.3059259892628961],
        [0.5155901048826692, 1.0000000000000002, 0.3935641481950805, 0.29170691938908283, 0.09194999008867254,
         0.4201745382218893, 0.2030644520009313],
        [0.6038406376999854, 0.3935641481950805, 1.0, 0.31267897728116145, 0.17025314090155616, 0.5503275674306067,
         0.29747162832071106],
        [0.41137440110773976, 0.29170691938908283, 0.31267897728116145, 0.9999999999999999, 0.1000300856885497,
         0.3989080004405842, 0.20289327843517183],
        [0.13761354734473275, 0.09194999008867254, 0.17025314090155616, 0.1000300856885497, 1.0, 0.12337163859686895,
         0.7819835392915694],
        [0.7924492583595167, 0.42158381564411906, 0.5500385935801732, 0.3991138330729612, 0.12337163859686895,
         0.9999999999999999, 0.2847490341051634],
        [0.31413677188270095, 0.2086210545382909, 0.29795467908105, 0.19913863903421813, 0.784297199388706,
         0.27980159994181103, 1.0]]
    plot_similarity(hashtags_politics,
                    similarity_array_politics)
    # To plot tweets similarity live or add more hashtags to hashtags_politics, remove similarity_array_politics from the argument and add your hashtag of choice to hashtags_politics list.)

    """Find cosine similarity for hashtags of news media."""
    hashtags_news = ['cnn', 'bloomberg', 'nytimes', 'foxnews', 'NBC', 'bbc', 'wallstreetjournal']

    # Uncomment this line to directly retrieve tweets of hashtags from the list above and compute pairwise cosine simlarity.
    # Result is stored in hashtags_similarity_dictionary.txt
    # find_cosine_similarity_multiple_hashtags_nonduplicates(hashtags_news)

    similarity_array_news = [
        [1.0, 0.34526056969840463, 0.39547866992477265, 0.42599251903981816, 0.23886373727037116, 0.2481222708251769,
         0.33946482860716765],
        [0.34526056969840463, 1.0000000000000002, 0.7640582448924537, 0.5678503404765541, 0.19930610010926644,
         0.12818332382065095, 0.22216341281884341],
        [0.38842529912113993, 0.7640582448924537, 0.9999999999999999, 0.6381350652753127, 0.2552314265554362,
         0.19393418457664685, 0.3623205733906708],
        [0.4104307024717553, 0.5678503404765541, 0.6381350652753127, 1.0, 0.23993812957885624, 0.2763724401973461,
         0.4239820942835812],
        [0.22823106776948243, 0.19930610010926644, 0.2552314265554362, 0.23918683229551801, 1.0, 0.27375702631486976,
         0.35481145770725386],
        [0.2344643316654752, 0.12806485029707512, 0.19245437874666452, 0.27246675282129884, 0.26750894567797695,
         0.9999999999999999, 0.5145650901823716],
        [0.3288589646653035, 0.22216341281884341, 0.3623205733906708, 0.41736935198140707, 0.35481145770725386,
         0.5100680134924432, 1.0]]
    plot_similarity(hashtags_news,
                    similarity_array_news)
    # To plot tweets similarity live or add more hashtags to hashtags_news, remove similarity_array_news from the argument and add your hashtag of choice to hashtags_news list.)
