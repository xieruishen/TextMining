# TextMining - Text Similarity

This is the base repo for the text mining and analysis project for Software Design at Olin College. For this project, text similarity analysis is conducted on both books downloaded from Project Gutenberg and tweets of chosen hashtags retrieved from twitter.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
- Go to [Project Gutenberg Webpage](https://github.com/sd18spring/TextMining-xieruishen.git) to download the txt file book of choice.

- To retrieve tweets from twitter, install python-twitter package.

```
$ pip install twython 
```
- To plot text similarity, ```scikit-learn.scikit-learn``` needs to be installed.


## Running Text Similarity Analysis

### Twitter Hashtag Tweets Similarity Analysis
- To plot tweets similarity result for hashtags chosen, run the following code.

```
cd Twitter_Text_Similarity
python textminng_twitter.py
```

- To plot tweets similarity for hashtags not included in the example and retrieve tweets live, follow instructions in ```textmining_twitter.py.```

### Books Similarity Analysis
- To plot book similarity result for books chosen, run the following code.

```
cd Book_Text_Similarity
python textminng_book.py
```

- To plot tweets similarity for books not included in the example, add ```.txt``` file of your book of choice to ```~\Book_Text_Similarity\books``` and follow instructions in ```textmining_book.py.```

## Project Reflection
View project reflection deliverable from [here](https://github.com/sd18spring/TextMining-xieruishen/blob/master/Sofdes%20MP3%20Reflection.pdf).

## Acknowledgments

* [Cosine Similarity](https://stackoverflow.com/questions/14156625/fetching-tweets-with-hashtag-from-twitter-using-python)
* [Remove @user and link of a tweet using regular expression](https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression/8377440)
