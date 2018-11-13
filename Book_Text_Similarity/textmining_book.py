"""Analyze the word frequencies and cosine similarity of books downloaded from Project Gutenberg.
@author: Sherrie Shen
"""

import string
import math
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


def skip_gutenberg_header(book):
    """Reads from book until it finds the line that ends the header.

    book: open file object

    """
    for line in book:
        if line.startswith('***'):
            break


def get_word_list(file_name):
    """Read the specified project Gutenberg book.

    Header comments, punctuation, and whitespace are stripped away.
    Returns a list of the words used in the book as a list. All words are
    converted to lower case.
    """
    book = open('books/%s' % file_name)
    skip_gutenberg_header(book)
    words_list = []
    for line in book:
        line = line.replace('-', ' ')  # replace hyphens with spaces before splitting
        words = line.split()
        # print('words=', words)
        stripables = string.whitespace + string.punctuation
        for word in words:
            word = word.strip(stripables)
            word = word.lower()
            # print(word)
            words_list.append(word)
    # print(words_list[0:4])
    return words_list


def histogram(word_list):
    """Return a book histogram"""
    book_his = {}
    for word in word_list:
        if word == '':
            continue
        book_his[word] = book_his.get(word, 0) + 1
    return book_his


def most_frequent_list(word_list):
    """ Return a sorted list of words based on frequency
    """
    book_his = histogram(word_list)
    ordered_by_frequency = sorted(book_his, key=book_his.get, reverse=True)
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
    book_his = histogram(word_list)
    ordered_by_frequency = most_frequent_list(word_list)
    for i in ordered_by_frequency[:n + 1]:  # extract the top 100 common ones
        word, frequency = i, book_his[i]
        top_n_list.append((word, frequency))
    return top_n_list


def print_top_n_words(word_list, n):
    """
    Print the most frequently appeared words from book and the corresponding frequencies.

    :param word_list:[list]
    :param n: [int] number of most frequent words
    """
    top_n_list = get_top_n_words(word_list, n)
    for words in top_n_list:
        word = words[0]
        freq = words[1]
        print(word + ':' + '\t' + '%g' % freq)


def find_word_frequency(file_name, n):
    word_list = get_word_list(file_name)
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


def find_cosine_similarity(file_name1, file_name2):
    """
    Calculate and print cosine similarity of tweets between two files.
    :param file_name1: [str]
    :param file_name2: [str]
    """
    word_list1 = get_word_list(file_name1)
    histogram1 = histogram(word_list1)
    word_list2 = get_word_list(file_name2)
    histogram2 = histogram(word_list2)
    cosine = get_cosine(histogram1, histogram2)
    print('Cosine:', cosine)


def find_cosine_similarity_multiple_books_nonduplicates(*file_names):
    """
    Calculate pairwise cosine similarity of multiple books and store each nonduplicated and nonidentical pair in a dictionary.
    The dictionary is saved in books_similarity_dictionary.txt
    :param file_names: books [str]
    :return: books_similarity_dictionary.txt
    """
    books = []
    cosines = {}
    for file in file_names:
        books.append(file)
    for i in range(0, len(books) - 1):
        for j in range(i + 1, len(books)):
            word_list1 = get_word_list(books[i])
            histogram1 = histogram(word_list1)
            word_list2 = get_word_list(books[j])
            histogram2 = histogram(word_list2)
            cosine = get_cosine(histogram1, histogram2)
            cosines[(books[i] + ' ' + books[j])] = cosine

    # save cosine similarity dictionary to txt
    with open("books_similarity_dictionary.txt", "w") as output:
        for key, cosine in cosines.items():
            output.write(str(key) + ':' + str(cosine) + '\t' + '\n\n')


def find_cosine_similarity_multiple_books(*file_names):
    """
    Calculate pairwise cosine similarity of multiple books. Each pair of cosine similarity is stored in a 2D symmetric list.
    :param file_names: books [str]
    :return: A 2D symmetric list of all pairwise cosine similarity
    """
    cosines_list = []
    books = []
    for file in file_names:
        books.append(file)
    # print(books)
    for i in range(len(books)):
        cosine_sublist = []
        for j in range(len(books)):
            # print(i,j)
            word_list1 = get_word_list(books[i])
            histogram1 = histogram(word_list1)
            word_list2 = get_word_list(books[j])
            histogram2 = histogram(word_list2)
            cosine = get_cosine(histogram1, histogram2)
            cosine_sublist.append(cosine)
        cosines_list.append(cosine_sublist)
    # print(cosines_list)
    return cosines_list


def plot_similarity(*file_names):
    """
    Plot Metric Multi-dimensional Scaling (MDS) to visualize the books in a two dimensional space.
    :param file_names: Books [str]
    :param similarity_array: A 2D symmetric array.
    :return: A matplotlib plot.
    """

    S = np.asarray(find_cosine_similarity_multiple_books(*file_names))

    # dissimilarity is 1 minus similarity
    dissimilarities = 1 - S

    # compute the embedding
    coord = MDS(dissimilarity='precomputed').fit_transform(dissimilarities)

    plt.scatter(coord[:, 0], coord[:, 1])

    # Label the points

    for i in range(coord.shape[0]):
        plt.annotate('%s' % file_names[i].replace('.txt', ''), (coord[i, :]), horizontalalignment='left',
                     verticalalignment='top')

    plt.show()


if __name__ == "__main__":
    # print(string.punctuation)
    # print("The top 100 frequent words in pride and prejudice are:")
    # find_word_frequency('pride_prejudice.txt',100)

    # Uncomment this line to view nonduplicate and nonidentical pairwise cosine similarity of books stored in a dictionary..
    # Result is stored in books_similarity_dictionary.txt
    # find_cosine_similarity_multiple_books_nonduplicates('pride_prejudice_JA.txt', 'sense_sensibility_JA.txt', 'lady_susan_JA.txt', 'emma_JA.txt','A_Christmas_Carol_CD.txt', 'Hard_Times_CD.txt', 'Oliver_Twist_CD.txt', 'A_Tale_of_Two_Cities_CD.txt', 'HuckFinn_MT.txt', 'Innocents_Abroad_MT.txt', 'Life_On_The_Mississippi_MT.txt', 'TomSawyer_MT.txt', 'gatsby_FSF.txt', 'Sherlock_ACD.txt')

    plot_similarity('pride_prejudice_JA.txt', 'sense_sensibility_JA.txt', 'lady_susan_JA.txt', 'emma_JA.txt',
                    'A_Christmas_Carol_CD.txt', 'Hard_Times_CD.txt', 'Oliver_Twist_CD.txt',
                    'A_Tale_of_Two_Cities_CD.txt', 'HuckFinn_MT.txt', 'Innocents_Abroad_MT.txt',
                    'Life_On_The_Mississippi_MT.txt', 'TomSawyer_MT.txt', 'gatsby_FSF.txt', 'Sherlock_ACD.txt')

    #To plot text similarity of book of your choice, add txt file of book to books folder and file name to the arguments above.
