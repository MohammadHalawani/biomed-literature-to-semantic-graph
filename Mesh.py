try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os
import os.path
import shutil
import sys
import time
import pickle
import multiprocessing
import string
import re
import csv
import unicodecsv
import pandas as pd
import numpy as np
import sqlite3
import difflib
import spell as sp
import tagLemmatize
import word2vec as w2v
import requests
try:
    import urllib
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
except ImportError:
    import urllib2
from ssl import CertificateError
import itertools
from itertools import repeat
from collections import Iterable, Counter, defaultdict
from operator import itemgetter
from Bio import Entrez
from habanero import Crossref, cn
import subprocess as sbp
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
# from polyglot.detect import Detector
from langdetect import detect, DetectorFactory, lang_detect_exception
import langdetect
import wordninja
from abbreviations import schwartz_hearst
from acora import AcoraBuilder
from flashtext import KeywordProcessor
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as wn_lemmatizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist
import nltk
import networkx as nx
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

if sys.version_info[0] == 2:  # Not named on 2.6
    newlinearg = {}
else:
    newlinearg = {'newline': ''}

numberOfProcesses = multiprocessing.cpu_count()
#numberOfProcesses = 45
Entrez.email = 'm.k.h.halawani2@newcastle.ac.uk'
cr = Crossref(mailto='m.k.h.halawani2@newcastle.ac.uk')

# click-through-token is obtained from ORCID and is saved as a dictionary pickle. It is passed as a header with the url requests to acquire access to full text articles that requires subscription or existence of this token for data mining purposes.
try:
    with open('pubmedAPI.txt', 'r') as f:
        Entrez.api_key = f.read().strip()
except FileNotFoundError:
    print('There is no PubMed API key')


def getPMIDs(query, fileName):
    try:
        return csvTolist(fileName)
    except FileNotFoundError:
        print("Didn't find ids file")
        term = query+' english[lang]'

        queryHandle = Entrez.egquery(term=term)
        queryResult = Entrez.read(queryHandle)
        queryHandle.close()
        numOfPubmedArticles = 0
        numOfPMCArticles = 0
        for row in queryResult['eGQueryResult']:
            if row['DbName'] == 'pubmed':
                numOfPubmedArticles = int(row['Count'])

        searchHandle = Entrez.esearch(
            db='pubmed', term=term, retmax=numOfPubmedArticles, usehistory='y')
        searchResult = Entrez.read(searchHandle)
        searchHandle.close()
        ids = searchResult['IdList']
        webEnv = searchResult['WebEnv']
        queryKey = searchResult['QueryKey']
        assert numOfPubmedArticles == len(ids)

        listTocsvRows(ids, fileName)
        return ids


def getPMCIDs(query, fileName):
    try:
        return csvTolist(fileName)
    except FileNotFoundError:
        print("Didn't find ids file")
        term = query+' english[lang]'

        queryHandle = Entrez.egquery(term=term)
        queryResult = Entrez.read(queryHandle)
        queryHandle.close()
        numOfPMCArticles = 0
        for row in queryResult['eGQueryResult']:
            if row['DbName'] == 'pmc':
                numOfPMCArticles = int(row['Count'])

        searchHandle = Entrez.esearch(
            db='pmc', term=term, usehistory='y')
        searchResult = Entrez.read(searchHandle)
        searchHandle.close()
        ids = searchResult['IdList']
        webEnv = searchResult['WebEnv']
        queryKey = searchResult['QueryKey']
        assert numOfPMCArticles == len(ids)

        directory = os.path.dirname(os.path.abspath(fileName))
        if not os.path.exists(directory):
            os.makedirs(directory)

        fileName = fileName.replace('.csv', '')
        with open(fileName+'.csv', 'w', **newlinearg) as csvFile:
            idsFile = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
            idsFile.writerow(ids)

        return ids


def get_mesh(pmids, xmlFileName):
    # change pmids to a list
    if (type(pmids) is not list):
        pmids = [pmids]
    # call PubMed API
    print('fetching..')
    handle = open(xmlFileName)
    print('reading..')
    xml_data = Entrez.read(handle)[u'PubmedArticle']
    for article in xml_data:
        pmid = article[u'MedlineCitation'][u'PMID'].title()
        # skip articles without MeSH terms
        if u'MeshHeadingList' in article[u'MedlineCitation']:
            for mesh in article[u'MedlineCitation'][u'MeshHeadingList']:
                # grab the qualifier major/minor flag, if any
                major = 'N'
                qualifiers = mesh[u'QualifierName']
                if qualifiers:
                    major = qualifiers[0].title()
                # grab descriptor name
                descr = mesh[u'DescriptorName']
                name = descr.title()

#                print(pmid, name, major)
                yield(pmid, name, major)


def etree_parser_get_mesh(handle):
    clear = True
    parser = ET.iterparse(handle, events=('start', 'end'))
    for event, elem in parser:
        if event == 'start' and elem.tag == 'MedlineCitation':
            clear = False
        if event == 'end' and elem.tag == 'MedlineCitation':
            clear = True
            pmid = elem.find('PMID').text
            meshs = elem.findall('MeshHeadingList/MeshHeading')
            for mesh in meshs:
                term = mesh.find('DescriptorName').text
                qualifier = mesh.find('QualifierName')
                if qualifier is None:
                    qualifier = ''
                else:
                    qualifier = qualifier.text
                yield(pmid, term, qualifier)
        if clear:
            elem.clear()  # discard the element


def writingFile(xmlFile, csvFile):
    csvFile = csvFile.replace('.csv', '')
    print('writing')
    # example output
    directory = os.path.dirname(os.path.abspath(csvFile))
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with open(csvFile+'.csv', 'a', **newlinearg) as out:
            csv_out = csv.writer(out)
#            print(('PMID', 'Term', 'Qualifier'))
            for pmid, term, qualifier in etree_parser_get_mesh(xmlFile):
                row = (pmid, term, qualifier)
                csv_out.writerow(row)
                print(row)
    except FileNotFoundError:
        with open(csvFile+'.csv', 'w', **newlinearg) as out:
            csv_out = csv.writer(out)
            csv_out.writerow(('PMID', 'Term', 'Qualifier'))
            print(('PMID', 'Term', 'Qualifier'))
            for pmid, term, qualifier in etree_parser_get_mesh(xmlFile):
                row = (pmid, term, qualifier)
                csv_out.writerow(row)
                print(row)


def getLastId(fileName):
    fileName = fileName.replace('.csv', '')
    with open(fileName+'.csv', 'r') as f:
        csv_in = csv.reader(f)
        existingIds = list(csv_in)
#        print (existingIds)
        lastId = existingIds[-2][0]
        print('lastId = ', lastId)
        return lastId


def fixNewlineEndings(readFile, writeFile):
    readFile = readFile.replace('.csv', '')
    writeFile = writeFile.replace('.csv', '')
    directory = os.path.dirname(os.path.abspath(writeFile))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(readFile+'.csv', 'r') as inf, open(writingFile+'.csv', 'w', **newlinearg) as outf:
        inFile = csv.reader(inf)
        outFile = csv.writer(outf)
        count = -1
        for row in inFile:
            count += 1
            if count % 2 == 0:
                outFile.writerow(row)


def addToDict(mydict, k, v):
    d = defaultdict(set, mydict)
    d[k] |= {v}
    return d


def removeKey(d, keys):
    r = dict(d)
    try:
        for key in keys:
            try:
                del r[key]
            except KeyError:
                pass
    except TypeError:
        del r[keys]
    except KeyError:
        pass
    return r


def countTerms(csvFileName):
    csvFileName = csvFileName.replace('.csv', '')
    with open(csvFileName+'.csv', 'r') as f:
        next(f)  # skip the header
        file = csv.reader(f)
        cn = Counter(map(itemgetter(1), file))
    return cn


def checkHeaders(csvFileName, headers):
    csvFileName = csvFileName.replace('.csv', '')
    if headers is None and not os.path.isfile(csvFileName+'.csv'):
        raise FileExistsError(
            'We need the headers to crearte the file for the first time.')
    elif headers and not os.path.isfile(csvFileName+'.csv'):
        directory = os.path.dirname(os.path.abspath(csvFileName))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(csvFileName+'.csv', 'a', **newlinearg) as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def switch(dictionary, variable):
    return dictionary.get(variable, variable)


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    From: https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python/

    """
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def dictTocsv(mydict, csvFileName, headers=None):
    csvFileName = csvFileName.replace('.csv', '')
    checkHeaders(csvFileName, headers)
    unicodeDict = dict()
    directory = os.path.dirname(os.path.abspath(csvFileName))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(csvFileName+'.csv', 'a', **newlinearg) as f:
        writer = csv.writer(f)
        for key, value in mydict.items():
            try:
                if type(value) is set or type(value) is list:
                    for x in value:
                        writer.writerow([key, x])
                else:
                    writer.writerow([key, value])
            except (UnicodeEncodeError, UnicodeDecodeError):
                unicodeDict[key] = value
    with open(csvFileName+'.csv', 'ab') as f:
        writer = unicodecsv.writer(f, encoding='utf-8')
#        writer = csv.writer(f, encoding='utf-8')
        for key, value in unicodeDict.items():
            if type(value) is set or type(value) is list:
                for x in value:
                    writer.writerow([key, x])
            else:
                writer.writerow([key, value])


def csvTolist(csvFileName):
    csvFileName = csvFileName.replace('.csv', '')
    cells = list()
    with open(csvFileName+'.csv', 'r') as inf:
        r = csv.reader(inf)
        for row in r:
            row = list(filter(None, row))
            if len(row) == 1:
                row = row[0]
            if row:
                cells.append(row)
        if len(cells) == 1:
            cells = cells[0]
    return cells


def csvTodict(csvFileName):
    csvFileName = csvFileName.replace('.csv', '')+'.csv'
    df = pd.read_csv(csvFileName)
    df = df.fillna('nan')
    mydict = dict(zip(list(df[list(df)[0]]), list(df[list(df)[1]])))
    # mydict = defaultdict(set)
    # with open(csvFileName+'.csv', 'r') as f:
    #     csvin = csv.reader(f)
    #     for row in csvin:
    #         k, v = row[0], row[1]
    #         mydict = addToDict(mydict, k, v)
    return mydict


def csvToDb(readFile):
    readFile = readFile.replace('.csv', '')
    with open(readFile+'.csv', 'r') as inf:
        csvin = csv.reader(inf)
        conn = sqlite3.connect('db.db')
        cur = conn.cursor()
        cur.execute('drop table if exists ' + readFile)
        cur.execute('create table ' + readFile + ' (id text, primary key(id))')
        conn.commit()
        count = 0
        for row in csvin:
            # assuming row returns a list-type object
            try:
                cur.execute('insert into ' + readFile + ' values(?)', row)
                count += 1
            except sqlite3.IntegrityError:
                pass
            if count == 50000:
                conn.commit()
                count = 0
        conn.commit()


def dbTocsv(writeFile):
    writeFile = writeFile.replace('.csv', '')
    directory = os.path.dirname(os.path.abspath(writeFile))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(writeFile+'NoDuplicates.csv', 'w', **newlinearg) as outf:
        csvout = csv.writer(outf, quoting=csv.QUOTE_ALL)
        conn = sqlite3.connect('db.db')
        cur = conn.cursor()
        cur.execute('select * from ' + writeFile)
        for row in cur:
            csvout.writerow(row)


def listTocsvCols(li, fileName):
    containsUnicode = False
    fileName = fileName.replace('.csv', '')
    directory = os.path.dirname(os.path.abspath(fileName))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName+'.csv', 'a', **newlinearg) as csvFile:
        liFile = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
        try:
            liFile.writerow(li)
        except (UnicodeEncodeError, UnicodeDecodeError):
            containsUnicode = True
    if containsUnicode:
        with open(fileName+'.csv', 'ab') as csvFile:
            liFile = unicodecsv.writer(
                csvFile, encoding='utf-8', quoting=csv.QUOTE_ALL)
#            liFile = csv.writer(csvFile, encoding='utf-8', quoting=csv.QUOTE_ALL)
            liFile.writerow(li)


def listTocsvRows(li, fileName):
    containsUnicode = False
    fileName = fileName.replace('.csv', '')
    directory = os.path.dirname(os.path.abspath(fileName))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName+'.csv', 'a', **newlinearg) as csvFile:
        liFile = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
        try:
            liFile.writerows([i] for i in li)
        except (UnicodeEncodeError, UnicodeDecodeError):
            containsUnicode = True
    if containsUnicode:
        with open(fileName+'.csv', 'ab') as csvFile:
            liFile = unicodecsv.writer(
                csvFile, encoding='utf-8', quoting=csv.QUOTE_ALL)
#            liFile = csv.writer(csvFile, encoding='utf-8', quoting=csv.QUOTE_ALL)
            liFile.writerows([i] for i in li)


def writecsvRow(fileName, row, headers=None):
    checkHeaders(fileName, headers)
    containsUnicode = False
    fileName = fileName.replace('.csv', '')
    directory = os.path.dirname(os.path.abspath(fileName))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName + '.csv', 'a', **newlinearg) as f:
        csvout = csv.writer(f)
        try:
            csvout.writerow(row)
        except (UnicodeEncodeError, UnicodeDecodeError):
            containsUnicode = True
    if containsUnicode:
        with open(fileName+'.csv', 'ab') as f:
            csvout = unicodecsv.writer(f, encoding='utf-8')
#          csvout = csv.writer(f, encoding='utf-8')
            csvout.writerow(row)


def proportionOfDict(li, di):
    resultDict = dict()
    for item in li:
        if item in di:
            resultDict[item] = di[item]
    return resultDict


def removeDuplicateChars(txt):
    resTxt = ''
    for word in txt.split():
        group = itertools.groupby(word)
        s = set()
        res = ''
        for ch, rep in group:
            s.add(len(list(rep)))
            res += ch
        if len(s) > 1:
            res = word
        resTxt += ' '+res
    return resTxt


def removeSpaces(x):
    remove = string.whitespace
    s = str(x)
    s = s.translate(str.maketrans(dict.fromkeys(remove)))
    return s


def removeDuplicateSpaces(x):
    return " ".join(x.split())


def removeChars(x, unwantedChars=[]):
    if not unwantedChars:
        remove = string.punctuation
        unwantedChars = {':', '-', '_', '!', '?', '.', ',', '"', "'"}
    s = str(x)
    for c in unwantedChars:
        s = s.replace(c, ' ')
    s = s.translate(str.maketrans(dict.fromkeys(remove)))
    return s


def removeWords(x, unwantedWords=[]):
    if not unwantedWords:
        determiners = ['a', 'an', 'the']
        conjuctions = ['for', 'and', 'nor',
                       'but', 'or', 'yet', 'so',
                       'as', 'that', 'till', 'until']
        prepostions = [
            'with', 'at', 'from', 'into', 'towards', 'upon', 'of', 'to', 'in', 'for', 'on', 'by', 'about', 'like', 'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within', 'along', 'but', 'up', 'out', 'around', 'down', 'off', 'above', 'near'
        ]
        unwantedWords = set(prepostions + conjuctions + determiners)
    s = str(x)
    for word in unwantedWords:
        s = re.sub('(^|\s+)'+word+'(\s+|$)', ' ', s, flags=re.I)
        s = s.strip()
    return s


def readTxt(fileName):
    with open(fileName, 'rb') as f:
        try:
            txt = f.read().decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            txt = f.read().decode('latin1')
        except AttributeError:
            txt = f.read()
    return txt


def writeTxt(txt, fileName):
    directory = os.path.dirname(os.path.abspath(fileName))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName, 'wb') as f:
        try:
            f.write(txt.encode('utf-8'))
        except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
            f.write(txt)


def appendToTxt(txt, fileName):
    directory = os.path.dirname(os.path.abspath(fileName))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName, 'ab') as f:
        try:
            f.write(txt.encode('utf-8'))
        except (UnicodeEncodeError, UnicodeDecodeError):
            f.write(txt)


def addTitleTag(term):
    term = term.replace(' ', ' [title] ') + ' [title]'
    term = term.replace('[title]  [title] ', '[title] ')
    return term


def addAuthorTag(term):
    term = term.replace(' ', ' [author] ') + ' [author]'
    term = term.replace('[author]  [author] ', '[author] ')
    return term


def prepare_pubmed_search_term_from_title(title):
    prepared_term = addTitleTag(removeDuplicateSpaces(
        removeWords(removeChars(title.lower()))))
    return prepared_term


def removeDuplicates(FileName):
    csvToDb(FileName)
    dbTocsv(FileName)


def getLinkedIds(ids, csvFile, header=True):
    csvFile = csvFile.replace('.csv', '')
    remainingIDs = list()
    batchSize = 5000
    last = len(ids)-1
    directory = os.path.dirname(os.path.abspath(csvFile))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(csvFile+'.csv', 'a', **newlinearg) as out:
        csv_out = csv.writer(out)
        if header:
            csv_out.writerow(('PMID', 'link', 'score'))
#           print(('PMID', 'link'))
        for start in range(0, len(ids), batchSize):
            end = min(last, start+batchSize)
            print("Downloading and writing record %i to %i" % (start, end))
            ids = list(ids)

            attempt = 0
            while attempt < 3:
                attempt += 1
                try:
                    handle = Entrez.elink(id=ids[start: end],
                                          term='english[lang]',
                                          linkname='pubmed_pubmed',
                                          cmd='neighbor_score')
                except HTTPError as e:
                    if 500 <= e.code <= 599:
                        print("Received error from server %s" % e)
                        print("Attempt %i of 3" % attempt)
                        time.sleep(15)
                    else:
                        raise

            records = Entrez.read(handle)
            for record in records:
                try:
                    for link in record["LinkSetDb"][0]["Link"]:
                        if not record['IdList'][0] == link['Id']:
                            row = (record['IdList'][0],
                                   link['Id'], link['Score'])
                            csv_out.writerow(row)
#                       print(row)
                except IndexError:
                    #                    print(record['IdList'][0], 'has',
                    #                          len(record["LinkSetDb"][0]["Link"])-1, 'links')
                    remainingIDs += [record['IdList'][0]]
#    print(remainingIDs)
    return remainingIDs


def writeUniqueLinks(readFile, writeFile):
    readFile = readFile.replace('.csv', '')
    writeFile = writeFile.replace('.csv', '')
    related = set()
    ids = set()
    directory = os.path.dirname(os.path.abspath(writeFile))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(readFile+'.csv', 'r') as inf:
        csvin = csv.reader(inf)
        for row in csvin:
            ids.add(row[0])
    with open(readFile+'.csv', 'r') as inf, open(writeFile+'.csv', 'w', **newlinearg) as outf:
        csvin = csv.reader(inf)
        csvout = csv.writer(outf, quoting=csv.QUOTE_ALL)
#        next(csvin)  #skip the header
        counter = 0
        for row in csvin:
            related.add(row[1])
            counter += 1
            if counter == 50000:
                related = related-ids
#                related = list(related)
#                related=sorted(related, reverse = True)
                for i in related:
                    csvout.writerow([i])
                related.clear()
                counter = 0
    removeDuplicates(writeFile)
#    return related


def strToint(data):
    x = [int(i) for i in data]
    return x


def normalise(data):
    minimum = min(data)
    maximum = max(data)
    normalised = [(i - minimum)/(maximum-minimum) for i in data]
    return normalised


def getExtension(filePathOrURL):
    ext = re.findall('.+\.([a-zA-Z]{3,10})"?$', filePathOrURL)
    if ext:
        return ext[-1]
    else:
        return None


def decideExtension(dictionaryOfcontentTypeDispositionURL):
    try:
        contentType = dictionaryOfcontentTypeDispositionURL['contentType']
    except KeyError:
        contentType = None
    try:
        crContentType = dictionaryOfcontentTypeDispositionURL['crContentType']
    except KeyError:
        crContentType = None
    try:
        contentDisposition = dictionaryOfcontentTypeDispositionURL['contentDisposition']
    except KeyError:
        contentDisposition = None
    try:
        url = dictionaryOfcontentTypeDispositionURL['url']
    except KeyError:
        url = None

    if contentType:
        ext = contentType.split('/')[-1].split(';')[0]
#        print('type')
    elif crContentType and crContentType != 'unspecified':
        ext = crContentType.split('/')[-1].split(';')[0]
#        print('cr')
    elif contentDisposition:
        ext = getExtension(contentDisposition)
#        print('dispo')
    elif url and getExtension(url):
        ext = getExtension(url)
#        print('url')
    else:
        ext = None
#        print('non')

    if 'api.elsevier' in url and (ext == 'plain' or ext == 'htm'or ext == 'html'):
        ext = 'txt'
    elif ext == 'plain' or ext == 'htm':
        ext = 'html'
    elif ext == 'msword':
        ext = 'doc'
    elif ext == 'octet-stream':
        ext = 'pdf'

    return ext


def prepareReplacements(replacements):
    if type(replacements) is str:
        replacements = csvTodict(replacements)

    # This does not work, maybe because we're chaqnging the dictionary while looping
    # replacements = {str(k):str(v) for k, v in replacements.items()}
    d = {str(k): str(v) for k, v in replacements.items() if k is not ''}
    replacements = d
    del d

    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
#    substrs = sorted(replacements, key=len, reverse=True)

    # The replacements values need to be added to the start of the dictionary to avoid replacing them
    substrs2 = sorted(replacements.values(), key=len, reverse=True)
#    substrs = substrs2 + substrs
    replacements2 = dict(zip(substrs2, substrs2))
    replacements2.update(replacements)
    replacements = replacements2

    # # create a csv file with the sorted replacements
    # dictTocsv(replacements, 'replacementsDict.csv',
    #           ['original', 'replacement'])

    return replacements


def multireplace(txt, replacements):
    replacements = prepareReplacements(replacements)

    if len(replacements) == 0:
        return txt

    inType = type(txt)
    if inType is list or type(txt) is set:
        txt = ' '.join(i for i in txt)

    builder = AcoraBuilder(list(replacements.keys()))
    ac = builder.build()
    matches = ac.findall(txt)

    newTxt = ''
    d = matches
    d = longest_match(matches)

    prevIdx = 0
    for idx, word in d.items():
        if idx >= prevIdx:
            newTxt += txt[prevIdx: idx] + txt[idx: idx +
                                              len(word)].replace(word, replacements[word])  # + txt[ :]
            prevIdx = idx+len(word.rstrip())
    newTxt += txt[prevIdx:]
    return newTxt.replace('t_ion', 'tion')


def longest_match(matches):
    matches = sorted(matches, key=lambda x: x[1])
    d = dict()
    prevK = 0
    prevHit = ''
    for k, g in itertools.groupby(matches, lambda x: x[1]):
        if k >= prevK+len(prevHit.rstrip()):
            hit = max([x[0] for x in g], key=len)
            d[k] = hit
            prevK = k
            prevHit = hit
    return d


def acoraMultireplace(txt, replacements):
    if type(replacements) is str:
        replacements = csvTodict(replacements)

    inType = type(txt)
    if inType is list or type(txt) is set:
        txt = ' '.join(i for i in txt)

    # This does not work, maybe because we're chaqnging the dictionary while looping
    # replacements = {str(k):str(v) for k, v in replacements.items()}

    d = {str(k): str(v) for k, v in replacements.items() if k is not ''}
    replacements = d
    del d

    if len(replacements) == 0:
        return txt

    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
#    substrs = sorted(replacements, key=len, reverse=True)

    # The replacements values need to be added to the start of the dictionary to avoid replacing them
    substrs2 = sorted(replacements.values(), key=len, reverse=True)
#    substrs = substrs2 + substrs
    replacements2 = dict(zip(substrs2, substrs2))
    replacements2.update(replacements)
    replacements = replacements2
    del replacements2, substrs2

    builder = AcoraBuilder(list(replacements.keys()))
    ac = builder.build()
    matches = ac.findall(txt)

    newTxt = ''
    d = longest_match(matches)
    prevIdx = 0
    for idx, word in d.items():
        if idx >= prevIdx:
            newTxt += txt[prevIdx: idx] + txt[idx: idx +
                                              len(word)].replace(word, replacements[word])  # + txt[ :]
            prevIdx = idx+len(word.rstrip())
    newTxt += txt[prevIdx:]
    return newTxt.replace('t_ion', 'tion')


def flashTextMultireplace(string, replacements):
    if type(replacements) is str:
        replacements = csvTodict(replacements)

    inType = type(string)
    if inType is list or type(string) is set:
        string = ' '.join(i for i in string)

    # This does not work, maybe because we're chaqnging the dictionary while looping
    # replacements = {str(k):str(v) for k, v in replacements.items()}

    d = {str(v): [str(k)] for k, v in replacements.items() if k is not ''}
    replacements = d
    del d

    if len(replacements) == 0:
        return string

    kp = KeywordProcessor()
    kp.add_keywords_from_dict(replacements)
    res = kp.replace_keywords(string)
    return res.replace('t_ion', 'tion')


def rreplace(s, old, new, occurrence):
    """
    From https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string?answertab=votes#tab-top
    """
    if old:
        li = s.rsplit(old, occurrence)
        return new.join(li)
    else:
        return s


def regexMultireplace(string, replacements):
    """
    From: https://gist.githubusercontent.com/bgusach/a967e0587d6e01e889fd1d776c5f3729/raw/a2ac838179d453a1b9a028cfee9a66ff0e0157dc/multireplace.py
    Given a string and a replacement map, it returns the replaced string.

    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str

    """
    if type(replacements) is str:
        replacements = csvTodict(replacements)

    inType = type(string)
    if inType is list or type(string) is set:
        string = ' '.join(i for i in string)

    # This does not work, maybe because we're chaqnging the dictionary while looping
    # replacements = {str(k):str(v) for k, v in replacements.items()}

    d = {str(k): str(v) for k, v in replacements.items() if k is not ''}
    replacements = d

    if len(replacements) == 0:
        return string

    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # The replacements values need to be added to the start of the dictionary to avoid replacing them
    substrs2 = sorted(replacements.values(), key=len, reverse=True)
    substrs = substrs2 + substrs
    replacements2 = dict(zip(substrs2, substrs2))
    replacements2.update(replacements)
    replacements = replacements2
    del replacements2, substrs2

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))
    del substrs

    # For each match, look up the new string in the replacements
    res = regexp.sub(lambda match: replacements[match.group(0)], string)
#    del replacements

    if inType is list or type(string) is set:
        res = res.split()
    return res.replace('t_ion', 'tion')


def extractDOI(doiLinkOrTag):
    return doiLinkOrTag.split('org/')[-1]


def doiTofileName(doi):
    notAllowedCharReplacements = {
        '/': '-',
        '\\': '-',
        '>': '}',
        '<': '{',
        ':': '_',
        '?': '!',
        '"': '\'',
        '*': '+',
        '|': '$'
    }
    return multireplace(doi, notAllowedCharReplacements)


def createFile(directory, doi, ext=None):
    if ext:
        fileName = os.path.join(directory,  ext, doiTofileName(doi)) + '.'+ext
    else:
        fileName = os.path.join(directory, 'other', doiTofileName(doi))
    directory = os.path.dirname(fileName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return fileName


# def parallellize(functionName, inputList, **kwargs):
#     if len(inputList) == 1:
#         out = functionName(inputList[0])
#     else:
#
#         # last = len(inputList)-1
#         # batchSize = max(1, int(len(inputList)/numberOfProcesses))
#         # sidx = [i for i in range(0, len(inputList), batchSize)]
#         # eidx = [min(last, start+batchSize) for start in sidx]

#         # ps = [inputList[st:en] for st, en in zip(sidx, eidx)]

#         # if kwargs:
#         #     kwargs = next(iter(kwargs.values()))
#         #     argsAndkwargs = [(sublist, kwargs) for sublist in ps]
#         #     p.starmap(functionName, argsAndkwargs))

#         out = list()
#         with multiprocessing.Pool(numberOfProcesses) as p:
#             if kwargs:
#                 print(kwargs)
#                 kwargs = next(iter(kwargs.values()))
#                 argsAndkwargs = [(elm, kwargs) for elm in inputList]
#                 print(kwargs)
#                 out = p.starmap(functionName, argsAndkwargs, chunksize=batchSize)
#             else:
#                 out = p.map(functionName, inputList, chunksize=batchSize)
#         return out


def parallellize(functionName, inputList, **kwargs):
    if len(inputList) == 1:
        out = functionName(inputList[0])
    else:

        batchSize = int(len(inputList)/numberOfProcesses)
        batchSize = max(batchSize, 1)
        sidx = [i for i in range(0, len(inputList), batchSize)]
        eidx = [min(len(inputList), start+batchSize) for start in sidx]

        ps = [inputList[st:en] for st, en in zip(sidx, eidx)]
        out = list()
        with multiprocessing.Pool(numberOfProcesses) as p:
            out = p.map(functionName, ps)
        return out


def parallellizeWithoutSubListing(functionName, inputList, **kwargs):
    if len(inputList) == 1:
        out = functionName(inputList[0])
    else:

        batchsize = max(1, int(len(inputList)/numberOfProcesses))
        with multiprocessing.Pool() as p:
            out = p.map(functionName, inputList,  chunksize=batchsize)
        return out


def makeParallelargs(filesOrDirectoryOfFiles, *directoriesOfOutputsArgs):
    listOftuplesArgs = list()
    if os.path.isdir(filesOrDirectoryOfFiles):
        inputFiles = os.listdir(filesOrDirectoryOfFiles)
        inputFiles = [os.path.join(filesOrDirectoryOfFiles, f)
                      for f in inputFiles]
    elif os.path.isfile(filesOrDirectoryOfFiles):
        inputFiles = [filesOrDirectoryOfFiles]
    elif filesOrDirectoryOfFiles is list:
        inputFiles = filesOrDirectoryOfFiles
    for inputFile in inputFiles:
        inputsDirectory = os.path.dirname(os.path.abspath(inputFile))
        outputsDirectories = list()
        for directoryOfOutputs in directoriesOfOutputsArgs:
            if len(directoryOfOutputs) == 0:
                outputsDirectory = inputsDirectory
            else:
                outputsDirectory = os.path.join(
                    inputsDirectory, directoryOfOutputs)
            outputsDirectories.append(outputsDirectory)
        listOftuplesArgs.append((inputFile, outputsDirectories))
    return listOftuplesArgs


def copyFiles(srcFiles, destFileOrDir):
    if os.path.isfile(destFileOrDir):
        directory = os.path.dirname(os.path.abspath(destFileOrDir))
    else:
        directory = os.path.abspath(destFileOrDir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for f in srcFiles:
        if os.path.isfile(f):
            shutil.copy2(f, destFileOrDir)


def fixDivisionLines(directory):
    wantedDivistionLine = '\n___________________________________________________________________________________________________________________________________________\n'
    processedDivisionLine = '\n __________________________________________________________\n'

    reps = {processedDivisionLine: wantedDivistionLine}

    parallellyApplyFunctionOnTxt(
        multireplace, directory, kwargs={'replacements': reps})


def fixDivisionLines2(directory):
    wantedDivistionLine = '\n___________________________________________________________________________________________________________________________________________\n'
    processedDivisionLine = ' __________________________________________________________ '

    reps = {processedDivisionLine: wantedDivistionLine}

    parallellyApplyFunctionOnTxt(
        multireplace, directory, kwargs={'replacements': reps})


def divideFile(f, avgSize):
    directory = os.path.dirname(f)
    name, ext = os.path.splitext(os.path.basename(f))
    divisionLine = '\n___________________________________________________________________________________________________________________________________________\n'
    with open(f) as fin:
        count = 1
        tempBuf = ''
        while True:
            buf = tempBuf + fin.read(avgSize)
            if not buf:
                break
            tempBuf = buf.split(divisionLine)[-1]
            buf = buf.replace(tempBuf, '')
            if not buf:
                buf = tempBuf
                tempBuf = ''
            outFileName = name + str(count) + ext
#            print(outFileName)
            with open(os.path.join(directory, outFileName), 'w') as fout:
                fout.writelines(buf)
            count += 1
    os.remove(f)


def divideDirectory(directory, largestSize):
    dirSize = sum(os.path.getsize(os.path.join(directory, f)) for f in os.listdir(
        directory) if os.path.isfile(os.path.join(directory, f)))

    avgSize = int(dirSize/numberOfProcesses)
    avgSize = 5*pow(2, 20)
    if dirSize > largestSize:
        for f in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, f)) and os.path.getsize(os.path.join(directory, f)) > avgSize:
                divideFile(os.path.abspath(
                    os.path.join(directory, f)), avgSize)


def joinFiles(listOfInputFilesPaths, outFileName):
    outExt = os.path.splitext(os.path.basename(outFileName))[1]
    if os.path.isfile(outFileName):
        raise FileExistsError(outFileName+': This file already exists')
    directory = os.path.dirname(os.path.abspath(outFileName))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outFileName, 'w') as fout:
        for f in listOfInputFilesPaths:
            directory = os.path.dirname(f)
            ext = os.path.splitext(os.path.basename(f))[1]
            if ext == outExt:
                with open(f) as fin:
                    try:
                        fout.write(fin.read())
                    except MemoryError:
                        while True:
                            buf = fin.read(50*pow(10, 6))
                            if not buf:
                                break
                            fout.write(buf)
    [os.remove(f) for f in listOfInputFilesPaths]


def joinFilesByName(directory):
    files = os.listdir(directory)
    outFiles = set()
    for f in files:
        if os.path.isfile(os.path.join(directory, f)):
            name, ext = os.path.splitext(f)
            name = re.sub(r'[0-9]+'+ext, '', f)
            outFiles.add(name+ext)

    for outf in outFiles:
        outName, outExt = os.path.splitext(outf)
        inFiles = list()
        for f in files:
            name, ext = os.path.splitext(f)
            if name.startswith(outName) and ext == outExt:
                inFiles.append(os.path.join(directory, f))
        if len(inFiles) > 1:
            inFiles = sorted_nicely(inFiles)
            joinFiles(inFiles, os.path.join(directory, outName+outExt))


def getDomain(url):
    domain = re.findall('//(.+?)/', url)
    if domain:
        return domain[-1]
    else:
        return None


def fixSpringerLinkURL(url):
    doi = re.findall('pdf/(.+)', url)[-1]
    doi = doi.replace('/', '%2F')
    newURL = 'https://link.springer.com/content/pdf/' + doi
    return newURL


def getCambridgeURL(doi):
    doi = re.findall('10.1017/s(.+)', doi)[-1]
    newURL = 'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S' + doi
    return newURL


def download(doi, url, requestHeaders, directory, wantedContentTypes=[]):
    downloadableViaRequests = False
    try:
        r = requests.head(url, headers=requestHeaders, allow_redirects=True)
        try:
            remainingHits = int(r.headers['CR-TDM-Rate-Limit-Remaining'])
            resetTime = int(r.headers['CR-TDM-Rate-Limit-Reset'][: -3])
            if remainingHits < 2:
                sleepingTime = abs(resetTime - time.time())
#                print('sleeping')
                time.sleep(sleepingTime+1)
#                print('woke up')
        except KeyError:
            pass
        if r.status_code == requests.codes.ok:
            downloadableViaRequests = True
    except (requests.exceptions.ConnectionError,
            requests.exceptions.TooManyRedirects) as e:
        #        print(doi, url)
        #        print(str(e))
        pass
    if downloadableViaRequests:
        downloadViaRequests(doi, url, requestHeaders,
                            directory, wantedContentTypes)
#        print('downloading via requests')
    else:
        downloadViaUrlLib(doi, url, requestHeaders,
                          directory, wantedContentTypes)
#        print('downloading via urllib')


def downloadViaRequests(doi, url, requestHeaders, directory, wantedContentTypes=[]):
    try:
        content = requests.get(url, headers=requestHeaders,
                               allow_redirects=True)  # , stream=True)
        try:
            contentType = content.headers['content-type']
        except KeyError:
            contentType = None
        try:
            contentDisposition = content.headers['content-disposition']
        except KeyError:
            contentDisposition = None

        d = {'contentType': contentType,
             'contentDisposition': contentDisposition,
             'url': url
             }

        ext = decideExtension(d)
        if len(wantedContentTypes) == 0 or ext and ext in wantedContentTypes:
            content = content.content
            contentSize = int(sys.getsizeof(content))
            fileSize = 0
            fileName = createFile(directory, doi, ext)
#            fileName = os.path.join(directory, fileName)
            print(fileName)
            if os.path.exists(fileName):
                fileSize = os.path.getsize(fileName)
        #        print(contentSize, fileSize)
            if contentSize-33 > fileSize:
                # directory = os.path.dirname(os.path.abspath(fileName))
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                with open(fileName, 'wb') as f:
                    #            print('writingFile')
                    #            for chunk in content.iter_content(chunk_size=1024):
                    #                if chunk:
                    f.write(content)
            writecsvRow(os.path.join(directory, 'downloadable dois'), [
                        doi, url, ext], ['DOI', 'URL', 'EXT'])
    except (requests.exceptions.HTTPError, requests.exceptions.Timeout, requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
        writecsvRow(os.path.join(directory, 'bad urls'),
                    [doi, url, str(e)], ['DOI', 'URL', 'ERROR'])
    except:
        print(doi, url)
        raise


def downloadViaUrlLib(doi, url, requestHeaders, directory, wantedContentTypes=[]):
    try:
        req = Request(url)
        k, v = next(iter(requestHeaders.items()))
        req.add_header(k, v)
        urlibContent = urlopen(req)
        try:
            contentType = urlibContent.info()['Content-Type']
        except KeyError:
            contentType = None

        d = {'contentType': contentType,
             'url': url
             }
        ext = decideExtension(d)
        if len(wantedContentTypes) == 0 or ext in wantedContentTypes:
            content = urlibContent.read()
            contentSize = sys.getsizeof(content)
            fileSize = 0
            fileName = createFile(directory, doi, ext)
#            fileName = os.path.join(directory, fileName)

            print(fileName)
            if os.path.exists(fileName):
                fileSize = os.path.getsize(fileName)
    #            print(contentSize, fileSize)
            if contentSize-33 > fileSize:
                # directory = os.path.dirname(os.path.abspath(fileName))
                # if not os.path.exists(directory):
                #     os.makedirs(directory)

                with open(fileName, 'wb') as f:
                    #                print('writingFile')
                    #                for chunk in content.iter_content(chunk_size=1024):
                    #                    if chunk:
                    f.write(content)
            writecsvRow(os.path.join(directory, 'downloadable dois'), [
                        doi, url, ext], ['DOI', 'URL', 'EXT'])
    except (HTTPError, TimeoutError, URLError, ConnectionResetError, CertificateError) as e:
        writecsvRow(os.path.join(directory, 'bad urls'),
                    [doi, url, str(e)], ['DOI', 'URL', 'ERROR'])
    except:
        print(doi, url)
        raise


def getRelativityScore(links, seeds):
    numberOfSharedItems = len(set(links) & set(seeds))
    relativityScore = numberOfSharedItems/max(len(links), len(seeds))
    return relativityScore


def getStrengthRelativityScore(links, seeds):
    linksWithScores = defaultdict(set)
    if type(list(links.values())[0]) is not (list or set):
        for k, v in links.items():
            linksWithScores[k] = (v, 1)
        links = linksWithScores
    if type(seeds) is not dict:
        seeds = dict.fromkeys(seeds, 1)

    strengthsInSeed = allStrengths = 0
    for k, v in links.items():
        allStrengths += v[0]*v[1]
        if k in seeds.keys():
            strengthsInSeed += v[0]*v[1]

    relativityScore = strengthsInSeed/max(allStrengths, sum(seeds.values()))

#    sizeOfLinks = len(links)
#    sizeOfSeeds = len(seeds)
#    shared = set(links.keys()) & set(seeds.keys())
#    sizeOfShared = len(shared)
#    numberOfSharedItems = len(links & seeds)
#    if sizeOfShared == sizeOfSeeds or sizeOfLinks < sizeOfSeeds:
#        relativityScore = strengthsInSeed/sum(seeds.vlaues())
#    else:
#        relativityScore = strengthsInSeed/allStrengths

    return relativityScore


def isRelativelyRelated(id, seeds, threshold):
    handle = Entrez.elink(
        id=id, term='english[lang]', linkname='pubmed_pubmed')
    records = Entrez.read(handle)
    seeds = set(seeds)
    for record in records:
        links = set()
        for link in record["LinkSetDb"][0]["Link"]:
            links.add(link['Id'])

        relativityScore = getRelativityScore(links, seeds)
        if relativityScore >= threshold:
            return True
        else:
            return False


def getSeedsRelativeIds(seeds, csvFile):
    t = type(seeds)
    if t is str:
        csvFile = seeds
        seeds = getPMIDs('', seeds)
    getLinkedIds(seeds, csvFile+'id link')
    writeUniqueLinks(csvFile+'id link', 'temp')
    temp = getPMIDs('', 'temp')
    getLinkedIds(temp, 'temp id link')
    temp = 'temp id link'
    getRelativeIds(temp, seeds, csvFile+' scores')


def getRelativeIds(ids, seeds, csvFile):
    csvFile = csvFile.replace('.csv', '')
    t = type(ids)
    if t is str:
        ids = csvTodict(ids)
        t = type(ids)
    if t is dict:
        directory = os.path.dirname(os.path.abspath(csvFile))
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(csvFile+'.csv', 'a', **newlinearg) as out:
            csvout = csv.writer(out)
            for k, links in ids.items():
                relativityScore = getRelativityScore(links, seeds)
#                print(numberOfSharedItems, len(x), len(y),  relativityScore)
#                if relativityScore >= threshold:
                row = [k, relativityScore]
                csvout.writerow(row)
    if t is list:
        remainingIDs = list()
        batchSize = 5000
        last = len(ids)-1
        directory = os.path.dirname(os.path.abspath(csvFile))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(csvFile+'.csv', 'a', **newlinearg) as out:
            csv_out = csv.writer(out)
            if header:
                csv_out.writerow(('PMID', 'link', 'score'))
#                print(('PMID', 'link'))
                for start in range(0, len(ids), batchSize):
                    end = min(last, start+batchSize)
                    print("Downloading and writing record %i to %i" %
                          (start, end))
                    ids = list(ids)

                    attempt = 0
                    while attempt < 3:
                        attempt += 1
                        try:
                            handle = Entrez.elink(id=ids[start: end],
                                                  term='english[lang]',
                                                  linkname='pubmed_pubmed',
                                                  cmd='neighbor_score')
                        except HTTPError as e:
                            if 500 <= e.code <= 599:
                                print("Received error from server %s" % e)
                                print("Attempt %i of 3" % attempt)
                                time.sleep(15)
                            else:
                                raise

                    records = Entrez.read(handle)
                    for record in records:
                        links = record["LinkSetDb"][0]["Link"]
                        relativityScore = getRelativityScore(links, seeds)
                        row = [record['IdList'][0], relativityScore]
#                        print(row)
                        csv_out.writerow(row)


def getNumberOfRelativeIds(ids, seeds, threshold):
    csvFile = 'temp'
    getRelativeIds(ids, seeds, threshold, csvFile)
    relativeIds = getPMIDs('', csvFile)
    os.remove(csvFile + '.csv')
    return len(relativeIds)


def getXYOfRelativeIds(ids, seeds):
    d = dict()
    allIds = getNumberOfRelativeIds(ids, seeds, 0.0)
    for x in range(100):
        numberOfIds = getNumberOfRelativeIds(ids, seeds, x/100)
        y = int(numberOfIds/allIds*100)
        if y == 0:
            break
        d[x] = y
#        print(x)
    return d


def get_titles_from_endNote(handle):
    titles = list()
    clear = True
    parser = ET.iterparse(handle, events=('start', 'end'))
    for event, elem in parser:
        if event == 'start' and elem.tag == 'title':
            clear = False
        if event == 'end' and elem.tag == 'title':
            clear = True
            title = elem.find('style').text
            titles.append(title)
        if clear:
            elem.clear()  # discard the element
    return titles


def get_pmids_titles_from_edited_endNote(handle):
    refs = defaultdict(set)
    clear = True
    parser = ET.iterparse(handle, events=('start', 'end'))
    for event, elem in parser:
        if event == 'start' and elem.tag == 'record':
            clear = False
        if event == 'end' and elem.tag == 'record':
            clear = True
            try:
                title = elem.find('titles/title/style').text
                urls = elem.findall('urls/related-urls/url/style')
                pmid = 'NoId'
                for url in urls:
                    url = url.text
                    foundpmid = re.search('^\d+', url, re.MULTILINE)
                    if foundpmid and foundpmid.group():
                        pmid = foundpmid.group(0)
                refs[pmid] = title
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
        if clear:
            elem.clear()  # discard the element
    return refs


def get_titles_authors_from_endNote(handle):
    searchTerms = list()
    clear = True
    parser = ET.iterparse(handle, events=('start', 'end'))
    for event, elem in parser:
        if event == 'start' and elem.tag == 'record':
            clear = False
        if event == 'end' and elem.tag == 'record':
            clear = True
            title = elem.find('titles/title/style').text
            title = prepare_pubmed_search_term_from_title(title)

            authors = elem.findall('contributors/authors/author/style')
            listOfAuthors = ''
            firstAuthor = ''
            counter = 0
            for author in authors:
                author = author.text
                author = re.search('^\w+', author)
                if author and author.group():
                    author = author.group(0)
                    author = addAuthorTag(author)
                    listOfAuthors += author + ' '
                    counter += 1
                    if counter == 1:
                        firstAuthor = listOfAuthors

            pages = elem.find('pages/style')
            if pages:
                pages = re.search('^\d+', pages.text)
                if pages and pages.group():
                    pages = pages.group(0)
                    pages = pages+'[pagination]'
            if pages is None:
                pages = ''

            term = title + ' ' + firstAuthor + ' ' + pages
            term = term.strip()
            searchTerms.append(term)
        if clear:
            elem.clear()  # discard the element
    return searchTerms


def pmidTopmcid(pmidsListOrCSVfile, pmcidsCSVfile):
    # reading the pmids
    pmids = pmidsListOrCSVfile
    if type(pmids) is str:
        pmids = csvTolist(pmidsListOrCSVfile)
    try:
        if type(pmids[0]) is list and len(pmids[0]) == 2:
            pmids = [row[0] for row in pmids]
    except TypeError:
        pass
    # retreive pmcid for each pmid
    temp = pmids
    while list(temp):
        for pmid in list(pmids):
            attempt = 0
            while attempt < 3:
                attempt += 1
                try:
                    handle = Entrez.elink(dbfrom="pubmed", db="pmc",
                                          LinkName="pubmed_pmc", id=pmid)
                except HTTPError as e:
                    if 500 <= e.code <= 599:
                        print("Received error from server %s" % e)
                        print("Attempt %i of 3" % attempt)
                        time.sleep(15)
                    else:
                        raise

            result = Entrez.read(handle)
            pmcid = ''
            if len(result[0]['LinkSetDb']) == 1:
                pmcid = result[0]['LinkSetDb'][0]['Link'][0]['Id']
            elif not result[0]['LinkSetDb']:
                pmcid = 'NoPMCID'
            else:
                pmcid = result[0]['LinkSetDb'][0]['Link'][0]['Id'][0]
                pmcid += 'MoreThanPMCID'
                print('MoreThanPMCID')

            row = pmid, pmcid
            writecsvRow(pmcidsCSVfile, row, ['PMID', 'PMCID'])
            pmids.remove(pmid)
        temp = pmids


def pmcidToXml(pmcid, directory):
    url = 'https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:' + \
        str(pmcid) + '&metadataPrefix=pmc'
    page = urlopen(url)
    xml = page.read()
    fileName = os.path.join(
        directory, 'full text/pmcid/xml', str(pmcid) + '.xml')
    directory = os.path.dirname(os.path.abspath(fileName))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(fileName, 'wb') as f:
        f.write(xml)


def xmlToTxt(pmcid, directory):
    noTxt = False
    fileName = ''
#    if pmcid.isdigit():
    fileName = os.path.join(directory, 'full text/pmcid/xml/', str(pmcid))

    tree = ET.parse(fileName + '.xml')
    root = tree.getroot()
    rootNS, _ = root.tag.split('}')
    rootNS = rootNS[1:]
    nsmap = {'rootNS': rootNS}
    metadata = root.find('.//rootNS:metadata', nsmap)
    try:
        article = metadata[0]
    except TypeError:
        article = None
        noTxt = True
    if article:
        articleNS, temp = article.tag.split('}')
        assert temp == 'article', 'No article tag'
        del(temp)
        articleNS = articleNS[1:]
        nsmap['articleNS'] = articleNS
        body = article.find('.//articleNS:body', nsmap)
        if body:
            txt = ' '.join(ET.tostringlist(
                body, encoding='unicode', method='text'))
            fileName = os.path.join(
                directory, 'full text/pmcid/txt',  str(pmcid) + '.txt')
            directory = os.path.dirname(os.path.abspath(fileName))
            if not os.path.exists(directory):
                os.makedirs(directory)
            writeTxt(txt, fileName)
        else:
            noTxt = True

    return noTxt


def getFullText(pmcids, directory):
    noTxt = list()
    temp = pmcids
    while list(temp):
        for pmcid in list(pmcids):
            purePmcid = int(''.join(filter(str.isdigit, pmcid)))
            pmcidToXml(purePmcid, directory)
            if xmlToTxt(purePmcid, directory):
                noTxt.append(purePmcid)
                print(purePmcid)
            pmcids.remove(pmcid)
        temp = pmcids
    return noTxt


def pmidEntrezSummaryRecordTodoi(esummaryRecord):
    summaryRecord = esummaryRecord[0]
    pmid = summaryRecord['Id']
    try:
        doi = summaryRecord['DOI']
        if not doi:
            raise ValueError
#        print('doi')
    except (KeyError, ValueError):
        try:
            doi = summaryRecord['ArticleIds']['doi']
            if not doi:
                raise ValueError
#            print('article id doi')
        except (KeyError, ValueError):
            try:
                doi = summaryRecord['ELocationID'].strip('doi: ')
                if not doi:
                    raise ValueError
#                print('elocation id doi')
            except (KeyError, ValueError):
                #                print('noDoi')
                pass
    try:
        #        print(len(doi))
        if doi:
            return doi
        else:
            raise ValueError
    except NameError:
        raise ValueError


def pmidTodoi(pmidsListOrCSVfile, doisCSVfile):
    # reading the pmids
    #    pmidsListOrCSVfile = 'related scores'
    #    doisCSVfile = 'dois'
    pmids = pmidsListOrCSVfile
    if type(pmids) is str:
        pmids = pd.read_csv(pmids+'.csv')
        pmids = set(pmids['PMID'])
        pmids = {str(pmid) for pmid in pmids}

    # retreive doi for each pmid
    try:
        unicodeErrorpmids = set(csvTolist('unicode Error pmids'))
        notFoundpmids = set(csvTolist('not Found pmids'))
        notFounddois = set(csvTolist('not Found dois'))
        pmid_doi = csvTodict(doisCSVfile)
        dois = set(pmid_doi.keys())
    except FileNotFoundError:
        unicodeErrorpmids = set()
        notFoundpmids = set()
        notFounddois = set()
        pmid_doi = dict()
        dois = set()

    pmids = pmids - unicodeErrorpmids - notFoundpmids - notFounddois - dois
    unicodeErrorpmids = set()
    notFoundpmids = set()
    notFounddois = set()
    pmid_doi = dict()
    dois = set()
    print(len(pmids))
    print('fetching..')
    counter = 0
    for pmid in pmids:
        counter += 1
#        print(pmid, counter)

        attempt = 0
        while attempt < 3:
            attempt += 1
            try:
                summaryHandle = Entrez.esummary(db='pubmed', id=pmid)
            except (HTTPError, URLError) as e:
                print("Received error from server %s" % e)
                print("Attempt %i of 3" % attempt)
                time.sleep(15)

        try:
            summaryRecord = Entrez.read(summaryHandle)
            try:
                doi = pmidEntrezSummaryRecordTodoi(summaryRecord)
                pmid_doi[pmid] = extractDOI(doi).lower()
            except ValueError:
                notFounddois.add(pmid)
#                print('no doi', pmid, counter)
        except (UnicodeDecodeError, UnicodeEncodeError):
            unicodeErrorpmids.add(pmid)
            print('unicode', pmid, counter)
        except RuntimeError as e:
            notFoundpmid = re.findall('\d+', str(e))[0]
            assert pmid == notFoundpmid
            notFoundpmids.add(pmid)
            print('no record', pmid, counter)

        if counter % 1000 == 0 or counter == len(pmids):
            dictTocsv(pmid_doi, doisCSVfile, ['PMID', 'DOI'])
            listTocsvRows(unicodeErrorpmids, 'unicode Error pmids')
            listTocsvRows(notFoundpmids, 'not Found pmids')
            listTocsvRows(notFounddois, 'not Found dois')
            time.sleep(15)
            pmid_doi = dict()
            unicodeErrorpmids = set()
            notFoundpmids = set()
            notFounddois = set()
            print('woke up')


# counter=0
# for summaryHandle in summaryHandles:
#     counter+=1
#     fileName='temp'+ str(counter) +'.xml'
#     xml = summaryHandle.read().decode('cp1252').encode('utf-8')
#     print('xml: \n', xml)
# with open('xmlTem.xml','w') as f:
#     f.write(summaryHandles)
# print('wrote files')

#     for i in range(counter):
#         fileName='temp'+ str(i+1) +'.xml'
#         tree=ET.parse(fileName)
#         root=tree.getroot()
#         errors = root.findall('ERROR')
#         [root.remove(error) for error in errors]
#         outFileName= 'noErrors'+ str(i+1) +'.xml'
#         writeXMLWithHeaderFrom(fileName, outFileName, tree)
#     print('rewrote files')

#     summaryRecords = list()
#     for i in range(counter):
#         fileName='noErrors'+ str(i+1) +'.xml'
#         with open(fileName,'r') as f:
#             summaryRecords += Entrez.read(f)

    # print('dicting..')
    # print('filing..')

    # remainingIDs = set(pmids)
    # - set(map(int, pmid_doi.keys()))
    # - set(map(int, unicodeErrorpmids))
    # - set(map(int, notFoundpmids))

    # print(len(remainingIDs), len(pmids), len(pmid_doi.keys()))
    # return unicodeErrorpmids, notFoundpmids, remainingIDs


def writeXMLWithHeaderFrom(fileNameOfHeaderFile, fileNameOfOutFile, tree=None):
    if tree is None:
        tree = ET.parse(fileNameOfHeaderFile)
    root = tree.getroot()
    directory = os.path.dirname(os.path.abspath(fileNameOfOutFile))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(fileNameOfHeaderFile, 'r') as inf, open(fileNameOfOutFile, 'wb') as outf:
        for line in inf:
            if line.strip() == '<' + root.tag + '>':
                break
            outf.write(line.encode('utf-8'))
        tree.write(outf)


def fixSomeUnicodeChar(s):
    newS = ''
    flag = False
    for c in s:
        if flag and c.isspace():
            flag = False
            continue
        if ord(c) > 128:
            d = {
                '\ufb00': 'ff',
                '\ufb01': 'fi',
                '\ufb02': 'fl',
                '\ufb03': 'ffi',
                '\ufb04': 'ffl',
                '\ufb05': 'st',
                '\ufb06': 'ft'
            }
            c = switch(d, c)
            flag = True
        newS += c
    return newS


def cidToTxt(txt, mappingFile):
    with open(mappingFile, 'r') as f:
        cidcodes = f.readlines()

    cids = [line.replace('\n', '') for line in cidcodes]

    cidDict = dict()
    counter = 0
    for cid in cids:
        try:
            k = '(cid:' + str(counter) + ')'
            # Obtain the Unicode
            v = chr(int(cid, 16))
            cidDict[k] = v
        except ValueError:
            pass
        counter += 1

    return multireplace(txt, cidDict)


def cermine(directory):
    cmd = ['java', '-cp', 'cermine-impl-1.13-jar-with-dependencies.jar',
           'pl.edu.icm.cermine.ContentExtractor', '-path', directory]
    rerun = True
    while(rerun):
        try:
            r = sbp.check_output(cmd)
            rerun = False
        except sbp.CalledProcessError as e:
            rerun = True
            filesToMove = re.findall('.+\S+\.pdf', e.output)
            filesToMove = list(set(filesToMove))
            filesToMove = [f.replace('File processed: ', '')
                           for f in filesToMove]
            movedFiles = [f.replace(os.path.basename(
                directory), 'no cermine') for f in filesToMove]
            for f in movedFiles:
                d = os.path.dirname(os.path.abspath(f))
                if not os.path.exists(d):
                    os.makedirs(d)
            for source, destination in zip(filesToMove, movedFiles):
                if directory in source:
                    os.rename(source, destination)


def joinTxts(directory):
    files = os.listdir(directory)
    divisionLine = '\n___________________________________________________________________________________________________________________________________________\n'
    with open(os.path.join(directory, 'joined.txt'), 'w') as outf:
        for f in files:
            if os.path.splitext(f)[1] == '.txt':
                with open(os.path.join(directory, f), 'r') as inf:
                    outf.write(inf.read())
                    outf.write(divisionLine)


def pdftotext(pdfsOrDirectoryOfpdfs):
    if os.path.isdir(pdfsOrDirectoryOfpdfs):
        pdfFiles = os.listdir(pdfsOrDirectoryOfpdfs)
    elif os.path.isfile(pdfsOrDirectoryOfpdfs):
        pdfFiles = [pdfsOrDirectoryOfpdfs]
    elif pdfsOrDirectoryOfpdfs is list:
        pdfFiles = pdfsOrDirectoryOfpdfs

    notxt = set()

    for pdfFile in pdfFiles:
        pdfsDirectory = os.path.dirname(os.path.abspath(pdfFile))

        cmd = ['pdftotext', '-cfg', '.xpdfrc.txt', os.path.join(pdfsDirectory, pdfFile), os.path.join(
            pdfsDirectory, 'txt', pdfFile.replace('.pdf', '.txt'))]
        try:
            r = sbp.Popen(cmd).wait()
            if r > 0:
                notxt.add(pdfFile)
        except:
            notxt.add(pdfFile)

    dfNotxt = pd.DataFrame(list(notxt))
    dfNotxt.to_csv(pdfsDirectory+'noTxT.csv')


def passer(txt):
    return txt


def applyFunctionOnTxt(functionName, filesOrDirectoryOfFiles, *directoriesOfOutputsArgs):
    if os.path.isdir(filesOrDirectoryOfFiles):
        inputFiles = os.listdir(filesOrDirectoryOfFiles)
        # if not filesOrDirectoryOfFiles.endswith('\\'):
        #     filesOrDirectoryOfFiles = filesOrDirectoryOfFiles + '\\'
        inputFiles = [os.path.join(filesOrDirectoryOfFiles, f)
                      for f in inputFiles]
    elif os.path.isfile(filesOrDirectoryOfFiles):
        inputFiles = [filesOrDirectoryOfFiles]
    elif filesOrDirectoryOfFiles is list:
        inputFiles = filesOrDirectoryOfFiles

    for inputFile in inputFiles:
        inputsDirectory = os.path.dirname(os.path.abspath(inputFile))
        outputsDirectories = list()
        for directoryOfOutputs in directoriesOfOutputsArgs:
            if len(directoryOfOutputs) == 0:
                outputsDirectory = inputsDirectory
            else:
                outputsDirectory = os.path.join(
                    inputsDirectory, directoryOfOutputs)
            outputsDirectories.append(outputsDirectory)

        if os.path.isfile(os.path.join(inputsDirectory, os.path.basename(inputFile))) and os.path.splitext(inputFile)[1] == '.txt':
            txt = readTxt(os.path.join(inputsDirectory,
                                       os.path.basename(inputFile)))
            result = functionName(txt)
            if type(result) is str or not isinstance(result, Iterable):
                result = [result]

            if len(outputsDirectories) == 0:
                for txt in result:
                    writeTxt(txt, os.path.join(inputsDirectory,
                                               os.path.basename(inputFile)))
            else:
                for txt, outputsDirectory in zip(result, outputsDirectories):
                    lang = detectLang(txt)
                    try:
                        writeTxt(txt, os.path.join(outputsDirectory+'/'+lang,
                                                   os.path.basename(inputFile)))
                    except TypeError:
                        writeTxt(txt, os.path.join(outputsDirectory,
                                                   os.path.basename(inputFile)))


def applyFunctionWithParamsOnTxt(functionName, filesOrDirectoryOfFiles, *params):
    if os.path.isdir(filesOrDirectoryOfFiles):
        inputFiles = os.listdir(filesOrDirectoryOfFiles)
        # if not filesOrDirectoryOfFiles.endswith('\\'):
        #     filesOrDirectoryOfFiles = filesOrDirectoryOfFiles + '\\'
        inputFiles = [os.path.join(filesOrDirectoryOfFiles, f)
                      for f in inputFiles]
    elif os.path.isfile(filesOrDirectoryOfFiles):
        inputFiles = [filesOrDirectoryOfFiles]
    elif filesOrDirectoryOfFiles is list:
        inputFiles = filesOrDirectoryOfFiles

    for inputFile in inputFiles:
        inputsDirectory = os.path.dirname(os.path.abspath(inputFile))

        if os.path.isfile(os.path.join(inputsDirectory, os.path.basename(inputFile))) and os.path.splitext(inputFile)[1] == '.txt':
            txt = readTxt(os.path.join(inputsDirectory,
                                       os.path.basename(inputFile)))
            if params:
                result = functionName(txt, *params)
            else:
                result = functionName(txt)
            writeTxt(result, os.path.join(inputsDirectory,
                                          os.path.basename(inputFile)))


def applyFunctionOnXml(functionName, filesOrDirectoryOfFiles, *directoriesOfOutputsArgs):
    if os.path.isdir(filesOrDirectoryOfFiles):
        inputFiles = os.listdir(filesOrDirectoryOfFiles)
        # if not filesOrDirectoryOfFiles.endswith('\\'):
        #     filesOrDirectoryOfFiles = filesOrDirectoryOfFiles + '\\'
        inputFiles = [os.path.join(filesOrDirectoryOfFiles, f)
                      for f in inputFiles]
    elif os.path.isfile(filesOrDirectoryOfFiles):
        inputFiles = [filesOrDirectoryOfFiles]
    elif filesOrDirectoryOfFiles is list:
        inputFiles = filesOrDirectoryOfFiles

    for inputFile in inputFiles:
        inputsDirectory = os.path.dirname(os.path.abspath(inputFile))
        outputsDirectories = list()
        for directoryOfOutputs in directoriesOfOutputsArgs:
            if len(directoryOfOutputs) == 0:
                outputsDirectory = inputsDirectory
            else:
                outputsDirectory = os.path.join(
                    inputsDirectory, directoryOfOutputs)
            outputsDirectories.append(outputsDirectory)

        if os.path.isfile(os.path.join(inputsDirectory, os.path.basename(inputFile))) and os.path.splitext(inputFile)[1] in ('.xml', '.cermxml'):
            tree = ET.parse(os.path.join(inputsDirectory,
                                         os.path.basename(inputFile)))
            try:
                result = functionName(tree)
            except ValueError:
                print(inputFile)
                break
            if type(result) is str or not isinstance(result, Iterable):
                result = [result]
            if len(outputsDirectories) == 0:
                for txt in result:
                    writeTxt(txt, os.path.join(inputsDirectory,
                                               os.path.basename(inputFile)))
            else:
                for txt, outputsDirectory in zip(result, outputsDirectories):
                    if type(txt) is str:
                        try:
                            writeTxt(txt.decode('utf-8'), os.path.join(outputsDirectory, os.path.splitext(
                                os.path.basename(inputFile))[0]+'.txt'))
                        except AttributeError:
                            writeTxt(txt, os.path.join(outputsDirectory, os.path.splitext(
                                os.path.basename(inputFile))[0]+'.txt'))
                    elif not txt:
                        writecsvRow(os.path.join(outputsDirectory, 'noTxT'), [
                                    os.path.basename(inputFile)], headers=['File name'])


def applyFunctionOnXmlForParallel(functionName, inputFileOutputsDirectoriesTuple):
    inputFile = inputFileOutputsDirectoriesTuple[0]
    outputsDirectories = inputFileOutputsDirectoriesTuple[1]
    try:
        params = inputFileOutputsDirectoriesTuple[2]
    except:
        params = None

    inputsDirectory = os.path.dirname(os.path.abspath(inputFile))
    if os.path.isfile(os.path.join(inputsDirectory, os.path.basename(inputFile))) and os.path.splitext(inputFile)[1] in ('.xml', '.cermxml'):
        tree = ET.parse(os.path.join(inputsDirectory,
                                     os.path.basename(inputFile)))
        if params:
            try:
                result = functionName(tree, *params)
            except ValueError:
                print(inputFile)
        else:
            try:
                result = functionName(tree)
            except ValueError:
                print(inputFile)
        if type(result) is str or not isinstance(result, Iterable):
            result = [result]
            if len(outputsDirectories) == 0:
                for txt in result:
                    if type(txt) is str:
                        lang = detectLang(txt)
                        try:
                            writeTxt(txt.decode('utf-8'), os.path.join(inputsDirectory+'/'+lang, os.path.splitext(
                                os.path.basename(inputFile))[0]+'.txt'))
                        except AttributeError:
                            try:
                                writeTxt(txt, os.path.join(inputsDirectory, os.path.splitext(
                                    os.path.basename(inputFile))[0]+'.txt'))
                            except TypeError:
                                writeTxt(txt, os.path.join(inputsDirectory, os.path.splitext(
                                    os.path.basename(inputFile))[0]+'.txt'))
                        except TypeError:
                            writeTxt(txt.decode('utf-8'), os.path.join(inputsDirectory, os.path.splitext(
                                os.path.basename(inputFile))[0]+'.txt'))
                    elif not txt:
                        writecsvRow(os.path.join(inputsDirectory, 'noTxT'), [
                                    os.path.basename(inputFile)], headers=['File name'])
            else:
                for txt, outputsDirectory in zip(result, outputsDirectories):
                    if type(txt) is str:
                        lang = detectLang(txt)
                        try:
                            writeTxt(txt.decode('utf-8'), os.path.join(outputsDirectory+'/'+lang, os.path.splitext(
                                os.path.basename(inputFile))[0]+'.txt'))
                        except AttributeError:
                            try:
                                writeTxt(txt, os.path.join(outputsDirectory, os.path.splitext(
                                    os.path.basename(inputFile))[0]+'.txt'))
                            except TypeError:
                                writeTxt(txt, os.path.join(outputsDirectory, os.path.splitext(
                                    os.path.basename(inputFile))[0]+'.txt'))
                        except TypeError:
                            writeTxt(txt.decode('utf-8'), os.path.join(outputsDirectory, os.path.splitext(
                                os.path.basename(inputFile))[0]+'.txt'))
                    elif not txt:
                        writecsvRow(os.path.join(outputsDirectory, 'noTxT'), [
                                    os.path.basename(inputFile)], headers=['File name'])


def applyFunctionOnTxtForParallel(functionName, inputFileOutputsDirectoriesTuple):
    inputFile = inputFileOutputsDirectoriesTuple[0]
    outputsDirectories = inputFileOutputsDirectoriesTuple[1]
    try:
        params = inputFileOutputsDirectoriesTuple[2]
    except:
        params = None

    inputsDirectory = os.path.dirname(os.path.abspath(inputFile))
    if os.path.isfile(os.path.join(inputsDirectory, os.path.basename(inputFile))) and os.path.splitext(inputFile)[1] == '.txt':
        try:
            txt = readTxt(os.path.join(inputsDirectory,
                                       os.path.basename(inputFile)))
            if params:
                #                params = tuple(kwargs.values())
                result = functionName(txt, **params)
            else:
                result = functionName(txt)
            if type(result) is str or not isinstance(result, Iterable):
                result = [result]

            if len(outputsDirectories) == 0:
                for txt in result:
                    writeTxt(txt, os.path.join(inputsDirectory,
                                               os.path.basename(inputFile)))
            else:
                for txt, outputsDirectory in zip(result, outputsDirectories):
                    lang = detectLang(txt)
                    try:
                        writeTxt(txt, os.path.join(outputsDirectory+'/'+lang,
                                                   os.path.basename(inputFile)))
                    except TypeError:
                        writeTxt(txt, os.path.join(outputsDirectory,
                                                   os.path.basename(inputFile)))
        except MemoryError:
            with open(os.path.join(inputsDirectory,
                                   os.path.basename(inputFile)), 'rb') as f:
                txt = ''
                count = 0
                for line in f:
                    try:
                        txt += line.encode('utf-8')
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        txt += line.decode('latin1')
                    if count == 50000:
                        if params:
                            #                params = tuple(kwargs.values())
                            result = functionName(txt, **params)
                        else:
                            result = functionName(txt)
                        if type(result) is str or not isinstance(result, Iterable):
                            result = [result]

                        if len(outputsDirectories) == 0:
                            for txt in result:
                                writeTxt(txt, os.path.join(inputsDirectory,
                                                           os.path.basename(inputFile)))
                        else:
                            for txt, outputsDirectory in zip(result, outputsDirectories):
                                lang = detectLang(txt)
                                try:
                                    appendToTxt(txt, os.path.join(outputsDirectory+'/'+lang,
                                                                  os.path.basename(inputFile)))
                                except TypeError:
                                    appendToTxt(txt, os.path.join(outputsDirectory,
                                                                  os.path.basename(inputFile)))
                        count = 0
                        txt = ''


def parallellyApplyFunctionOnTxt(functionName, filesOrDirectoryOfFiles, *directoriesOfOutputsArgs, **kwargs):
    listOftuplesArgs = makeParallelargs(
        filesOrDirectoryOfFiles, *directoriesOfOutputsArgs)

    batchsize = max(1, int(len(listOftuplesArgs)/numberOfProcesses))
    with multiprocessing.Pool(numberOfProcesses) as p:
        if kwargs:
            kwargs = next(iter(kwargs.values()))
            argsAndkwargs = [(a, b, kwargs) for a, b in listOftuplesArgs]
            p.starmap(applyFunctionOnTxtForParallel, zip(
                repeat(functionName), argsAndkwargs), chunksize=batchsize)
        else:
            p.starmap(applyFunctionOnTxtForParallel, zip(
                repeat(functionName), listOftuplesArgs), chunksize=batchsize)


def parallellyApplyFunctionOnXml(functionName, filesOrDirectoryOfFiles, *directoriesOfOutputsArgs, **kwargs):
    listOftuplesArgs = makeParallelargs(
        filesOrDirectoryOfFiles, *directoriesOfOutputsArgs)

    batchsize = max(1, int(len(listOftuplesArgs)/numberOfProcesses))
    with multiprocessing.Pool(numberOfProcesses) as p:
        if kwargs:
            kwargs = next(iter(kwargs.values()))
            argsAndkwargs = [(a, b, kwargs) for a, b in listOftuplesArgs]
            p.starmap(applyFunctionOnXmlForParallel, zip(
                repeat(functionName), argsAndkwargs), chunksize=batchsize)
        else:
            p.starmap(applyFunctionOnXmlForParallel, zip(
                repeat(functionName), listOftuplesArgs), chunksize=batchsize)


def pmcXmlToTxt(tree):
    root = tree.getroot()
    rootNS, _ = root.tag.split('}')
    rootNS = rootNS[1:]
    nsmap = {'rootNS': rootNS}
    metadata = root.find('.//rootNS:metadata', nsmap)
    txt = ''
    try:
        article = metadata[0]
    except TypeError:
        article = None
    if article:
        articleNS, temp = article.tag.split('}')
        assert temp == 'article', 'No article tag'
        del(temp)
        articleNS = articleNS[1:]
        nsmap['articleNS'] = articleNS
        bodies = article.findall('.//articleNS:body', nsmap)
        if not bodies:
            bodies = article.findall('./articleNS:body', nsmap)
        if not bodies:
            bodies = article.findall(
                './/articleNS:sub-article/articleNS:body', nsmap)
        if not bodies:
            bodies = article.findall(
                './articleNS:sub-article/articleNS:body', nsmap)
        abstracts = article.findall(
            './/articleNS:front/articleNS:article-meta/articleNS:abstract', nsmap)
        if not abstracts:
            abstracts = article.findall(
                './articleNS:front/articleNS:article-meta/articleNS:abstract', nsmap)

        if abstracts:
            txt += ' '.join(ET.tostringlist(abstracts[0],
                                            encoding='unicode', method='text'))
        if bodies:
            for body in bodies:
                xrefsParents = body.findall('.//articleNS:xref/..', nsmap)

                for parent in xrefsParents:
                    xrefs = parent.findall('articleNS:xref', nsmap)
                    for xref in xrefs:
                        parent.remove(xref)
                txt += ' '.join(ET.tostringlist(body,
                                                encoding='unicode', method='text'))

    if not txt:
        return False
    else:
        return txt


def cermineXmlToTxt(tree):
    root = tree.getroot()
    abstracts = root.findall('.//abstract')
    bodies = root.findall('.//body')

    txt = ''
    if abstracts:
        txt += ' '.join(ET.tostringlist(abstracts[0],
                                        encoding='unicode', method='text'))
    if bodies:
        for body in bodies:
            xrefsParents = body.findall('.//xref/..')

            for parent in xrefsParents:
                xrefs = parent.findall('xref')
                for xref in xrefs:
                    parent.remove(xref)
            txt += ' '.join(ET.tostringlist(body,
                                            encoding='unicode', method='text'))

    if not txt:
        return False
    else:
        return fixSomeUnicodeChar(txt)


# def xmlToTxt(tree):
#     root = tree.getroot()
#     rootNS, _ = root.tag.split('}')
#     rootNS = rootNS [1 :]
#     nsmap = {'rootNS':rootNS}
#     metadata = root.find('.//rootNS:metadata', nsmap)
#     try:
#         article = metadata[0]
#     except TypeError:
#         article = None
#     if article:
#         articleNS, temp = article.tag.split('}')
#         assert temp == 'article', 'No article tag'
#         del(temp)
#         articleNS = articleNS[1 :]
#         nsmap['articleNS'] = articleNS
#         bodies = article.findall('.//articleNS:body', nsmap)
#         if not bodies:
#             bodies = article.findall('./articleNS:body', nsmap)
#         if not bodies:
#             bodies = article.findall('.//articleNS:sub-article/articleNS:body', nsmap)
#         if not bodies:
#             bodies = article.findall('./articleNS:sub-article/articleNS:body', nsmap)
#         abstracts = article.findall('.//articleNS:front/articleNS:article-meta/articleNS:abstract', nsmap)
#         if not abstracts:
#             abstracts = article.findall('./articleNS:front/articleNS:article-meta/articleNS:abstract', nsmap)

#         txt = ''
#         if abstracts:
#             txt += ' '.join(ET.tostringlist(abstracts[0], encoding='unicode', method='text'))
#         if bodies:
#             for body in bodies:
#                 xrefsParents = body.findall('.//articleNS:xref/..', nsmap)

#                 for parent in xrefsParents:
#                     xrefs = parent.findall('articleNS:xref', nsmap)
#                     for xref in xrefs:
#                         parent.remove(xref)
#                 txt += ' '.join(ET.tostringlist(body, encoding='unicode', method='text'))

#         if not txt:
#             return False
#         else:
#             return txt


def detectLang(txt):
    #     if len(txt) < 3:
    #         return 'None'
    # #    b = Detector(txt)
    #     b = TextBlob(txt)
    #     res = b.detect_language()
    #     return res

    DetectorFactory.seed = 0
    try:
        res = detect(txt)
        return res
    except langdetect.lang_detect_exception.LangDetectException:
        pass
#        print(txt)


def britishiseSpellings(txt, usuk_spellingsFile='USUK_spellings.csv'):
    usuk_spellingsFile = usuk_spellingsFile.replace('.csv', '')
    df = pd.read_csv(usuk_spellingsFile+'.csv')
    usUkSpellDict = dict(zip(list(df.US), list(df.UK)))
    txt = multireplace(txt, usUkSpellDict)
    return txt


def americaniseSpellings(txt, usuk_spellingsFile='USUK_spellings.csv'):
    usuk_spellingsFile = usuk_spellingsFile.replace('.csv', '')
    df = pd.read_csv(usuk_spellingsFile+'.csv')
    ukUsSpellDict = dict(zip(list(df.UK), list(df.US)))
    txt = multireplace(txt, ukUsSpellDict)
    return txt


def replaceAcronymsWithFullName(txt):
    txt = txt.replace('_', ' ')

    pairs = dict()
    sent_tokens = sent_tokenize(txt)
    del txt
    new_sent_tokens = list()
    for token in sent_tokens:
        newPairs = schwartz_hearst.extract_abbreviation_definition_pairs(
            doc_text=token)
        newPairs = {k: v.replace(' ', '_') for k, v in newPairs.items()}
        processedPairs = {v.replace('_', ' '): v for k, v in newPairs.items()}
        processedPairs.update(
            {v.replace('_', ' ')+' ' + v: v for k, v in newPairs.items()})
        processedPairs.update(
            {v.replace('_', ' ')+' (' + v + ')': v for k, v in newPairs.items()})
        processedPairs.update(
            {v+' (' + v + ')': v for k, v in newPairs.items()})

        newPairs.update(processedPairs)
        del processedPairs

        pairs.update(newPairs)
        if len(pairs) > 0:
            token = multireplace(token, pairs)
            token = multireplace(token, pairs)
        new_sent_tokens.append(token)

    txt = ' '.join(new_sent_tokens)

    return txt


def cleanText(txt):
    # replace ampresands by 'and'
    txt = re.sub(r'&', 'and', txt)

    patrns = [
        # remove apostrophe S "'s"
        r'(?i)\'s ',
        # remove email addresses
        r'([\w\.-]+)@([\w\.-]+)',
        # remove websites (as much as possible).
        r'((?i)f|ht)tp\S+',
        r'(?i)www\.\S+',
        r'(?i)\S+\.(com|org|net|edu|gov|ac)(\.\S\S)?',
        # only keep alphanumeric characters, dots, underscores and newline chars (\n)
        r'[^0-9A-Za-z_.\n]+'
    ]
    combined_pat = r'|'.join(map(r'(?:{})'.format, patrns))
    txt = re.sub(combined_pat, ' ', txt)

    # remove multispaces
    txt = [x for x in txt.split(' ') if x is not '']
    txt = ' '.join(txt).lower()

    # remove duplicate chars
    txt = removeDuplicateChars(txt)

    #    txt = britishiseSpellings(txt, 'USUK_spellings.csv')
    return txt


def numTokenize(txt):
    # NUM token for numbers (1,2,3,... one, two, ..)
    words = nltk.word_tokenize(txt)
    tagged = nltk.pos_tag(words)
    del txt, words
    # txt = ' '.join([w[0] if w[1] != 'CD' or '_' in w[0]
    #                 else 'NUM' for w in tagged])
    res = list()
    for w in tagged:
        if w[1] != 'CD':
            res.append(w[0])
        elif '_' in w[0]:
            temp = numTokenize(' '.join(w[0].split('_')))
            if temp == 'NUM ':
                res.append('NUM')
            else:
                res.append(w[0])
        else:
            res.append('NUM')
    res = ' '.join(r for r in res)
    res = re.sub(r'(NUM *)+', 'NUM ', res)
    return res


def lemmatise(txt):
    res = list()
    for w in txt.split():
        try:
            res.append(wn.synsets(w)[0].lemma_names()[0])
        except IndexError:
            res.append(w)
    return ' '.join(res)


def lemmatiseNoPhrases(txt):
    return tagLemmatize.tag_and_lem(txt)


def lemmatisePhrases(txtOrListOfPhrases):
    isList = False
    if type(txtOrListOfPhrases) is list:
        isList = True
        txtOrListOfPhrases = ' '.join(txtOrListOfPhrases)
    txt = txtOrListOfPhrases
    words = nltk.word_tokenize(txt)
    del txtOrListOfPhrases, txt
    res = list()
    for phrase in words:
        if '_' in phrase:
            txt = phrase.replace('_', ' ')
            phraseRes = tagLemmatize.tag_and_lem(txt).replace(' ', '_')
            res.append(phraseRes)
            del txt, phraseRes
        else:
            res.append(tagLemmatize.tag_and_lem(phrase))
    return res if isList else ' '.join(res)


def lemmatiseCSVDict(inFileName, outFileName):
    inFileName = inFileName.replace('.csv', '')+'.csv'
    outFileName = outFileName.replace('.csv', '')+'.csv'
    x = csvTodict(inFileName)
    resDict = dict()
    del x['terms']
    del x['term']
    for k, v in x.items():
        resDict[lemmatisePhrases(k)] = v
    dictTocsv(resDict, outFileName, headers=['Term', 'tf-idf'])


def terminePreprocess(txt):
    # termine ready text
    sents = sent_tokenize(txt)
    txt = '\n'.join(sents)
    del sents
    txt = txt.replace('.\n', '\n')
    txt = txt.replace('_', ' ')
    return txt


def phrasePreprocess(txt):
    # phrases ready text
    # only keep alphanumeric characters, underscores and new lines
    txt = re.sub(r'[^0-9A-Za-z_\n]+', ' ', txt)
    # remove multispaces
    txt = [x for x in txt.split(' ') if x is not '']
    txt = ' '.join(txt)
    return txt


def removeDuplicateWords(phrase):
    original = list()
    li = list()
    for word in phrase.split('_'):
        if word not in li:
            li.append(word)
        else:
            original.append(li)
            li = list()
            li.append(word)
    original.append(li)
    del phrase, li
    original = ['_'.join(li) for li in original]
    for i in range(len(original)-1, 0, -1):
        currentPh = original[i]
        previousPh = original[i-1]
        if currentPh in previousPh:
            original.remove(currentPh)
    return '_'.join(original)


def fixPhrasesWithDuplicateWords(txt):
    resTxt = ''
    for word in txt.split():
        if '_' in word:
            res = removeDuplicateWords(word)
        else:
            res = word
        resTxt += ' '+res
    return resTxt


def removeAndStripMultipleUnderscores(txt):
    res = re.sub('(_+)', '_', txt)
    res = re.sub('(^|\s+)_|_($|\s+)', '', res)
    return res


def correct(directory):
    parallellyApplyFunctionOnTxt(acoraMultireplace, directory, kwargs={
                                 'replacements': 'corrections.csv'})
    parallellyApplyFunctionOnTxt(acoraMultireplace, directory, kwargs={
                                 'replacements': 'spaced_prefixes.csv'})
    parallellyApplyFunctionOnTxt(acoraMultireplace, directory, kwargs={
                                 'replacements': 'corrections.csv'})
    parallellyApplyFunctionOnTxt(acoraMultireplace, directory, kwargs={
                                 'replacements': 'spaced_prefixes.csv'})
    parallellyApplyFunctionOnTxt(acoraMultireplace, directory, kwargs={
                                 'replacements': 'corrections.csv'})


def phrasePostProcess(txt):
    txt = fixPhrasesWithDuplicateWords(txt)
    txt = removeAndStripMultipleUnderscores(txt)
    # remove multispaces
    txt = [x for x in txt.split(' ') if x is not '']
    txt = ' '.join(txt)
    return txt


def preprocessText(txt):
    noAcronyms = replaceAcronymsWithFullName(txt)
    del txt
    clean = cleanText(noAcronyms)
    termine = terminePreprocess(clean)
    phrase = phrasePreprocess(clean)
    return noAcronyms, clean, termine, phrase


# def preprocessText(txt):
#     termine = terminePreprocess(cleanText(replaceAcronymsWithFullName(txt)))
#     return termine


def haveSemanticsInCommon(w1, w2):
    syn1 = set(wn.synsets(w1))
    syn2 = set(wn.synsets(w2))
    if len(syn1 & syn2) > 0:
        return True
    else:
        return False


def addUnderScoredPrefix(terms, prefixes):
    newUnderscoradded = dict()
    for prefix in prefixes:
        underscored = set()
        prefixed = set()
        underscoradded = dict()
        for t in terms:
            if prefix+'_' in t:
                underscored.add(t)
            elif prefix in t:
                prefixed.add(t)

        for t in prefixed:
            underscoradded[t] = t.replace(prefix, prefix+'_')

        if len(set(underscoradded.values()) & underscored) > 0:
            print(prefix, len(set(underscoradded.values()) & underscored))
            intersect = set(underscoradded.values()) & underscored
            newUnderscoraddedVersion = {k.replace(
                prefix+'_', prefix): underscoradded[k.replace(prefix+'_', prefix)] for k in intersect}
            newUnderscoradded.update(newUnderscoraddedVersion)
    dictTocsv(newUnderscoradded, 'prefixes.csv',
              headers=['prefixed', 'underscored'])
#        print(newUnderscoradded)


def splitSuffix(w):
    originalWord = w
    suffix = ''
    modifiedWord = originalWord
    # ending with s (e.g. plural)
    if modifiedWord.endswith('ies'):
        suffix = 'ies'+suffix
    elif modifiedWord.endswith('es'):
        suffix = 'es'+suffix
    elif modifiedWord.endswith('s'):
        suffix = 's'+suffix
    modifiedWord = rreplace(originalWord, suffix, '', 1)
    # ending with r
    if modifiedWord.endswith('ier'):
        suffix = 'ier'+suffix
    elif modifiedWord.endswith('er'):
        suffix = 'er'+suffix
    elif modifiedWord.endswith('r'):
        suffix = 'r'+suffix
    modifiedWord = rreplace(originalWord, suffix, '', 1)
    # ending with tion
    if modifiedWord.endswith('ization'):
        suffix = 'ization'+suffix
    elif modifiedWord.endswith('isation'):
        suffix = 'isation'+suffix
    elif modifiedWord.endswith('ation'):
        suffix = 'ation'+suffix
    elif modifiedWord.endswith('tion'):
        suffix = 'tion'+suffix
    elif modifiedWord.endswith('sion'):
        suffix = 'sion'+suffix
    elif modifiedWord.endswith('ion'):
        suffix = 'ion'+suffix
    modifiedWord = rreplace(originalWord, suffix, '', 1)
    # ending with ing
    if modifiedWord.endswith('ing'):
        suffix = 'ing'+suffix
    modifiedWord = rreplace(originalWord, suffix, '', 1)
    # ending with d (i.e. past tense)
    if modifiedWord.endswith('ied'):
        suffix = 'ied'+suffix
    elif modifiedWord.endswith('ed'):
        suffix = 'ed'+suffix
    elif modifiedWord.endswith('d'):
        suffix = 'd'+suffix
    modifiedWord = rreplace(originalWord, suffix, '', 1)

    if (modifiedWord.endswith('is') or modifiedWord.endswith('ys') or modifiedWord.endswith('iz') or modifiedWord.endswith('yz')) and suffix and suffix[0] == 'e':
        suffix = modifiedWord[-2:]+suffix
        modifiedWord = rreplace(originalWord, suffix, '', 1)
    elif (modifiedWord.endswith('enc') or modifiedWord.endswith('ogu')) and suffix and suffix[0] == 'e':
        suffix = modifiedWord[-3:]+suffix
        modifiedWord = rreplace(originalWord, suffix, '', 1)

    return modifiedWord, suffix


def predictAmericanSuffix(suffix):
    res = suffix.replace('ise', 'ize').replace('yse', 'yze').replace(
        'ence', 'ense').replace('ogue', 'og').replace('isation', 'ization')
    return res


def predictAmerican(word):
    word, suffix = splitSuffix(word)
    res = word + predictAmericanSuffix(suffix)
    if res.endswith('our') and len(res) > 4:
        res = res[: -3] + 'or'
    if res.endswith('ise'):
        res = res[: -3] + 'ize'
    if res.endswith('yse'):
        res = res[: -3] + 'yze'
    if res.endswith('ence'):
        res = res[: -4] + 'ense'
    if res.endswith('ogue'):
        res = res[: -4] + 'og'
    if 'all' in res[3:] or 'ell' in res[3:] or 'ill' in res[3:] or 'oll' in res[3:] or 'ull' in res[3:]:
        if not res.endswith('lly'):
            res = res.replace('ll', 'l')
    if 'oe' in res and len(res) > 4 or 'ae' in res and len(res) > 4:
        res = res.replace('ae', 'e').replace('oe', 'e')
    return res


def buildVocabulary(directory):
    vocab = set()
    files = os.listdir(directory)
    for f in files:
        if os.path.splitext(f)[1] == '.txt':
            with open(os.path.join(directory, f), 'r') as inf:
                txt = cleanText(inf.read()).replace('_', ' ')
                vocab |= set(re.split('[^a-zA-Z]+', txt))
    return list(vocab)


def buildVocabularyWithUnderscores(directory):
    vocab = set()
    files = os.listdir(directory)
    for f in files:
        if os.path.splitext(f)[1] == '.txt':
            with open(os.path.join(directory, f), 'r') as inf:
                txt = cleanText(inf.read())
                vocab |= set(re.split('[^a-zA-Z]+', txt))
    return list(vocab)


def build_us_uk_spellings_dicts(setOfWords, generalVocabFile='vocab.txt'):
    #    writeTxt(' '.join(setOfWords), 'vocab.txt')
    sp.setInputFile(generalVocabFile)
    us_uk_dict = dict()
    us_uk_dict_maybe = dict()
    # print(type(setOfWords))
    # time.sleep(15)
    for w in setOfWords:
        correct = sp.correctionFromKnown(predictAmerican(w))
        if w != correct and correct in setOfWords and w not in us_uk_dict_maybe and w not in us_uk_dict_maybe.values():
            # if w in ['etaiuvncrloeooelipotsgeanytva', 'orsecsoppeecstivfoerlyd', 'twoebreeailnsocraenaaslyezteod'] or correct in ['etaiuvncrleoelipotsgeanytva', 'orsecsoppeecstivferlyd', 'twebreeailnsocrenaaslyezteod']:
            #     print(w, correct)
            # time.sleep(3)
            for i, s in enumerate(difflib.ndiff(w, correct)):
                if s[-1] in 'aolu':
                    if s[0] == '-':
                        if haveSemanticsInCommon(w, correct) or len(w) > 5 or len(correct) > 5:
                            us_uk_dict[correct] = w
                        else:
                            us_uk_dict_maybe[correct] = w
                    elif s[0] == '+':
                        if haveSemanticsInCommon(w, correct) or len(w) > 5 or len(correct) > 5:
                            us_uk_dict[w] = correct
                        else:
                            us_uk_dict_maybe[w] = correct
                elif s[-1] in 'scz':
                    try:
                        if s[0] == '-' and s[-1] == 's' and correct[i] == 'z':
                            if haveSemanticsInCommon(w, correct) or len(w) > 5 or len(correct) > 5:
                                us_uk_dict[correct] = w
                            else:
                                us_uk_dict_maybe[correct] = w
                        elif s[0] == '-' and s[-1] == 's' and correct[i] == 'c':
                            if haveSemanticsInCommon(w, correct) or len(w) > 5 or len(correct) > 5:
                                us_uk_dict[correct] = w
                            else:
                                us_uk_dict_maybe[correct] = w
                        elif s[0] == '-' and s[-1] == 'c' and correct[i] == 's':
                            if haveSemanticsInCommon(w, correct) or len(w) > 5 or len(correct) > 5:
                                us_uk_dict[correct] = w
                            else:
                                us_uk_dict_maybe[correct] = w
                        elif s[0] == '-' and s[-1] == 'z' and correct[i] == 's':
                            if haveSemanticsInCommon(w, correct) or len(w) > 5 or len(correct) > 5:
                                us_uk_dict[correct] = w
                            else:
                                us_uk_dict_maybe[correct] = w
                    except IndexError:
                        print(s, i, w, correct)

    for k, v in us_uk_dict.copy().items():
        if not haveSemanticsInCommon(k, v) and len(wn.synsets(k)) > 0 and len(wn.synsets(v)) > 0:
            del us_uk_dict[k]

    us_uk_dict_with_spaces = {' '+k+' ': ' ' +
                              v+' ' for k, v in us_uk_dict.items()}
    us_uk_dict_maybe_with_spaces = {
        ' '+k+' ': ' '+v+' ' for k, v in us_uk_dict_maybe.items()}

    # if os.path.exists('vocab.txt'):
    #     os.remove('vocab.txt')
    return us_uk_dict_with_spaces, us_uk_dict_maybe_with_spaces


def filterErrorsInDict(original_dict):
    keys = set(original_dict.keys())
    values = set(original_dict.values())
    intersection = keys & values
    errors_dict = {original_dict.pop(
        item, None): item for item in intersection}
    return original_dict, errors_dict


def splitMashedWords(w):
    try:
        if len(wn.synsets(w)) > 0:
            return w
    except (nltk.corpus.reader.wordnet.WordNetError, AssertionError):
        return splitMashedWords(w)
    splited = wordninja.split(w)
    res = ''
    for spl in splited:
        try:
            if len(spl) < 3 or len(wn.synsets(spl)) < 1:
                res += spl
#                break
            else:
                res += ' ' + spl + ' '
        except (nltk.corpus.reader.wordnet.WordNetError, AssertionError):
            res = splitMashedWords(w)
            break
    return res.strip()


def build_split_wrong_concatenated_words_dict(setOfWords, generalVocabFile='wordninja_words.txt'):
    wrong_concatenated_words_dict = dict()
#    errors_dict = dict()
    generalVocab = set()
    generalVocab |= set(re.split('[^a-zA-Z]+', readTxt(generalVocabFile)))
    sp.setInputFile(generalVocabFile)
#    print(len(setOfWords))
    for w in setOfWords:
        correct = sp.correction(w)
        if w not in generalVocab and correct not in generalVocab:
            splited = splitMashedWords(w)
            if splited != w:
                wrong_concatenated_words_dict[w] = splited

    return wrong_concatenated_words_dict


def addToSpellingsDict(new_us_uk_dict, csvFileName='USUK_spellings.csv', headers=None):
    try:
        old_us_uk_dict = csvTodict(csvFileName)
    except FileNotFoundError:
        old_us_uk_dict = dict()
    for american, british in new_us_uk_dict.copy().items():
        if american in old_us_uk_dict:
            del new_us_uk_dict[american]
    dictTocsv(new_us_uk_dict, csvFileName, headers)


def isNounPhrase(term):
    isNounPh = True
    term = term.replace('_', ' ')
    if len(term) < 2 or term == '</s>':
        isNounPh = False
    words = nltk.word_tokenize(term)
    tagged = nltk.pos_tag(words, tagset='universal')
    if 'NOUN' not in [word[1] for word in tagged]:
        isNounPh = False
    return isNounPh


def getNounPhrases(listOfTuples):
    return [tup for tup in listOfTuples if isNounPhrase(tup[0])]


def writeIEERCorpus(fileName, xmlCorpus=nltk.corpus.ieer):
    for f in ieer.fileids():
        txt = ieer.raw(f)
        txt = txt.replace('<e_', '</b_')
        txt = re.sub(r'&.+;', ' ', txt)
        try:
            xml = ET.fromstring(txt)
        except Exception as e:
            print('Error in parsing: ', f, e)
        xmlLi = ET.tostringlist(xml, encoding='unicode', method='text')
        parsedTxt = ' '.join(xmlLi)
        appendToTxt(parsedTxt, fileName)


def buildPMCCorpus(term, size):
    directory = '../'+term+' corpus'
    pmids = getPMIDs(term, os.path.join(directory, 'pmids.csv'))
    pmcids = set()
    for count in range(int(len(pmids)/size)):
        pmidTopmcid(pmids[size*(count-1): size*count],
                    os.path.join(directory, 'pmcids'))
        d = csvTodict(os.path.join(directory, 'pmcids.csv'))
        pmcids = {d.pop(k) for k, v in d.copy().items()}
        try:
            pmcids.remove('PMCID')
        except KeyError:
            pass
        try:
            pmcids.remove('NoPMCID')
        except KeyError:
            pass
        noTxt = getFullText(pmcids, directory)
    listTocsvRows(noTxt, os.path.join(
        directory, 'full text/pmcid/txt/', 'noTxt'))
    print('remaining pmcids', len(pmcids))


def buildDOICorpus(term, size, contentTypes=[]):
    directory = '../'+term+' corpus'
    pmids = getPMIDs(
        term + ' AND english[Language]', os.path.join(directory, 'pmids.csv'))
    pmids = set(pmids[: size])
    try:
        # reading the DOIs
        dois = pd.read_csv(os.path.join(directory, 'dois.csv'))
        dois = set(dois['DOI'])
    except FileNotFoundError:
        dois = set()
        pmidTodoi(pmids, os.path.join(directory, 'dois'))
        dois = pd.read_csv(os.path.join(directory, 'dois.csv'))
        dois = set(dois['DOI'])
    directory = os.path.join(directory, 'full text/doi')
    # the variable "articles" is a list of dictionaries (a dictionary for each article). The dictionary is the result of querying habanero crossref using the article's doi. Here we are trying to load the variable "articles" if it was already pickled or start with an empty list.
    try:
        with open(os.path.join(directory, 'articles.pkl'), 'rb') as f:
            articles = pickle.load(f)
    except FileNotFoundError:
        articles = list()

    # some dois are not found in crossref. Here we are trying to read them or start with an empty set
    try:
        notFounddois = set(csvTolist(os.path.join(
            directory, 'dois notFoundin crossRef')))
    except FileNotFoundError:
        notFounddois = set()

    # querying crossref by the dois and saving the results
    # we wrap the main process in a while loop to resume the process if it was interrupted due to network errors.
    firstTime = True
    while(dois):
        # first we filter out the dois that were already processed
        collecteddois = set()
        for article in articles:
            collecteddois.add(article['message']['DOI'])
        collecteddois = {extractDOI(i).lower() for i in collecteddois}
        notFounddois = {extractDOI(i).lower() for i in notFounddois}

        dois = {i.lower() for i in dois}
        dois = dois - notFounddois - collecteddois
        if firstTime:
            lenOfIn = len(dois)
            firstTime = False
        elif len(dois) == lenOfIn:
            print('Remaining: ', len(dois), 'DOIs')
            break

        # This is the main process of querying crossref for the dois that are not processed yet
        for doi in dois:
            try:
                articles.append(cr.works(ids=doi))
            except (HTTPError, requests.exceptions.HTTPError, URLError, CertificateError, requests.exceptions.SSLError, TimeoutError) as e:
                if 'Not Found' in str(e):
                    notFounddois.add(doi)
                else:
                    print(doi)
                    raise e
    notFounddois = {extractDOI(i).lower() for i in notFounddois}
    listTocsvRows(notFounddois, os.path.join(
        directory, 'dois notFoundin crossRef'))

    # remove duplicate results
    temparticles = list()
    seendois = set()
    for article in articles:
        if article['message']['DOI'] not in seendois:
            temparticles.append(article)
            seendois.add(article['message']['DOI'])
    articles = temparticles
    del temparticles

    # pickle the variable "articles", due to its size and to the time (network time) needed to build it.
    with open(os.path.join(directory, 'articles.pkl'), 'wb') as f:
        pickle.dump(articles, f)
    with open(os.path.join(directory, 'articles.pkl'), 'rb') as f:
        articles = pickle.load(f)

    # Extracting urls that enable us to download full text articles. Some dois does not have a url in crossref
    articlesURLs = dict()
    doisWithNoLink = set()
    for article in articles:
        doi = article['message']['DOI']
        try:
            links = article['message']['link']
        except KeyError:
            doisWithNoLink.add(doi)
            continue
        urls = set()
        [urls.add((link['URL'], link['content-type'])) for link in links]
        if urls:
            articlesURLs[doi] = urls
        else:
            doisWithNoLink.add(doi)
    listTocsvRows(doisWithNoLink, os.path.join(directory, 'dois With No Link'))

    # Read dois and urls that are already downloaded if they exist. Also read urls that are known to be broken
    try:
        downloadabledois = set(pd.read_csv(os.path.join(directory,
                                                        'downloadable dois.csv'))['DOI'].dropna())
    except FileNotFoundError:
        downloadabledois = set()
    try:
        downloadableURLs = set(pd.read_csv(os.path.join(directory,
                                                        'downloadable dois.csv'))['URL'].dropna())
    except FileNotFoundError:
        downloadableURLs = set()
    try:
        badURLs = set(pd.read_csv(os.path.join(directory,
                                               'bad urls.csv'))['URL'].dropna())
    except FileNotFoundError:
        badURLs = set()


# click-through-token is obtained from ORCID and is saved as a dictionary pickle. It is passed as a header with the url requests to acquire access to full text articles that requires subscription or existence of this token for data mining purposes.
    try:
        with open('clickThroughToken.txt', 'r') as f:
            h = {'CR-Clickthrough-Client-Token': f.read().strip()}
    except FileNotFoundError:
        print('There is no XRef click through token')
        h = None

        # downloading the articles
        for doi, urls_types in articlesURLs.items():
            # if the article was available from cambridge core
            if '10.1017/s' in doi:
                url = getCambridgeURL(doi)
                if url and url not in downloadableURLs | badURLs:
                    download(doi, url, h, directory, contentTypes)
            else:
                for url_type in urls_types:
                    url, crContentType = url_type
                    if crContentType in contentTypes:
                        print(crContentType)

                        if url and url not in downloadableURLs | badURLs:
                            # springer links of the form "springerlink.com/..." need to be fixed
                            if 'springerlink' in url:
                                url = fixSpringerLinkURL(url)
                                if url and url in downloadableURLs | badURLs:
                                    continue
                            download(doi, url, h, directory, contentTypes)
        print('finish')

        # Retry (several times) downloading bad (non-downloaded) articles to avoid timeout and similar errors
        lenOfIn = 1
        lenOfOut = lenOfIn - 1
        try:
            while lenOfOut < lenOfIn:
                badArticles = pd.read_csv(
                    os.path.join(directory, 'bad urls.csv'))
                badArticles = badArticles[['DOI', 'URL']].dropna()
                badURLs = set(badArticles['URL'].dropna())
                downloadableURLs = set(pd.read_csv(os.path.join(directory,
                                                                'downloadable dois.csv'))['URL'].dropna())
                lenOfIn = len(badURLs - downloadableURLs)
                # downloading the articles
                for index, row in badArticles.iterrows():
                    doi = row['DOI']
                    url = row['URL']
                    if url == '###':
                        continue
                    # if the article was available from cambridge core
                    if '10.1017/s' in doi:
                        url = getCambridgeURL(doi)
                        if url and url not in downloadableURLs:
                            download(doi, url, h, directory, contentTypes)
                    elif url and url not in downloadableURLs:
                        # springer links of the form "springerlink.com/..." need to be fixed
                        if 'springerlink' in url:
                            url = fixSpringerLinkURL(url)
                            if url and url in downloadableURLs:
                                continue
                        download(doi, url, h, directory, contentTypes)
                print('finish')
                downloadableURLs = set(pd.read_csv(os.path.join(directory,
                                                                'downloadable dois.csv'))['URL'].dropna())
                lenOfOut = len(badURLs - downloadableURLs)
        except FileNotFoundError as e:
            if 'bad urls' in str(e):
                pass
            else:
                raise e
        print('Remaining: ', lenOfOut, 'IDs')

        # converting PDFs to txts after converting them to XMLs internally
        directory = os.path.join(directory, 'pdf')
        if os.path.exists(directory):
            cermine(directory)
            applyFunctionOnXml(cermineXmlToTxt, directory, 'txt')


def freqDistFor(corpusFilePaths):
    for corpusFilePath in corpusFilePaths:
        directory = os.path.dirname(os.path.abspath(corpusFilePath))+'/'
        corpusFilePath = corpusFilePath.replace(directory, '')
        corpus = PlaintextCorpusReader(directory, r'.*\.txt')
        try:
            with open(directory+'FreqDists/freqDist_'+corpusFilePath.replace('.txt', '.pkl'), 'rb') as f:
                fdist = pickle.load(f)
        except FileNotFoundError:
            if 'rehab' in corpusFilePath:
                fdist = FreqDist(corpus.words(corpusFilePath))
            else:
                fdist = FreqDist(corpus.words(corpusFilePath))
            writeTxt(pickle.dumps(fdist), directory +
                     '/FreqDists/freqDist_'+corpusFilePath.replace('.txt', '.pkl'))


def extractNonVocab(vocab):
    vocab = set(vocab)
    nonvocab = {i for i in vocab if len(i) > 30}
    vocab -= nonvocab
    return list(vocab), list(nonvocab)


def applyWord2phrase(directory, ngrams=5):
    pyCode = """
import os
import shutil
import word2vec as w2v

os.chdir('.')

ngrams = """ + str(ngrams) + """

files = [f for f in os.listdir('.') if os.path.isfile(
    f) and os.path.splitext(f)[1] == '.txt']
for f in files:
    if f.endswith('.txt'):
        f = f.replace('.txt', '')
        shutil.copy2(f + '.txt', f + '_original.txt')
        for i in range(ngrams - 1):
            infName = f + '_original.txt'
            outfName = f + '_phrases.txt'
            print(infName)
            with open(infName, 'rb') as fi:
                try:
                    txt = fi.read().decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    txt = fi.read().decode('latin1')
                except AttributeError:
                    txt = fi.read()
            txt = txt.split()
            numberOfWords = len(txt)
            vocabSize = len(set(txt))
            # minCount = int(numberOfWords/vocabSize)
            minCount = 7
            del txt
            w2v.word2phrase(infName, outfName,  min_count= minCount, verbose=True)
            os.remove(infName)
            # os.remove(f + '.txt')
            os.rename(outfName, infName)
    """
    writeTxt(pyCode, os.path.join(directory, 'w2p.py'))
    cmd = ['python3', os.path.join(os.path.abspath(directory), 'w2p.py')]
    curDir = os.getcwd()
    os.chdir((os.path.abspath(directory)))
    r = sbp.check_output(cmd)
    os.chdir(curDir)
    os.remove(os.path.join(directory, 'w2p.py'))
    phrasedDir = os.path.join(directory, 'phrased')
    if not os.path.exists(phrasedDir):
        os.makedirs(phrasedDir)
    for f in os.listdir(directory):
        if '_original' in f:
            newF = f.replace('_original', '')
            shutil.move(os.path.join(directory, f),
                        os.path.join(phrasedDir, newF))


# def applyWord2Vec(directory):
#     w2vDir = os.path.join(directory, 'w2v')
#     if not os.path.exists(w2vDir):
#         os.makedirs(w2vDir)
#     files = [f for f in os.listdir(
#         directory) if os.path.splitext(f)[1] == '.txt']
#     for f in files:
#         if f.endswith('.txt'):
#             txt = readTxt(os.path.join(directory, f)).split()
#             numberOfWords = len(txt)
#             vocabSize = len(set(txt))
#             minCount = int(numberOfWords/vocabSize)
#             del txt
#             w2v.word2vec(os.path.join(directory, f),
#                          os.path.join(w2vDir, f.replace('.txt', '.bin')), size=2000, verbose=True)
#             w2v.word2clusters(os.path.join(directory, f),
#                               os.path.join(w2vDir, f.replace('.txt', '_clusters.txt')), 500,  min_count=minCount, verbose=True)


def applyWord2Vec(txtFileName):
    directory = os.path.dirname(txtFileName)
    f = os.path.basename(txtFileName)
    w2vDir = os.path.join(directory, 'w2v')
    if not os.path.exists(w2vDir):
        os.makedirs(w2vDir)
    txt = readTxt(os.path.join(directory, f)).split()
    numberOfWords = len(txt)
    vocabSize = len(set(txt))
    # minCount = int(numberOfWords/vocabSize)
    minCount = 7
    del txt
    w2v.word2vec(os.path.join(directory, f),
                 os.path.join(w2vDir, f.replace('.txt', '.bin')), size=2000, verbose=True)
    w2v.word2clusters(os.path.join(directory, f),
                      os.path.join(w2vDir, f.replace('.txt', '_clusters.txt')), 500,  min_count=minCount, verbose=True)


def serialyApplyWord2Vec(directory):
    w2vDir = os.path.join(directory, 'w2v')
    if not os.path.exists(w2vDir):
        os.makedirs(w2vDir)
    files = [f for f in os.listdir(
        directory) if os.path.splitext(f)[1] == '.txt']
    files = [os.path.join(directory, f) for f in files]
    [applyWord2Vec(f) for f in files]


def parallellyApplyWord2Vec(directory):
    files = os.listdir(directory)
    files = [os.path.join(directory, f) for f in files]
    [applyWord2Vec(f) for f in files]


def applyTfidf(directory):
    #   files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.splitext(f)[1] == '.txt']
    vectorizer = TfidfVectorizer(
        input='filename',
        decode_error='ignore',
        strip_accents='ascii',
        analyzer='word',
        stop_words='english',
        # ngram_range=(1,5),
        # max_df=1.0,
        # min_df=1,
        sublinear_tf=True
    )

    files = [os.path.abspath(os.path.join(directory, f))
             for f in os.listdir(directory)
             if os.path.splitext(f)[1] == '.txt']
    tfidf = vectorizer
    tfidfMatrix = tfidf.fit_transform(files)
#    terms = tfidf.inverse_transform(tfidfMatrix)
    termsNames = tfidf.get_feature_names()
    tfidfDense = tfidfMatrix.todense()
    count = 0
    for doc in tfidfDense:
        f = os.path.basename(files[count]).replace('.txt', '.csv')
        fileName = os.path.abspath(os.path.join(directory, 'tf-idf', f))
        tfidf = tfidfDense[count].tolist()[0]
        termTfidf = [pair for pair in zip(
            range(0, len(tfidf)), tfidf) if pair[1] > 0]
        termTfidf = sorted(termTfidf, key=lambda t: t[1] * -1)
        termTfidf = [(termsNames[word_id], score)
                     for (word_id, score) in termTfidf]
        termTfidf = dict(termTfidf)
        dictTocsv(termTfidf, fileName, headers=['Term', 'tf-idf'])
        count += 1


def csvToTerms(tfIdfCsvFile):
    terms = csvTolist(tfIdfCsvFile)
    if len(terms[0]) > 1:
        justTerms = list()
        for t in terms:
            justTerms.append(t[0])
        terms = justTerms[1:]
        del justTerms
    return terms


def buildDAG(tfIdfCsvFile, graphFileName):
    graphFileName = graphFileName.replace('.graphml', '')+'.graphml'
    terms = csvToTerms(tfIdfCsvFile)
    print(len(terms), graphFileName)
    directory = os.path.dirname(os.path.abspath(graphFileName))
    if not os.path.exists(directory):
        os.makedirs(directory)
    graph = nx.DiGraph()
    graph.add_nodes_from(terms)
    # list(graph.in_degree())

    vertices = list(graph.nodes())

    for i in range(0, len(vertices)):
        for j in range(i+1, len(vertices)):
            t1 = vertices[i]
            t2 = vertices[j]
            splittedT1 = set(t1.split('_'))
            splittedT2 = set(t2.split('_'))
    #        print (splittedT1, splittedT2, splittedT1 in splittedT2, splittedT2 in splittedT1)
            if t1 in t2 and len(t1) > 4 and '_' not in t1:
                graph.add_edge(t1, t2)
            elif t2 in t1 and len(t2) > 4 and '_' not in t2:
                graph.add_edge(t2, t1)
            elif splittedT1.issubset(splittedT2):
                graph.add_edge(t1, t2)
            elif splittedT2.issubset(splittedT1):
                graph.add_edge(t2, t1)

    nx.readwrite.graphml.write_graphml(graph, graphFileName)


def termsToSubclasses(tfIdfCsvFile, graphFileName, w2vModel, w2vClusters, leftTerm1='', leftTerm2='', rightTerm1='', rightTerm2='', relation=''):
    terms = csvToTerms(tfIdfCsvFile)
    graph = nx.readwrite.graphml.read_graphml(graphFileName)

    model = w2v.load(w2vModel)
    clusters = w2v.load_clusters(w2vClusters)
    model.clusters = clusters
    vocab = model.vocab

    terms = [t for t in terms if t in vocab]

    directory = os.path.dirname(
        os.path.dirname(os.path.abspath(graphFileName)))
    directory = os.path.join(directory, 'taxonomy')
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not (leftTerm1 or leftTerm2 or rightTerm1 or rightTerm2 or relation):
        leftTerm1 = 'speech_language_therapy'
        rightTerm1 = 'therapy'
        leftTerm2 = 'acquired_brain_injury'
        rightTerm2 = 'brain_injury'
        relation = 'hasSubclass'

    if leftTerm1 in terms and rightTerm1 in terms and (leftTerm2 in terms or lemmatisePhrases(leftTerm2) in terms) and rightTerm2 in terms:
        print('yaaaaaaaaaaaaaaaaaaaaaay')
        term_subclasses_dict = dict()
#        i = -1
        for term in terms:
            #            i += 1
            pos = [term, leftTerm1]
            neg = [rightTerm1]
            if rightTerm1 in term:
                if leftTerm2 in terms:
                    pos = [term, leftTerm2]
                else:
                    pos = [term, lemmatisePhrases(leftTerm2)]
                neg = [rightTerm2]

            indexes, metrics = model.analogy(pos=pos, neg=neg)
            # if term in vocab:
            #     indexes, metrics = model.analogy(pos=pos, neg=neg)
            # else:
            #     print (i, term)
            #     continue
            try:
                response = getNounPhrases(
                    model.generate_response(indexes, metrics).tolist())
            except IndexError as e:
                boundary = int(str(e).split()[-1])
                inds = np.where(indexes >= boundary)
                indexes = np.delete(indexes, inds)
                metrics = np.delete(metrics, inds)
                response = getNounPhrases(
                    model.generate_response(indexes, metrics).tolist())
            clusterSet = set()
            for t, _, c in response:
                if t in set(graph.neighbors(term)):
                    clusterSet.add(c)
            subclasses = set()
            for t, _, c in response:
                if c in clusterSet:
                    subclasses.add(t)
            term_subclasses_dict[t] = subclasses

        dictTocsv(term_subclasses_dict, directory +
                  '/term_subclasses_dict.csv', headers=['term', relation])


def termSublcassesCSVtoGraph(directory):
    termSubclassesFileName = directory+'/taxonomy/term_subclasses_dict.csv'
    graphFileName = directory+'/graph/taxonomy.graphml'

    graph = nx.read_edgelist(termSubclassesFileName,
                             delimiter=',', create_using=nx.DiGraph, data=[('relation', str)])

    nx.readwrite.graphml.write_graphml(graph, graphFileName)
