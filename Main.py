from Mesh import *

# read seed set of PMIDS
seeds = getPMIDs('', 'seed2')
seeds = strToint(seeds)

# related articles to seed set and their strengths
getLinkedIds(seeds, 'seed2 id link')

# read PMIDs of related articles
related = pd.read_csv('seed2 id link.csv')
related = set(related['link'])
firstTime = True

# retreiving related articles of the related articles and their strengths
while related:
    lenOfIn = len(related)
    if firstTime:
        remaining = getLinkedIds(related, 'related id link')
        firstTime = False
    else:
        remaining = getLinkedIds(related, 'related id link', header=False)
    existing = pd.read_csv('related id link.csv')
    existing = set(existing['PMID'])
    related = pd.read_csv('seed2 id link.csv')
    related = set(related['link'])
    related = related - existing - set(map(int, remaining))
    lenOfOut = len(related)
    if lenOfIn == lenOfOut:
        print('Remaining: ' + len(remaining) + 'IDs')
        break

# normalising the strengths to be in the range of 0-1
fileName = 'related id link.csv'
df = pd.read_csv(fileName)
df['score'] = normalise(df['score'])
df.to_csv('normalised '+fileName, index=False, chunksize=100000)

# calculating and saving the strengthRelativityScore for each related pmid

# reading the related pmids
fileName = 'normalised related id link.csv'
df = pd.read_csv(fileName)
related = set(df['PMID'])
#related = list(related)[0]

# calculating the strengthRelativityScore for each related pmid
strengthRelativityScores = dict()
for pmid in related:
    linksDF = df[df.PMID == pmid][['link', 'score']]
    links = linksDF.set_index('link')['score'].to_dict()
    strengthRelativityScores[pmid] = getStrengthRelativityScore(links, seeds)

# saving the strengthRelativityScore for each related pmid
relatedScoresDF = pd.DataFrame.from_dict(
    strengthRelativityScores, orient='index')
relatedScoresDF.to_csv('related scores.csv',
                       header=['Strengrh relativity Score'],
                       index_label='PMID',
                       chunksize=100000)


# converting pmids to dois to use them for downloading full text articles
pmidTodoi('related scores', 'dois')

# reading the DOIs
dois = pd.read_csv('dois.csv')
dois = set(dois['DOI'])

# the variable "articles" is a list of dictionaries (a dictionary for each article). The dictionary is the result of querying habanero crossref using the article's doi. Here we are trying to load the variable "articles" if it was already pickled or start with an empty list.
try:
    with open('articles.pkl', 'rb') as f:
        articles = pickle.load(f)
except FileNotFoundError:
    articles = list()

# some dois are not found in crossref. Here we are trying to read them or start with an empty set
try:
    notFounddois = set(csvTolist('full text/doi/dois notFoundin crossRef'))
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
listTocsvRows(notFounddois, 'full text/doi/dois notFoundin crossRef')

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
with open('articles.pkl', 'wb') as f:
    pickle.dump(articles, f)
with open('articles.pkl', 'rb') as f:
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
listTocsvRows(doisWithNoLink, 'full text/doi/dois With No Link')

# apiArticlesURLs = dict()
# for doi, urls_types in articlesURLs.items():
#     for url_type in urls_types:
#         url, crContentType = url_type
#         if 'api.' in url:
#             apiArticlesURLs[doi] = articlesURLs[doi]

# elsevierdois = set()
# wileydois = set()
# for doi, urls_types in articlesURLs.items():
#     for url_type in urls_types:
#         url, crContentType = url_type
#         if 'api.elsevier' in url:
#             elsevierdois.add(doi)
#         elif 'api.wiley' in url:
#             wileydois.add(doi)
# listTocsvRows(elsevierdois, 'full text/doi/elsevier dois')
# listTocsvRows(wileydois, 'full text/doi/wiley dois')

# all(map(articlesURLs.pop, apiArticlesURLs))


# Read dois and urls that are already downloaded if they exist. Also read urls that are known to be broken
try:
    downloadabledois = set(pd.read_csv(
        'full text/doi/downloadable dois.csv')['DOI'].dropna())
except FileNotFoundError:
    downloadabledois = set()
try:
    downloadableURLs = set(pd.read_csv(
        'full text/doi/downloadable dois.csv')['URL'].dropna())
except FileNotFoundError:
    downloadableURLs = set()
try:
    badURLs = set(pd.read_csv('full text/doi/bad urls.csv')['URL'].dropna())
except FileNotFoundError:
    badURLs = set()

# click-through-token is obtained from ORCID and is saved as a dictionary pickle. It is passed as a header with the url requests to acquire access to full text articles that requires subscription or existence of this token for data mining purposes.
try:
    with open('clickThroughToken.txt', 'r') as f:
        h = {'CR-Clickthrough-Client-Token': f.read().strip()}
except FileNotFoundError:
    print('There is no XRef click through token')
    h = None

# for doi, urls_types in apiArticlesURLs.items():
#     for url_type in urls_types:
#         url, crContentType = url_type
#         if url and url not in downloadableURLs|badURLs:
#             download(doi, url, h)

# downloading the articles
for doi, urls_types in articlesURLs.items():
    # if the article was available from cambridge core
    if '10.1017/s' in doi:
        url = getCambridgeURL(doi)
        if url and url not in downloadableURLs | badURLs:
            download(doi, url, h)
    else:
        for url_type in urls_types:
            url, crContentType = url_type
            if url and url not in downloadableURLs | badURLs:
                # springer links of the form "springerlink.com/..." need to be fixed
                if 'springerlink' in url:
                    url = fixSpringerLinkURL(url)
                    if url and url in downloadableURLs | badURLs:
                        continue
                download(doi, url, h)
print('finish')

# Retry (several times) downloading bad (non-downloaded) articles to avoid timeout and similar errors
lenOfIn = 1
lenOfOut = lenOfIn - 1
while lenOfOut < lenOfIn:
    badArticles = pd.read_csv('full text/doi/bad urls.csv')
    badArticles = badArticles[['DOI', 'URL']].dropna()
    badURLs = set(badArticles['URL'].dropna())
    downloadableURLs = set(pd.read_csv(
        'full text/doi/downloadable dois.csv')['URL'].dropna())
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
                download(doi, url, h)
        elif url and url not in downloadableURLs:
            # springer links of the form "springerlink.com/..." need to be fixed
            if 'springerlink' in url:
                url = fixSpringerLinkURL(url)
                if url and url in downloadableURLs:
                    continue
            download(doi, url, h)
    print('finish')
    downloadableURLs = set(pd.read_csv(
        'full text/doi/downloadable dois.csv')['URL'].dropna())
    lenOfOut = len(badURLs - downloadableURLs)
print('Remaining: ', lenOfOut, 'IDs')

# converting PDFs to txts after converting them to XMLs internally
directory = 'full text/doi/pdf'
cermine(directory)
applyFunctionOnXml(cermineXmlToTxt, directory, 'txt')
# converting XMLs from PMC to txts
directory = 'full text/pmcid/xml'
applyFunctionOnXml(pmcXmlToTxt, directory, 'txt')


# # preprocessing the text files originated from pdfs (replaceAcronymsWithFullName, cleanText, seperate different languages,..)
# directory = 'full text/doi/pdf/txt'
# parallellyApplyFunctionOnTxt(preprocessText, directory, 'no acronyms',
#                              'clean', 'termine', 'phrase')
# print('finish')

# detecting languages
directory = '../english corpus'
parallellyApplyFunctionOnTxt(passer, directory, 'lang')

# joining all of the text in one file (representing our scope) and moving it to a corpus  folder
directory = 'full text/doi/pdf/txt/lang/en'
joinTxts(directory)
src = os.path.join(directory, 'joined.txt')
target = os.path.join('../corpus', 'rehab.txt')
shutil.move(src, target)

# directory = directory + '/phrase/en'
# joinTxts(directory)
# directory = directory.replace('/phrase/en', '/termine/en')
# joinTxts(directory)

# including texts from other domains in the corpus to help performing tf-idf
# nltk.download('europarl_raw')
# nltk.download('gutenberg')
# nltk.download('movie_reviews')
# nltk.download('brown')
# nltk.download('genesis')
# nltk.download('inaugural')
# nltk.download('ieer')
# parlimant = nltk.corpus.europarl_raw.english
# gutenberg = nltk.corpus.gutenberg
# movies = nltk.corpus.movie_reviews
# brown = nltk.corpus.brown
# genesis = nltk.corpus.genesis
# inagural = nltk.corpus.inaugural
# ieer = nltk.corpus.ieer
# writeTxt(parlimant.raw(), directory+'/parlimant.txt')
# writeTxt(gutenberg.raw(), directory+'/gutenberg.txt')
# writeTxt(genesis.raw('english-web.txt'), directory+'/genesis.txt')
# writeTxt(inagural.raw(), directory+'/inagural.txt')
# writeTxt(' '.join(brown.words()), directory+'/brown.txt')
# writeIEERCorpus(directory+'/ieer.txt')
# writeTxt(movies.raw(), directory+'/movies.txt')

# including texts from other domains in the corpus to help performing tf-idf
# terms = ['medicine not rehab', 'biology', 'chemistry']
# for term in terms:
#     buildPMCCorpus(term, 30000)
#     directory = '../'+term+' corpus'
#     directory = os.path.join(directory, 'full text/pmcid/txt')
#     # parallellyApplyFunctionOnTxt(preprocessText, directory, 'no acronyms',
#     #                          'clean', 'termine', 'phrase')
#     # directory = os.path.join(directory, 'termine/en')
#     joinTxts(directory)
#     src = os.path.join(directory, 'joined.txt')
#     target = os.path.join('../corpus', term+'.txt')
#     shutil.move(src, target)


# including texts from other domains in the corpus to help performing tf-idf
terms = ['medicine not rehab', 'biology', 'chemistry']
for term in terms:
    #    buildDOICorpus(term, 500, contentTypes=['pdf', 'unspecified'])
    directory = '../'+term+' corpus'
    directory = os.path.join(directory, 'full text/doi/pdf/txt')
    # parallellyApplyFunctionOnTxt(preprocessText, directory, 'no acronyms',
    #                          'clean', 'termine', 'phrase')
    # directory = os.path.join(directory, 'termine/en')
    if os.path.exists(directory):
        joinTxts(directory)
        src = os.path.join(directory, 'joined.txt')
        target = os.path.join('../corpus', term+'.txt')
        shutil.move(src, target)


# moving the joint texts representing the corpus iin scope to the corpus folder
# directory = directory + '/phrase/en'
# shutil.move('full text/doi/pdf/txt/phrase/en/joined.txt',
#             directory+'/rehab.txt')
# directory = directory.replace('/phrase/en', '/termine/en')
# shutil.move('full text/doi/pdf/txt/termine/en/joined.txt',
#             directory+'/rehab.txt')

# divide the corpus directory to ease parallellization and gain time
directory = '../corpus'
joinFilesByName(directory)
divideDirectory(directory, 5 * pow(10, 6))

# preprocessing the text files in the corpus
parallellyApplyFunctionOnTxt(preprocessText, directory, 'no acronyms',
                             'clean', 'termine', 'phrase')
print('finish')

# correct some mis-spelt words (e.g. finaly -> finally, fluense -> fluence)
subDirectories = ['no acronyms/en', 'clean/en', 'termine/en', 'phrase/en']
# rep = prepareReplacements('corrections.csv')
for sub in subDirectories:
    subDirectory = os.path.join(directory, sub)
    parallellyApplyFunctionOnTxt(multireplace, subDirectory, kwargs={
                                 'replacements': 'corrections.csv'})


# building the vocabulary of the corpus (a set of all words in the corpus)
directory = directory+'/termine/en'
# corpus = PlaintextCorpusReader(directory, r'.*\.txt')
vocab = buildVocabulary(directory)
writeTxt(' '.join(vocab), 'vocab.txt')

# fix wrongly concatenated words by building a dictionary e.g. 'ingoodmood': 'in good mood'
concat_split_dict = dict()
out = parallellize(build_split_wrong_concatenated_words_dict, vocab)
[concat_split_dict.update(o) for o in out]
addToSpellingsDict(concat_split_dict, 'concatenated_split_words.csv', [
                   'CONCATENATED', 'SPLIT'])
parallellyApplyFunctionOnTxt(
    multireplace, directory, kwargs={'replacements': 'concatenated_split_words.csv'})


# re-building the vocabulary
applyFunctionWithParamsOnTxt(
    multireplace, 'vocab.txt', 'concatenated_split_words.csv')
vocab = readTxt('vocab.txt').split()

# extracting non-words (words longer than 30 chars), usually appear because of converting pdfs to txts
out = parallellize(extractNonVocab, vocab)
vocab, nonvocab = list(), list()
for o in out:
    v, n = o
    vocab += v
    nonvocab += n

writeTxt(' '.join(nonvocab), 'non-vocab.txt')
writeTxt(' '.join(vocab), 'vocab.txt')

# building an American-British spelling mapping dictionary in a csv file
out = parallellize(build_us_uk_spellings_dicts, vocab)
us_uk_dict, us_uk_dict_maybe = dict(), dict()
for o in out:
    o0, o1 = o
    us_uk_dict.update(' '+o0+' ')
    us_uk_dict_maybe.update(' '+o1+' ')
us_uk_dict, errors_dict = filterErrorsInDict(us_uk_dict)

addToSpellingsDict(us_uk_dict, 'USUK_spellings.csv', ['US', 'UK'])
addToSpellingsDict(us_uk_dict_maybe, 'USUK_spellings_maybe.csv', ['US', 'UK'])
addToSpellingsDict(errors_dict, 'USUK_spellings_errors.csv',
                   ['ERROR', 'CORRECT'])


# a function to fix spelling errors twice (to fix the unfixed) then americanize
def fixSpellingAndAmericanize(txt):
    txt = multireplace(txt, 'concatenated_split_words.csv')
    print(txt)
    txt = multireplace(txt, 'corrections.csv')
    print(txt)
    txt = multireplace(txt, 'USUK_spellings_errors.csv')
    txt = multireplace(txt, 'USUK_spellings.csv')
    print(txt)
    corrections = csvTodict('corrections.csv')
    us_uk = csvTodict('USUK_spellings.csv')
    uk_us = {v: k for k, v in us_uk.items() if v not in corrections}
    txt = multireplace(txt, uk_us)
    print(txt)
    return txt


parallellyApplyFunctionOnTxt(fixSpellingAndAmericanize, directory)
#applyFunctionOnTxt(fixSpellingAndAmericanize, directory)
applyFunctionOnTxt(fixSpellingAndAmericanize, 'vocab.txt')


# re-building the vocabulary
vocab = buildVocabulary(directory)
writeTxt(' '.join(vocab), 'vocab.txt')

# preparing for word2phrase
parallellyApplyFunctionOnTxt(phrasePreprocess, directory, 'phrase')
#applyFunctionOnTxt(phrasePreprocess, directory, 'phrase')
directory += '/phrase/en'
# apply word2phrase to detect up to 20-gram phrases
joinFilesByName(directory)
applyWord2phrase(directory, ngrams=20)
fixDivisionLines(directory)
fixDivisionLines2(directory)
divideDirectory(directory, 50*pow(10, 6))

# fixing division lines and dividing files
directory += '/phrased'
fixDivisionLines(directory)
fixDivisionLines2(directory)
divideDirectory(directory, 50*pow(10, 6))
parallellyApplyFunctionOnTxt(phrasePostProcess, directory)


# NUM token for numbers (1,2,3,...,one,two,three,...), clean single chars, remove multispaces
def process(txt):
    txt = numTokenize(txt)
    txt = [x for x in txt.split(' ') if len(x) > 1]
    txt = ' '.join(txt)
#    txt = lemmatiseNoPhrases(txt)
    txt = [x for x in txt.split(' ') if x is not '']
    txt = ' '.join(txt)
    return txt


parallellyApplyFunctionOnTxt(process, directory)

# apply word2phrase to detect 20-gram phrases
joinFilesByName(directory)
applyWord2phrase(directory, ngrams=20)

# fixing division lines and dividing files
directory += '/phrased'
parallellyApplyFunctionOnTxt(correct, directory)
parallellyApplyFunctionOnTxt(phrasePostProcess, directory)
fixDivisionLines2(directory)
divideDirectory(directory, 50*pow(10, 6))

# lemmatise
parallellyApplyFunctionOnTxt(lemmatiseNoPhrases, directory, 'lemmatised')
directory += '/lemmatised/en'
parallellyApplyFunctionOnTxt(lemmatisePhrases, directory, 'lemmatised phrases')

directory += '/lemmatised phrases/en'
joinFilesByName(directory)
directory = directory.replace('/lemmatised phrases/en', '')
joinFilesByName(directory)
directory = directory.replace('/lemmatised/en', '')
joinFilesByName(directory)

directories = ['../corpus/termine/en/phrase/en/phrased/phrased',
               '../corpus/termine/en/phrase/en/phrased/phrased/lemmatised/en',
               '../corpus/termine/en/phrase/en/phrased/phrased/lemmatised/en/lemmatised phrases/en']

for directory in directories:
    applyTfidf(directory)
    serialyApplyWord2Vec(directory)
    terms = directory+'/tf-idf/rehab.csv'
    graph = directory+'/graph/rehab.graphml'
    w2vModel = directory+'/w2v/rehab.bin'
    w2vClusters = directory+'/w2v/rehab_clusters.txt'
    buildDAG(terms, graph)
    termsToSubclasses(terms, graph, w2vModel, w2vClusters)
    termSublcassesCSVtoGraph(directory)


majr = set(getPMIDs('rehab[majr]', 'rehab[majr]'))

x = getPMIDs('rehab[mesh]', 'rehab[mesh]')
mesh = set(getPMIDs('rehab[mesh]', 'rehab[mesh]'))
rehab = set(getPMIDs('rehab NOT rehab[mesh]rehab', 'rehab NOT rehab[mesh]'))

robquery = expandedquery = ''
rob = set(getPMIDs(robquery, 'rob'))
expanded = set(getPMIDs(expandedquery, 'expanded'))

expanded = expanded-rehab-mesh-majr
rehab = rehab-mesh-majr
mesh = mesh-majr

#getLinkedIds(list(rob), 'rob id link')
#writeUniqueLinks('rob id link', 'rob related')
# print('finish')
meshrelated = set(getPMIDs('', 'rehab[mesh] related'))
meshrelated = meshrelated - mesh
robrelated = set(getPMIDs('', 'rob related2'))
robrelated = robrelated - rob

#import csv
#import sqlite3
#writeUniqueLinks('rob id link', 'rob relate')

print('finish')


pmidTopmcid('seed1', 'pmcids')
pmidTopmcid('seed1 scores', 'pmcids')
print('finish')


d = csvTodict('pmcids')
pmcids = {v.pop() for k, v in d.items()}
pmcids.remove('PMCID')
pmcids.remove('NoPMCID')
noTxt = getFullText(pmcids)
listTocsv(noTxt, 'full text/txt/noTxt')
print('finish')
print(len(pmcids))
