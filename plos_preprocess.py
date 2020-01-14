from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stopwords=set(stopwords.words('english'))
stopwords=[word for word in stopwords]
stopwords.append('et')
stopwords.append('al')
stopwords.append('github')
stopwords.append('https')
stopwords.append('www')
stopwords.append('com')
stopwords.append('dnn')
stopwords.append('cnn')
stopwords.append('survey')
stopwords.append('review')

##----------------------------------------------------------------------------
#Removes most punctuation while maintaining readability
def cleanformat(paper):
    paper=re.sub('\[[^\]\[]*\]', '', str(paper))
    paper=re.sub('<sub>', '_', paper)
    paper=re.sub('</sub>', '', paper)
    paper=re.sub('\s{1,}|-', ' ', paper)
    paper=re.sub('<title>.*</title>', '', paper)
    paper=re.sub('<[^<>]*>', ' ', paper)
    paper=re.sub('b\'', '', paper)
    paper=re.sub('n\'', '', paper)
    paper=re.sub('\\\\', '', paper)
    paper=re.sub('\'', '', paper)
    return paper.strip()

# Strips special characters from text and converts to lower case
def cleantext(paper):
    paper=re.sub('\[[^\]\[]*\]', '', str(paper))
    paper=re.sub('<sub>', '_', paper)
    paper=re.sub('</sub>', '', paper)
    paper=re.sub('\s{1,}|-', ' ', paper)
    paper=re.sub('<title>.*</title>', '', paper)
    paper=re.sub('<[^<>]*>', ' ', paper)
    paper=re.sub('[(~%)\-/,=:;+``#:.]', '', paper)
    paper=re.sub('b\'', '', paper)
    paper=re.sub('n\'', '', paper)
    paper=re.sub('\\\\', '', paper)
    paper=re.sub('\'', '', paper)
    return paper.lower().strip()

#Compiles all glossaries in a folder into a single glossary
#Note: Should look more closely at how repeat abbreviations are handled
def megaglossary(all_files):
    fullglossary={}
    for filepath in all_files:
        tree=ET.parse(filepath)
        glossary=getglossary(tree)
        for abbr in glossary:
            if not abbr in fullglossary:
                fullglossary[abbr]=glossary[abbr]
    return fullglossary

#Replaces abbreviations with the corresponding definition found in the glossary
def translatepaper(text, glossary):
    text=cleantext(text)
    tokens=nltk.word_tokenize(text)
    for i in range(len(tokens)-1, 0, -1):
        if tokens[i] in glossary:
            tokens[i]=glossary[tokens[i]]
    return re.sub('\s{1,}',' ', ' '.join(tokens))

#returns the glossary of the article
def getglossary(tree):
    root=tree.getroot()
    terms=root.findall("./back/glossary/def-list/")
    glossary={}
    for term in terms:
        raw_entry=str(ET.tostring(term, encoding='utf-8', method='xml'))
        try:
            abbr=cleantext(re.search('<term>.*</term>', raw_entry).group())
            definition=cleantext(re.search("<def>.*</def>", raw_entry).group())
            abbr=re.sub(' ','', abbr)
            glossary[abbr.lower()]=definition
        except:
            print('yikes')
    return glossary

#Returns the title of the article
def gettitle(tree):
    root=tree.getroot()
    title=cleanformat(ET.tostring(root.findall("./front//article-title")[0], encoding='utf-8', method='xml'))
    try:
        alt_title=cleanformat(ET.tostring(root.findall("./front//alt-title")[0], encoding='utf-8', method='xml'))
    except:
        alt_title=''
    return re.sub('\s{1,}', ' ', title + ': ' + alt_title)

#returns the abstract of the article
def getabs(tree):
    root=tree.getroot()
    abstracts=root.findall(".//abstract")
    abstract='';
    synopsis='';
    for section in abstracts:
        text=str(ET.tostring(section, encoding='utf-8', method='xml'))
        if 'abstract-type' in section.attrib:
            synopsis=text
        else:
            abstract=text
    return " ".join([abstract, synopsis])

#returns the body of the article
def getbody(tree):
    root=tree.getroot()
    sections=root.findall("./body/")
    body=''
    for section in sections:
        paper=ET.tostring(section, encoding="utf-8", method="xml")
        paper=cleantext(paper)
        body+=' '+paper
    return body.strip().lower()

#returns all titles and abstracts of articles found in the folder
def getallabs(path, max_files):
    allfiles = [''.join([path, f]) for f in listdir(path) if isfile(join(path, f))]
    if max_files<len(allfiles):
        allfiles= allfiles[:max_files]
    max_files=len(allfiles)
    titles=list()
    raw_abstracts=list()
    clean_abstracts=list()
    for i in range(max_files):
        tree=ET.parse(allfiles[i])
        titles.append(gettitle(tree))
        abstract=cleanformat(getabs(tree))
        if len(abstract)>0:
            glossary=getglossary(tree)
            cleaned=translatepaper(cleantext(abstract), glossary)
            raw_abstracts.append(abstract)
            clean_abstracts.append(cleaned)
    return titles, raw_abstracts, clean_abstracts

# path='./pcbi/'
# titles, raw_abstracts, clean_abstracts=getAllAbstractData(path)
# print(titles[1])
# print('-------------------------------')
# print(raw_abstracts[1])
# print('-------------------------------')
# print(clean_abstracts[1])
