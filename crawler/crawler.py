import requests
from bs4 import BeautifulSoup
import unidecode
import math

import re
import pandas as pd 
import sys
import PIL.Image as Image
import io 

from urllib.request import urlopen
import os 
import time
import pickle
import argparse
import numpy as np
import cv2
import itertools

import nltk
from nltk.corpus import stopwords

# def save_image(title, img, save_path):
#     for item in csv : 
#         img_dir = csv["href"]
#         os.systme("python script/extract_features.py --image_dir ")
#     img = np.array(img)
#     img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
#     path = os.path.join(save_path, title+'_0.npy')
#     feature_extractor = FeatureExtractor(img)
#     feature_extractor.extract_features()
#     np.save(unique_file(path,'.npy',),img,allow_pickle=True)

# def unique_file(basename, ext):
#     actualname = "%s.%s" % (basename, ext)
#     c = itertools.count()
#     while os.path.exists(actualname):
#         actualname = "%s_(%d).%s" % (basename, next(c), ext)
#     return actualname

def clean_parenthesis(data : "str"):
    return re.sub("[\[].*?[\]]", "", data)

def clean_text(data : "str") :
    # 1. reference 제거 , \xa0 제거 , \n 제거 
    data = re.sub("[\[].*?[\]]", "", data)
    data = re.sub('\n','',data) 
    data = re.sub('\xa0','',data)

    # 2. only ENG (영문자 이외 문자 공백)
    data = re.sub(r'[^a-zA-Z\u00c0-\u00FF0-9\s]', '', data)
    data = unidecode.unidecode(data) 

    # 3. lower
    data = data.lower()
    
    return data

def generate_nostop_ngrams(s, n):
    s = s.split()

    # elimiate stop words 
    stops = set(stopwords.words('english'))
    no_stopwords_list = [word for word in s if not word in stops]

    # stemming
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmer_words = [stemmer.stem(word) for word in no_stopwords_list]
    
    s = " ".join(stemmer_words)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != "" and len(token)!=1]
    tokens = list(set(tokens))

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def my_ner(document):
    capitals = r"((?:[A-Z][a-z]+ ?)+)"
    numbers = r"(?:\+|\-|\$)?\d{1,}(?:\,?\d{3})*(?:\.\d+)?%?"
    ne = set(map(str.strip,re.findall(capitals, document)))
    cd = set(re.findall(numbers, document))
    firstWords = set()
    for sentence in nltk.tokenize.sent_tokenize(document):
        word = sentence.split()[0]
        if not nltk.pos_tag([word])[0][1] in ['NNP','NNPS'] : firstWords.add(word)
    return (ne - firstWords) | cd

def computeTF(wordDict, bow): #각 문단에서 캡션과 겹치는 단어가 발생하는지 1 or 0  #각 문단에서 가지는 단어들 
    tfDict = {}
    bowCount = len(bow)
    if bowCount == 0 : 
      return {}
    else : 
      for word, count in wordDict.items():
          tfDict[word] = count/float(bowCount) # normalizing 이 되는 거 같은데 이게,, 일종의 문단 길이를 반영한다고도 볼 수 있을까??
      return tfDict # 각 문단에서의 캡션에 있는 단어의 frequency 를 나타냄 

def computeIDF(docList): # 각 문단별로 캡션에 있는 단어가 존재하는지 보는 1 or 0의 wordDict들 
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0) # 캡션에 존재하는 단어에 대한 Dict 
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1 # 모든 문단들을 보면서 전체 발생 빈도를 체크 
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val)+1)) # 분모의 +1 은 한번도 발생하지 않는 경우 대비  # 문단전체 / 단어발생비율 >> 단어 발생 횟수가 적을수록 큰 값
        
    return idfDict # 특정 캡션에 대한 IDF score dict 리턴

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf
    
def areSubString(captionSet: set, contextSet: set)-> bool:
    for word in captionSet: 
        if not any(((word in s) or (s in word)) for s in contextSet): return False
    return True

def extract_pairs(title, paragraph, imgs, captions, contexts, unigram_contexts, csv_dataset, save_path, page_num) :

  cleaned_captions = [] # for calculation
  related_contexts = [] # data (requiring return)
  related_idx = [] # data (requiring return)
  
  for id, item in enumerate(unigram_contexts):
    if not 15 <= len(item) <=150 :
      unigram_contexts[id].clear()

  for img, caption in zip(imgs, captions) : 
    if img and caption :
        caption_ner = my_ner(caption) 

        if len(caption_ner)>=1 : # NE>1 이면 
          context_ner_checker = [] #캡션별 NER이 포함된 cont만
          for context in contexts: # NER 을 가진 context 추출 
              if areSubString(caption_ner,my_ner(context)) :
                context_ner_checker.append(True)
              else:
                context_ner_checker.append(False)

          cleaned_caption = clean_text(caption)
          cleaned_captions.append(cleaned_caption)

          # unigram caption 
          uni_cap = generate_nostop_ngrams(cleaned_caption, 1)

          if len(uni_cap) >= 5:
            capSet = set(uni_cap)   
            
            contexts_TF_list = []
            for id in range(len(unigram_contexts)):
              contexts_TF_list.append(dict.fromkeys(capSet, 0))
              if unigram_contexts[id]:
                for word in unigram_contexts[id]:
                  if contexts_TF_list[id].get(word) is not None  : 
                    contexts_TF_list[id][word] +=1  

            computed_TF = [] # 각 문단별로 Term frequency 구하는 과정
            for tf, unigram_context in zip(contexts_TF_list, unigram_contexts):
              computed_TF.append(computeTF(tf, unigram_context))
            
            idf_score = computeIDF(contexts_TF_list) # 그 소문단에 대한 IDF 계산 

            TFIDF_list = [] # 각 문단 별로 TFIDF 가 계산된 list
            for tf in computed_TF : 
              TFIDF_list.append(computeTFIDF(tf, idf_score))

            max_id, max_score = -1, 0 
            for id, tfidf in enumerate(TFIDF_list) : 
              score = 0
              for val in tfidf.values():
                  score += val
              if score > max_score :
                max_score = score 
                max_id = id 

            related_context_NER = my_ner(contexts[max_id])
            NER_ratio = len(related_context_NER)/len(caption_ner)

            if context_ner_checker[max_id] and max_id >=0 :  # NER ratio deleted ! 
              
                info_dict = {}

                related_contexts.append(contexts[max_id])
                related_idx.append(max_id)


                info_dict["title"] = re.sub("[\[].*?[\]]", "", title)
                info_dict["paragraph"] = re.sub("[\[].*?[\]]", "", paragraph)
                info_dict["caption"] = caption
                info_dict["contexts"] = contexts[max_id]
                info_dict["image url"] = img['src']
                info_dict["page num"] = page_num

                info_dict["caption unigram"] = uni_cap
                info_dict["context unigram"] = unigram_contexts[max_id]    
                info_dict["caption NER"] = caption_ner 
                info_dict["context NER"] = related_context_NER
                info_dict["TFIDF"] = TFIDF_list[max_id]
                info_dict["TFIDF score"] = max_score 

                csv_dataset = csv_dataset.append(info_dict, ignore_index=True)

            else:
              related_idx.append(-1) # TF IDF가 0점인 경우  
              continue

          else : # caption 의 unigram size가 minimum 을 넘지 못할 때 
            related_contexts.append([])
            related_idx.append(-2) # caption의 unigram size < 5 

        else : # NER 없음 
          cleaned_captions.append([])
          related_contexts.append([])
          related_idx.append(-3)

    else : # img caption pair 가 True가 아닐 때 
      cleaned_captions.append([])
      related_contexts.append([])
      related_idx.append(-4)
  
  return csv_dataset

def get_partial_pairs(href) : 
  url = "https://en.wikipedia.org" + href

  response = requests.get(url)
  assert (response.status_code == 200)
  
  html = response.text
  soup = BeautifulSoup(html,'html.parser')
  output = soup.select_one('div.mw-parser-output')

  whole_pairs = []
  partial_imgs, partial_captions, partial_contexts = [], [], []

  elements = output.findAll(['div','p','h2'])
  H2 = output.select('h2')

  for elem in elements : 
    if elem.select("span.mw-headline"): # 소제목일 때  
        if H2 : 
            paragraph = H2.pop(0) 
        else : continue
        whole_pairs.append((partial_imgs, partial_captions, partial_contexts, paragraph.text))
        partial_imgs, partial_captions, partial_contexts = [], [], []
    elif elem.select("div.thumbinner"): # 이미지를 포함한 div 일 때 
        if elem.select_one("div.thumbinner > div.thumbcaption") :
            partial_imgs.append(elem.select_one("div.thumbinner > a.image > img.thumbimage"))
            partial_captions.append(elem.select_one("div.thumbinner > div.thumbcaption").get_text()) # caption text 만 따옴
    elif elem.name == 'p': # context 일 때
        partial_contexts.append(elem.get_text()) # text 만 따옴 
  
  return whole_pairs # imgs, captions, context, h2

def get_data_from_page(args, page, save_path, csv_dataset, page_num):
  title, href = page
  title = re.sub('/','',title)

  updated_csv_dataset = pd.DataFrame()

  whole_pairs = get_partial_pairs(href) 
  for th_para, (imgs, captions, contexts, paragraph) in enumerate(whole_pairs) : 

    if imgs and captions and contexts:
      captions = list(map(clean_parenthesis,captions))
      contexts = list(map(clean_parenthesis,contexts))
      cleaned_contexts = list(map(clean_text,contexts))

      # n-gram contexts 
      unigram_contexts = [] # for calculation
      for c in cleaned_contexts : 
        unigram_contexts.append(generate_nostop_ngrams(c, 1))

      updated_csv_dataset = extract_pairs(title, paragraph, imgs, captions, contexts, unigram_contexts, csv_dataset, save_path, page_num)
      # gen, ignored = download(imgs, captions, contexts, paragraph, related_idx, category, title, level, gen, ignored)

    else : 
      continue
  csv_save_path = save_path + '/' + args.fname

  if len(updated_csv_dataset)!= 0 : 
      updated_csv_dataset.to_csv(csv_save_path, index=True)
      print(f"{[page_num]}. {title} saved")
      return updated_csv_dataset
  else : 
      return csv_dataset


def get_next_page(url) : 
  response_ = requests.get(url)
  assert (response_.status_code == 200)
  html_ = response_.text 
  soup_ = BeautifulSoup(html_, 'html.parser')

  page_mover = soup_.select('div#mw-content-text > div.mw-allpages-nav > a')

  for item in page_mover : 
    if "Next page" in item.text : 
      return item['href'] 
    else : 
      continue 

  print("Failed to find a next page")
  return False

def check_part_range(part):
    if 1<=part<=10: return part
    raise argparse.ArgumentTypeError("%s is an int part value" % part)

def check_quality_range(quality):
    if quality in ['FA','GA','AA']:
        return quality
    raise argparse.ArgumentTypeError("%s is an int part value" % quality)

def load(part, quality):
    print("Page Loading...")

    if quality == 'AA':
        def get_all_articles(url) : 
            response_ = requests.get(url)
            assert (response_.status_code == 200)
            html_ = response_.text 
            soup_ = BeautifulSoup(html_, 'html.parser')
            good_lists = soup_.select('div#mw-content-text > div.mw-allpages-body > ul > li')
            return good_lists
            
        all_articles = []
        pages = []
        with open('./all_pages.pkl','rb') as r : 
            all_urls = pickle.load(r)

        # all_urls are including each urls which contains a bit of article lists 
        # the number of these urls is about 46000 
        # these urls needed to be divided into small parts 
        # basically, we divide it by 10 parts 
        # part = 10 

        partial_num = int(len(all_urls)/part)
        total_parts_list = [ all_urls[partial_num*i : partial_num*(i+1)] if i!= part-1 
                                else all_urls[partial_num*(i):] for i in range(part) ]
    
        for idx, each_part in enumerate(total_parts_list) : 
            with open(f'partial_list_{idx}.pkl', 'wb') as f : 
                pickle.dump(each_part, f)
                
        print(f"each part is downloaded!")

        assert len(total_parts_list) == part 
        print(f"yes, total_parts_list are including {part} lists") 
        
        return total_parts_list 

    elif quality == 'FA':
        featured_articles_wiki = "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles"
        response = requests.get(featured_articles_wiki)

        assert (response.status_code == 200)
        html = response.text 
        soup = BeautifulSoup(html, 'html.parser')
        featured_articles = soup.select('div.mw-parser-output > div.hlist > ul > li')

        pages = []
        for _, article in enumerate(featured_articles) : 
            pages.append((article.get_text(), article.a['href'])) # tuple -> (article title, article href) 
        print(f"{len(pages)} pages are ready to be crawled")

        if pages : 
            return pages
        else : 
            print(f"Page Load Error : pages are not loaded\nPlease check the html parser.")
            exit(100)

    elif quality == 'GA':
        def get_good_articles(url) : 
            response_ = requests.get(url)
            assert (response_.status_code == 200)
            html_ = response_.text 
            soup_ = BeautifulSoup(html_, 'html.parser')

            good_lists = soup_.select('div.wp-ga-topic > div.mw-collapsible > div.mw-collapsible-content > p > a')
            return good_lists

        good_articles_wiki = "https://en.wikipedia.org/wiki/Wikipedia:Good_articles/all"
        response = requests.get(good_articles_wiki)

        assert (response.status_code == 200)
        html = response.text 
        soup = BeautifulSoup(html, 'html.parser')

        previous_good_lists = soup.select('div.mw-parser-output > div.wp-ga-topic')[1:] # Contents 제외 
        topics_list = soup.select('div.mw-parser-output > div.wp-ga-summary > div.wp-ga-summary-topics > ul.wp-ga-summary-topics-list > li') # write all the sequences as argument ? 
        good_articles = []
        for each_topic in topics_list : 
            url = "https://en.wikipedia.org" + each_topic.a['href']
            good_articles.extend(get_good_articles(url))

        pages = []
        for _, article in enumerate(good_articles) : 
            pages.append((article.get_text(), article['href']))
        return pages

def get_all_articles(url) : 
    response_ = requests.get(url)
    assert (response_.status_code == 200)
    html_ = response_.text 
    soup_ = BeautifulSoup(html_, 'html.parser')

    all_lists = soup_.select('div#mw-content-text > div.mw-allpages-body > ul > li')
    return all_lists

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--part",
        type=int,
        required = False,
        help="part to crawl during AA"
    )
    parser.add_argument(
        "--quality",
        type=check_quality_range,
        required = True,
        help="part to crawl"
    )
    parser.add_argument(
      "--save",
      type=str,
      default= "./result",
      help="where will you save the csv dataset?"
    )
    parser.add_argument(
      "--load_csv",
      type=str,
      help="load a csv dataset to collect pairs continuously"
    )
    parser.add_argument(
      "--fname",
      type=str,
      default='data.csv',
      help="name to csv file that will be saved"
    )
    parser.add_argument(
      "--aa_pkl",
      type=str,
      default=None, 
      help= 'load a saved pkl to crawl data'
    )
    parser.add_argument(
      "--aa_pages_pkl",
      type=str,
      default=None, 
      help = "load a saved all_pages pkl to crwal data "
    )
    args = parser.parse_args()
    
    #pages = load(args.part, args.quality)
    save_path = args.save

    if args.load_csv : 
      csv_dataset = pd.read_csv(args.load_csv) 
      start_page =int(csv_dataset.iloc[-1]['page num']+1)
      print(f"crawling process will begin from {start_page}-th article from {args.load_csv}")
    else : 
      csv_dataset = pd.DataFrame()
      start_page = 0

    total, ig = 0, 0

    if args.quality == 'AA': 
        
        if args.aa_pkl is not None : 
            with open(args.aa_pkl, 'rb') as f : 
                a_list = pickle.load(f)
            total_parts_list = []
            total_parts_list.append(a_list)
            count = int(os.path.splitext(args.aa_pkl)[0][-1])
           # print(total_parts_list[0])
            print("a pkl length : ",len(total_parts_list[0]))

        elif args.aa_pages_pkl is not None :
            with open(args.aa_pages_pkl, 'rb') as f : 
                all_pages = pickle.load(f) 
            total_parts_list = []
            total_parts_list.append(all_pages)
            count = int(os.path.splitext(args.aa_pages_pkl)[0][-1])

        else :
            pages = load(args.part, args.quality)
            total_parts_list = pages 
            count = 0

        while total_parts_list : 
            a_partial_list = total_parts_list.pop(0)  
            all_articles = []
            iter_pages = []
            each_list_len = []
            
            if args.aa_pkl is not None : 
                print("Getting pages ...")
                for each_url in a_partial_list : 
                    #print(f"each url ; {each_url}")
                    all_articles.extend(get_all_articles(each_url))

                for article in all_articles :
                    iter_pages.append((article.get_text(), article.a['href']))
            
                print("We've got all pages for this partition") 

                with open(f"all_pages_for_partial_list{count}.pkl", 'wb') as f : 
                    pickle.dump(iter_pages, f)

                print("all pages for this partition are downloaded") 
            
            if args.aa_pages_pkl is not None : 
                iter_pages = a_partial_list
                print(f"A already existed all pages pkl for a partial list is loaded!")
            
            print(f"{count}-th's total pages : {len(iter_pages)}")
            print(f"{count}-th list started !")

            each_list_len.append(len(iter_pages))
            
            print("Crawling ...") 
            for page_num in range(start_page, len(iter_pages)):
                try :        
                    csv_dataset = get_data_from_page(args,iter_pages[page_num], save_path, csv_dataset, page_num)

                except Exception as e : 
                    print(f"Error : {e} In : {iter_pages[page_num]}")
            count += 1
            print(f"{count}-th list in total is done!") 

        print(f"crawling all AA articles is completed! Nice job!")
    
    else : # for FA and GA
        
        pages = load(args.part, args.quality)

        print("Crawling ...")
        for page_num in range(start_page, len(pages)):
            try :
                csv_dataset = get_data_from_page(args, pages[page_num], save_path, csv_dataset, page_num)
            except Exception as e : 
                # Too many req 대비책
                print(f"Error : {e} In : {pages[page_num]}")
    
        print(f"All {args.quality} pages are crawled!!!")

if __name__ == "__main__":
    main()
