
## File Type 
- `FA` : it represents "Featured Articles" which is regarded as high-quality articles by Wikipedia editors.
- `GA` : it represents "Good Articles" which is also regarded as high-quality articles by Wikipedia editors, but lower level than FA
- `AA_#num` : it represents "All Articles". There's no article quality. Since the size of all articles are too big, it is divided into 10 parts. A number represents which part this file is.

## Column Description 
- `title` : An article's title
- `paragraph` : A paragraph name which a crawled context contained
- `image url` : An image url to access that image
- `caption` : A crawled caption  
- `context` : A most related context among all contexts in an articles 
- `page num` 
- `TFIDF` : TFIDF score for each word in a caption
- `TFIDF score` : we used TFIDF technique to get a most related context, and saved each TFIDF score
- `caption NER` : Named Entities in a caption
- `caption unigram` : unigrams in a caption 
- `context NER` : Named Entities in a context
- `context unigram` : unigrams in a context 

## File Number
|  Type |  # Pairs   | 
|:-----:|:----------:|
|  AA0  |    5039    |
|  AA1  |    27255   |
|  AA2  |    39899   |
|  AA3  |    19225   |
|  AA4  |    21934   |
|  AA5  |    21107   |
|  AA6  |    12369   |
|  AA7  |    9611    |
|  AA8  |    8515    |
|  AA9  |    33886   |
|  FA   |    2262    |
|  GA   |    4627    |
