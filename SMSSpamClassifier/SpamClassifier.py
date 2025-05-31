import re
import nltk
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def clean_text(sms):
    sms.lower()
    sms = re.sub("[^a-z0-9]", ' ', sms)
    sms = nltk.word_tokenize(sms)
    sms = [t for t in sms if len(t) > 1]
    sms = [sn.stem(word) for word in sms if word not in stop]
    sms = ' '.join(sms)

    return sms

def wordCloud(data):
    word = ' '.join(data)
    wc = WordCloud(background_color='black')
    wc = wc.generate(word)

    plt.imshow(wc)
    plt.axis('off')
    plt.show()

file_path = r"C:\Users\Akshya\PycharmProjects\machine learning\.venv\SMSSpamClassifier\SMSSpamCollection"
data = pd.read_csv(file_path, sep='\t', names=['Label', 'Text'])

sn = SnowballStemmer("english") # what stemmer basically do is converting words into small parts for example running -> run and happier -> happi
stop = set(stopwords.words('english')) # stopwords contains all the words that are not meaningful like "is","are","the", etc
data['clean_text'] = data['Text'].apply(clean_text) # in this line the clean_text is getting added to the data with all the cleaned text
ham_data =data[data['Label'] == 'ham'] #
ham_data = ham_data['clean_text']
wordCloud(ham_data)
spam_data = data[data['Label']== 'spam']
spam_data = spam_data['clean_text']
wordCloud(spam_data)

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(data['clean_text']).toarray()
y = pd.get_dummies(data["Label"])
y = y["spam"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(train_score)
print(test_score)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(y)
print(X.shape)