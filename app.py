import pandas as pd

from sklearn.model_selection import train_test_split
import time
from wordcloud import WordCloud, STOPWORDS
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Detecting Hate speech using Weak Supervision')
# st.info('Please Explore the data by displaying the wordcloud')
df_raw = pd.read_csv('labeled_data.csv')

df_raw = df_raw.rename(columns={'class':'label'})

df_raw = df_raw[['tweet','label']]

df_raw['tweet_id'] = df_raw.tweet.map(hash)

df_raw.loc[df_raw.label == 0,'label'] = 1

df_raw.loc[df_raw.label == 2,'label'] = 0

df_train, df_test = train_test_split(df_raw, train_size = 0.8,random_state=123)


df_train.label.value_counts()


comment_words = ''
stopwords = set(STOPWORDS)
 
# iterate through the csv file
for val in df_train.tweet:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "



wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
expander_ph = st.empty()
with expander_ph.expander('Click here for visualization'):
    if st.checkbox('Show Histogram of labels'):
        sns.countplot(df_train.label)
        st.pyplot()
    if st.checkbox('Display wordcloud'): 
        # plot the WordCloud image                      
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        
        plt.show()
        st.pyplot()


# Create our labeling fxn
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

# Define Constants
ABSTAIN = -1
NEUTRAL = 0
HATE = 1

offensive_words_input = st.text_area("Enter the hateful keywords")
if offensive_words_input == '':
    st.stop()
else:
    offensive_words = offensive_words_input.split('\n')


neutral_words_input = st.text_area("Enter the neutral keywords")
if neutral_words_input == '':
    st.stop()
else:
    neutral_words = neutral_words_input.split('\n')

@labeling_function()
def offensive_keywords(x,offensive_words=offensive_words):
    for word in offensive_words:
        if word in x.tweet.lower():
            return HATE
        else:
            return ABSTAIN

@labeling_function()
def neutral_keywords(x,neutral_words=neutral_words):
    for word in neutral_words:
        if word in x.tweet.lower():
            return NEUTRAL
        else:
            return ABSTAIN

@labeling_function()
def neutral_keywords_v2(x,neutral_words=neutral_words):
    for word in ['good']:
        if word in x.tweet.lower():
            return NEUTRAL
        else:
            return ABSTAIN

### Apply on Pandas
lfs = [offensive_keywords, neutral_keywords,neutral_keywords_v2]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

# Find percentage of dataset that was labels [Coverage]
coverage_hate_offensive, coverage_neutral, coverage_neutral_v2  = (L_train != ABSTAIN).mean(axis=0)
if st.button('Get LFs coverage'):
    st.error(f"Offensive keywords has the coverage of: {coverage_hate_offensive * 100:.1f}%")
    st.success(f"Neutral keywords has the coverage of: {coverage_neutral * 100:.1f}%")


from snorkel.labeling import LFAnalysis
LFAnalysis(L=L_train, lfs=lfs).lf_summary()

from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

Y_test = df_test.label.values

def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=10)
    plt.xlabel("Probability of HATE")
    plt.ylabel("Number of data points")
    plt.show()
    st.pyplot()


probs_train = label_model.predict_proba(L=L_train)


if st.button("Get generative model's performances"):
    majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
        "accuracy"
    ]
    st.warning(f"{'Majority Vote Model Accuracy:':<25} {majority_acc * 100:.1f}%")

    label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
        "accuracy"
    ]
    st.success(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
    plot_probabilities_histogram(probs_train[:, HATE])

# if not st.button('Proceed'):
#     st.stop()

from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.tweet.tolist())
X_test = vectorizer.transform(df_test.tweet.tolist())

from snorkel.utils import probs_to_preds

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)
if st.button('Fit a classifier'):
    st.warning('Please wait while the classifier is trained')
    time.sleep(2)
    st.info(f"Classifier got the Test Accuracy of: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")
