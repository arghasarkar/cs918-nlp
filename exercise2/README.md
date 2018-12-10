# External files

The list here contains a list of external files that have been used for this project.

1) WordsWithStrength - A list of words along with a number indicating how positive and negative it is. 
```1``` means it's extremely positive. ```-1``` means it's extremely negative. Anything in between is less positive / negative.
This has been used for a naive lexicon classifier as a base.

Repository: https://github.com/hitesh915/sentimentstrength/

Direct link: https://raw.githubusercontent.com/hitesh915/sentimentstrength/master/wordwithStrength.txt

2) Use of glove for WordEmbeddings - This project uses glove for word embedding features. The embedding that's being used for this is the **Twitter** dataset.
The twitter dataset has 4 files in it. The one being used here is called **glove.twitter.27B.25d.txt**. 

**Usage instructions**
 - Download the glove zip file and unzip it. 
 - Copy the file **glove.twitter.27B.25d.txt** and place it in the same directory as this project. 

Link to project: https://nlp.stanford.edu/projects/glove/

Direct link to Glove Twitter's zip file: http://nlp.stanford.edu/data/glove.twitter.27B.zip

# Pickles

The SVM model has been provided as a pickle called ```my_svm_model.pkl```. If it is not provided, then the training time takes around 10 to 15 minutes. This file can also be renamed or deleted to enable training from scratch. 

# Source declarations

Sources for code or ideas found online or other sources are mentioned here:

1) Regex for matching Twitter's user mention was found in a StackOverflow post. 

```^(?!.*\bRT\b)(?:.+\s)?@\w+``` found on [Regex validation twitter mention.](https://stackoverflow.com/questions/7150652/regex-valid-twitter-mention) Date accessed: 6th December 2018.