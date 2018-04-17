import os
import sys
import re
import numpy as np
import csv
import time
from pycorenlp import StanfordCoreNLP

COMPLETE = False

sentiment = {'very positive': 5.0, 'positive': 4.0, 'neutral': 3.0, 'negative': 2.0, 'very negative': 1.0}

flag = False
corr_pred = 0
cnt = 0
nlp = StanfordCoreNLP('http://localhost:9000')
qid = 0
with open('cloze_test_test__spring2016 - cloze_test_ALL_test.csv', 'rb') as f:
    f_csv = csv.reader(f, delimiter=',')
    for line in f_csv:
        #print qid
        #qid += 1
        if flag:
            cnt += 1.0
            if COMPLETE:
                story = '. '.join( [ l.replace('.', '') for l in line[1:5] ] )
            else:
                story = line[4].replace('.', '')
            res = nlp.annotate(story, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 1000})
            total = 0.0
            for s in res["sentences"]:
                total += int(s["sentimentValue"])
            if COMPLETE:
                story_snt = total / 4.0
            else:
                story_snt = total / 1.0

            end1 = line[5]
            end2 = line[6]
            res = nlp.annotate(end1, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 1000})
            total = 0.0
            for s in res["sentences"]:
                total += int(s["sentimentValue"])
            end1_snt = total / 1.0
            res = nlp.annotate(end2, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 1000})
            total = 0.0
            for s in res["sentences"]:
                total += int(s["sentimentValue"])
            end2_snt = total / 1.0

            if abs( end1_snt - story_snt ) < abs( end2_snt - story_snt ):
                label = 1
            else:
                label = 2
            corr_label = int(line[7])
            print 'Correct Label', corr_label, 'Prediction', label
            if label == corr_label:
                corr_pred += 1
        else:
            flag = True
print 'Final Accuracy', corr_pred / cnt
