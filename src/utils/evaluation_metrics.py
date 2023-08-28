import numpy as np
import collections
import pickle
import random
import sys, nltk
import pandas as pd

def precision_at_N(y_true, y_pred):
    if len(set(y_pred))==0:
        return 0
    p=len(set(y_pred) & set(y_true)) * 1.0 / len(set(y_pred))
    if p==0:
        return 0
    return p

def recall_at_N(y_true, y_pred, N):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)

def f_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)