import numpy as np
import collections
import pickle
import random
import sys, nltk, ast
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


def literal_eval(s):
    try:
        return ast.literal_eval(s)
    except:
        return s

def write_classes_to_file(df, f_name):
    lbe_kw = LabelEncoder()
    lbe_kw.fit_transform(df['KEYWORD'])
    
    f = open(f_name, "w")
    f.writelines(["%s\n" % item  for item in list(lbe_kw.classes_)])
    f.close()
    return lbe_kw.classes_

def get_list_ele_occ_more_than(dict_occ, th=500):
    list_ele=[]
    for i, (k, v) in enumerate(dict_occ.items()):
        if v>=th:
            list_ele.append(k)
    return list_ele

def get_list_ele_occ_less_than(dict_occ, th=5):
    list_ele=[]
    for i, (k, v) in enumerate(dict_occ.items()):
        if v<=th:
            list_ele.append(k)
    return list_ele

def get_list_ele_len_less_than(dict_occ, th=2):
    list_ele=[]
    for i, (k, v) in enumerate(dict_occ.items()):
        if len(k)<=th:
            list_ele.append(k)
    return list_ele


def get_list_pha_len_less(dict_occ, th=2):
    list_ele=[]
    for i, (k, v) in enumerate(dict_occ.items()):
        if len(k.split())<th:
            list_ele.append(k)
    return list_ele

def get_occurance_dict(dw_id_kw, columns):
	mylist = []
	for c in columns:
		mylist+=list(np.hstack(dw_id_kw[c].to_numpy()))
	counter=dict(collections.Counter(mylist))
	sorted_dict=dict(sorted(counter.items(), key=lambda item: item[1]))
	print('occurance dictionary len:', len(sorted_dict))
	return sorted_dict

def clean_lst(text):
    return list(text.replace("[", "").replace("]", "").replace("'", "").split(', '))
    
def get_one_column_as_flat_list(df, clm):
	tien_lst=[clean_lst(l) for l in df[clm].tolist()]
	tien_flat=[item for sublist in tien_lst for item in sublist]
	return tien_flat

def find_number_of_occurance(dw_id_kw):
    
	kws=np.hstack(dw_id_kw['KEYWORD'].to_numpy())
	counter=dict(collections.Counter(list(kws)))
	sorted_dict_n_1_wo_f=dict(sorted(counter.items(), key=lambda item: item[1]))
	print('len of unique kws:',len(sorted_dict_n_1_wo_f))
	return sorted_dict_n_1_wo_f

def remove_rows_contain_kw_in_list(list_stop_words, df_id_kw):
    count=0
    list_idx=[]
    list_found_stop_words=[]
    for i, row in df_id_kw.iterrows():
        kw=row['KEYWORD']
        if kw in list_stop_words:# or last_chars=="ing": #or sorted_dict.get(kw)<6:
            count+=1
            list_idx.append(i)
    print('removing rows from the df')
    return df_id_kw[~df_id_kw.index.isin(list_idx)]

def save_obj_as_txt(data, name):
	with open(str(name)+".txt", "w") as file:
		file.write(str(data))

def read_obj_as_txt(name):
	with open(str(name)+".txt", "w") as file:
		return eval(file.readline())

def save_obj(obj, name ):
    with open('/home/rtue/playground/jupyter-notebook/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
    #with open('/home/rtue/playground/jupyter-notebook/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def check_nan_rows(df):
	is_NaN = df.isnull()
	row_has_NaN = is_NaN.any(axis=1)
	rows_with_NaN = df[row_has_NaN]
	print('rows_with_NaN:',rows_with_NaN)

def read_dataset(path_dw_id_kw, path_dw_id_tit_abst_claim):
	data_id_kw = pd.read_csv(path_dw_id_kw,sep='\t')
	data_id_tit_abs_clm = pd.read_csv(path_dw_id_tit_abst_claim ,sep='\t')

	check_nan_rows(data_id_kw)
	check_nan_rows(data_id_tit_abs_clm)

	data_id_kw.dropna(subset = ["KEYWORD"], inplace=True)

	
	data_kw = pd.merge(data_id_kw,data_id_tit_abs_clm, on='PPID')
	print('data_kw is merged version and len(data_kw):',len(data_kw))

	if data_kw.isnull().values.any():
	    print('data_kw contains null element')
	return data_id_kw, data_id_tit_abs_clm, data_kw  

def remove_rows_contain(df, column, str_):
    #df[~df.C.str.contains("XYZ")]
    return df[df[column].str.contains(str_)==False]

def generate_word_index_freq(data_kw):
    old=np.empty([0, 0])
    for column in data_kw:
        old=np.concatenate((old, np.hstack(data_kw[column])), axis=None)
    return generate_word_index_dict(old)

def generate_word_index_dict(list_of_pharases, cutoff_for_rare_words=0):
		word_dict = dict()
		list_kw_no_space=[key.replace(" ", " ") for key in list_of_pharases]
		text=list_kw_no_space

		if len(text) > 1:
			flat_text = [item for item in text]
		else:
			flat_text = text
	    
	    # get word freuqncy
		fdist = nltk.FreqDist(flat_text)

	    # Convert to Pandas dataframe
		df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
		df_fdist.columns = ['Frequency']

	    # Sort by word frequency
		df_fdist.sort_values(by=['Frequency'], ascending=False, inplace=True)

	     # Add word index
		number_of_words = df_fdist.shape[0]
		df_fdist['word_index'] = list(np.arange(number_of_words)+1)

		# replace rare words with index zero
		frequency = df_fdist['Frequency'].values
		word_index = df_fdist['word_index'].values
		mask = frequency <= cutoff_for_rare_words
		word_index[mask] = 0
		df_fdist['word_index'] =  word_index

		# Convert pandas to dictionary
		word_dict = df_fdist['word_index'].to_dict()      

		print('dictionary length: ', str(len(word_dict)))
		print('list_kw_no_space: ', str(len(list_kw_no_space)))
		print('unique kws: ', str(len(set(list_kw_no_space))))
		print(list_kw_no_space[2])
		print(list_of_pharases[2])

		return word_dict
