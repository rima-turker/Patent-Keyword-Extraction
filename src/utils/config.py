from utils.stop_words import StopWords

class Config:
	sw = StopWords().sw_all
	n_gram_range_start = 1
	n_gram_range_end = 2