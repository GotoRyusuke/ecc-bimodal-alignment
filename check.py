import pandas as pd
import os

os.chdir('F:/project_cctranscripts')

panel_cc = pd.read_parquet('E:/ECC Transcripts/Data_preambles/Data_preambles_2003-2021.parquet')
test_file = pd.read_parquet('E:/ECC Transcripts/Data_texts/2021/20210323-2485782-C.xml/content.parquet')